use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use hf_hub::api::sync::Api;

use crate::error::LoadConfigError;
use std::path::{Path, PathBuf};

use super::error::{
    CosineSimilarityError, DownloadHFModelError, FastTokenBatchError, LoadSafeTensorError,
};

/// Load a json config file into some deserializable struct
pub fn load_config<T: for<'a> serde::Deserialize<'a>>(path: &Path) -> Result<T, LoadConfigError> {
    let json = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}

/// Calculate the cosine similarity between two float slices
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, CosineSimilarityError> {
    let a_len = a.len();
    let b_len = b.len();

    // check if vectors are zero lengthed
    if a_len == 0 || b_len == 0 {
        return Err(CosineSimilarityError::ZeroSizedVectorSimUndefined);
    }

    // check if the sizes are the same
    if a_len != b_len {
        return Err(CosineSimilarityError::DifferentLenVectorSimUndefined {
            lhs: a_len,
            rhs: b_len,
        });
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let sim = match (norm_a, norm_b) {
        (0.0, _) | (_, 0.0) => 0.0,
        _ => dot_product / (norm_a * norm_b),
    };

    Ok(sim)
}

/// memmap is unsafe because the file can be modified during or after
/// the file is read.
pub fn load_safetensors<'a, P: AsRef<std::path::Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, LoadSafeTensorError> {
    Ok(unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device) }?)
}

/// downloads a specific file from a specified HuggingFace Repo
pub fn download_hf_hub_file(
    model_id: &str,
    filename: &str,
) -> Result<PathBuf, DownloadHFModelError> {
    let api = Api::new()?;
    let model_repo = api.model(model_id.to_string());
    let filepath = model_repo.get(filename)?;
    Ok(filepath)
}

/// Below is the code for doing token-based batching.
/// Given input_ids, and some `batch_size`, we build batches such that
/// the max number of tokens doesn't exceed `batch_size`. We also make
/// sure that batches are in multiples of 8 for better GPU utilization.
///
/// The TokenBatchOutput has the original_ids field, which contains the
/// original location of the sentences before batching as well as the
/// sentence length which was used for sorted and batching.
///
/// Will refactor later.
pub struct Batch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub token_type_ids: Tensor,
}

pub struct TokenBatchOutput {
    pub batches: Vec<Batch>,
    pub original_ids: Vec<(usize, usize)>, // this is currently the idx and the sentence len / token cnt
}

pub fn fast_token_based_batching(
    token_ids: &mut Vec<Vec<u32>>,
    pad_token_id: usize,
    batch_size: usize,
    device: &Device,
) -> Result<TokenBatchOutput, FastTokenBatchError> {
    let mut finalized_batches = vec![];
    let pad_token_id = pad_token_id as u32;

    // sort the sentences based on length as if it was already paded to the first multiple-of-8th token
    let mut sentence_lens = token_ids
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let tok_len = 8 * ((t.len() + 7) / 8);
            (i, tok_len)
        })
        .collect::<Vec<(usize, usize)>>();
    sentence_lens.sort_by_key(|(_, len)| *len);

    // collate the data into batches that fill the batch_size
    let mut batches: Vec<Vec<usize>> = vec![];
    let mut current_batch: Vec<usize> = vec![];
    let mut current_token_count = 0;

    for (idx, tok_len) in sentence_lens.iter() {
        let idx = *idx;
        let tok_len = *tok_len;

        if current_token_count + tok_len > batch_size {
            batches.push(std::mem::take(&mut current_batch));
            current_batch.push(idx);
            current_token_count = tok_len;
        } else {
            current_batch.push(idx);
            current_token_count += tok_len;
        }
    }
    if current_batch.len() > 0 {
        batches.push(std::mem::take(&mut current_batch));
    }

    // for each batch run embeddings
    for b in batches.iter() {
        let mut current_token_ids = b
            .iter()
            .map(|idx| std::mem::take(&mut token_ids[*idx]))
            .collect::<Vec<Vec<u32>>>();

        let mut max_seq_length = current_token_ids.iter().map(|tok| tok.len()).max().unwrap();
        max_seq_length = 8 * ((max_seq_length + 7) / 8);

        let mut current_attn_mask = vec![];

        // pad tokens
        for i in 0..current_token_ids.len() {
            // attention mask first, since it depends on len of input_ids before padding
            let mut this_attn_mask = vec![1u32; current_token_ids[i].len()];
            this_attn_mask.resize(max_seq_length, 0u32);
            current_attn_mask.push(std::mem::take(&mut this_attn_mask));

            // pad token_ids
            current_token_ids[i].resize(max_seq_length, pad_token_id);
        }

        // convert input_ids to tensor
        let (a, b) = (current_token_ids.len(), max_seq_length);
        let current_token_ids = Tensor::from_vec(
            current_token_ids
                .into_iter()
                .flatten()
                .collect::<Vec<u32>>(),
            a * b,
            device,
        )?
        .reshape((a, b))?;

        // convert attn_mask to tensor
        let current_attn_mask = Tensor::from_vec(
            current_attn_mask
                .into_iter()
                .flatten()
                .collect::<Vec<u32>>(),
            a * b,
            device,
        )?
        .reshape((a, b))?;

        // create token_type_ids
        let current_token_type_ids = Tensor::zeros_like(&current_token_ids)?;

        finalized_batches.push(Batch {
            input_ids: current_token_ids,
            attention_mask: current_attn_mask,
            token_type_ids: current_token_type_ids,
        });
    }

    Ok(TokenBatchOutput {
        batches: finalized_batches,
        original_ids: sentence_lens,
    })
}
