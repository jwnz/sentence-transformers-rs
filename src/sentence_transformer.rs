use std::fmt;

use candle_core::Device;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use serde_json;
use tokenizers::{Tokenizer, TruncationParams};

use super::{
    config::{ModelConfig, SentenceBertConfig},
    dense::{Dense, DenseConfig},
    error::{EmbedError, SentenceTransformerBuilderError},
    normalize::Normalize,
    pooling::{PoolingConfig, PoolingStrategy},
    utils::{download_hf_hub_file, fast_token_based_batching, load_safetensors},
};

const DEFAULT_BATCH_SIZE: usize = 2048;
const DEFAULT_WITH_SAFETENSORS: bool = false;
const DEFAULT_WITH_NORMALIZE: bool = false;
const DEFAULT_PAD_TOKEN_ID: usize = 0;

pub enum Which {
    AllMiniLML6v2,
    AllMiniLML12v2,
    ParaphraseMiniLML6v2,
    ParaphraseMultilingualMiniLML12v2,
    LaBSE,
}

impl fmt::Display for Which {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Which::AllMiniLML6v2 => write!(f, "sentence-transformers/all-MiniLM-L6-v2"),
            Which::AllMiniLML12v2 => write!(f, "sentence-transformers/all-MiniLM-L12-v2"),
            Which::ParaphraseMiniLML6v2 => {
                write!(f, "sentence-transformers/paraphrase-MiniLM-L6-v2")
            }
            Which::ParaphraseMultilingualMiniLML12v2 => {
                write!(
                    f,
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            }
            Which::LaBSE => write!(f, "sentence-transformers/LaBSE"),
        }
    }
}

pub struct SentenceTransformerBuilder {
    model_id: String,
    with_safetensors: bool,
    with_normalization: bool,
    batch_size: usize,
    device: Option<Device>,
    pooling_path: Option<String>,
    dense_paths: Vec<String>,
}

impl SentenceTransformerBuilder {
    pub fn new(model_id: impl AsRef<str>) -> Self {
        Self {
            model_id: model_id.as_ref().to_string(),
            with_safetensors: DEFAULT_WITH_SAFETENSORS,
            with_normalization: DEFAULT_WITH_NORMALIZE,
            batch_size: DEFAULT_BATCH_SIZE,
            device: None,
            pooling_path: None,
            dense_paths: vec![],
        }
    }

    pub fn with_sentence_transformer(model: &Which) -> SentenceTransformerBuilder {
        match model {
            Which::LaBSE => SentenceTransformerBuilder::new(model.to_string())
                .with_safetensors()
                .with_normalization()
                .with_pooling("1_Pooling".to_string())
                .with_dense("2_Dense".to_string()),
            Which::ParaphraseMultilingualMiniLML12v2 | Which::ParaphraseMiniLML6v2 => {
                SentenceTransformerBuilder::new(model.to_string())
                    .with_safetensors()
                    .with_pooling("1_Pooling".to_string())
            }
            Which::AllMiniLML6v2 | Which::AllMiniLML12v2 => {
                SentenceTransformerBuilder::new(model.to_string())
                    .with_safetensors()
                    .with_normalization()
                    .with_pooling("1_Pooling".to_string())
            }
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> SentenceTransformerBuilder {
        self.batch_size = batch_size;
        self
    }

    pub fn with_safetensors(mut self) -> SentenceTransformerBuilder {
        self.with_safetensors = true;
        self
    }

    pub fn with_normalization(mut self) -> SentenceTransformerBuilder {
        self.with_normalization = true;
        self
    }

    pub fn with_device(mut self, device: &Device) -> SentenceTransformerBuilder {
        self.device = Some(device.clone());
        self
    }

    pub fn with_pooling(mut self, pooling: String) -> SentenceTransformerBuilder {
        self.pooling_path = Some(pooling);
        self
    }

    pub fn with_dense(mut self, dense_path: String) -> SentenceTransformerBuilder {
        self.dense_paths.push(dense_path);
        self
    }

    pub fn build(self) -> Result<SentenceTransformer, SentenceTransformerBuilderError> {
        // Device must be specified
        let device = self
            .device
            .ok_or_else(|| SentenceTransformerBuilderError::DeviceNotSpecified)?;

        // The pooling method must also be specified
        let pooling_method = self
            .pooling_path
            .ok_or_else(|| SentenceTransformerBuilderError::PoolingMethodNotSpecified)?;
        let pooling_method = format!("{pooling_method}/config.json");

        // load the model's hf_hub repo config
        let config_filename = download_hf_hub_file(&self.model_id, "config.json")?;
        let model_config =
            serde_json::from_str::<ModelConfig>(&std::fs::read_to_string(&config_filename)?)?;

        // Load the transformer model config
        let config = serde_json::from_str::<Config>(&std::fs::read_to_string(&config_filename)?)?;

        // Load the sbert config
        let sbert_config_filename =
            download_hf_hub_file(&self.model_id, "sentence_bert_config.json")?;
        let sbert_config = serde_json::from_str::<SentenceBertConfig>(&std::fs::read_to_string(
            &sbert_config_filename,
        )?)?;

        // Load the Tokenizer
        let tokenizer_filename = download_hf_hub_file(&self.model_id, "tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: sbert_config.max_seq_length,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            direction: tokenizers::TruncationDirection::Right,
            stride: 0,
        }))?;
        tokenizer.with_padding(None);

        // Load the transformer
        let vb = if self.with_safetensors {
            let weights_filename = download_hf_hub_file(&self.model_id, "model.safetensors")?;
            load_safetensors(&[weights_filename], DTYPE, &device)?
        } else {
            let weights_filename = download_hf_hub_file(&self.model_id, "pytorch_model.bin")?;
            candle_nn::VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        };
        let bert = BertModel::load(vb, &config)?;

        // load the pooler
        let pooling_config_filename = download_hf_hub_file(&self.model_id, &pooling_method)?;
        let pooling_config = serde_json::from_str::<PoolingConfig>(&std::fs::read_to_string(
            &pooling_config_filename,
        )?)?;
        let pooler = PoolingStrategy::from_config(pooling_config);

        // Load the dense layers
        let mut dense_layers = vec![];
        for dense_path in self.dense_paths.iter() {
            let dense_config_filename =
                download_hf_hub_file(&self.model_id, &format!("{dense_path}/config.json"))?;
            let dense_config = serde_json::from_str::<DenseConfig>(&std::fs::read_to_string(
                dense_config_filename,
            )?)?;

            let dense_vb = if self.with_safetensors {
                let weights_filename = download_hf_hub_file(
                    &self.model_id,
                    &format!("{dense_path}/model.safetensors"),
                )?;
                load_safetensors(&[weights_filename], DTYPE, &device)?
            } else {
                let weights_filename = download_hf_hub_file(
                    &self.model_id,
                    &format!("{dense_path}/pytorch_model.bin"),
                )?;
                candle_nn::VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
            };

            let layer = Dense::from_config(dense_vb, dense_config)?;
            dense_layers.push(layer);
        }

        // normalize
        let normalize = if self.with_normalization {
            Some(Normalize)
        } else {
            None
        };

        Ok(SentenceTransformer {
            model_config: model_config,
            batch_size: self.batch_size,
            device: device,
            tokenizer: tokenizer,
            transformer: bert,
            pooler: pooler,
            dense_layers: dense_layers,
            normalize: normalize,
        })
    }
}

pub struct SentenceTransformer {
    model_config: ModelConfig,
    batch_size: usize,
    device: Device,
    tokenizer: Tokenizer,
    transformer: BertModel,
    pooler: PoolingStrategy,
    dense_layers: Vec<Dense>,
    normalize: Option<Normalize>,
}

impl SentenceTransformer {
    pub fn embed(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut embeddings: Vec<Vec<f32>> = vec![];

        let mut token_ids = self
            .tokenizer
            .encode_batch(lines.to_vec(), true)?
            .iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect::<Vec<Vec<u32>>>();

        let pad_token = self
            .model_config
            .pad_token_id
            .unwrap_or(DEFAULT_PAD_TOKEN_ID);
        let batches =
            fast_token_based_batching(&mut token_ids, pad_token, self.batch_size, &self.device)?;

        for batch in batches.batches.iter() {
            // transformer
            let mut batch_embeddings = self.transformer.forward(
                &batch.input_ids,
                &batch.token_type_ids,
                Some(&batch.attention_mask),
            )?;

            // pool
            batch_embeddings = self
                .pooler
                .forward(&batch_embeddings, &batch.attention_mask)?;

            // dense
            for dense in self.dense_layers.iter() {
                batch_embeddings = dense.forward(&batch_embeddings)?;
            }

            // norm
            batch_embeddings = if let Some(norm) = &self.normalize {
                norm.forward(&batch_embeddings)
            } else {
                Ok(batch_embeddings)
            }?;

            // add batch embeddings to final embeddings - still unsorted at this point
            for emb in batch_embeddings.to_vec2()?.into_iter() {
                embeddings.push(emb);
            }
        }

        let mut sorted_embeddings = vec![vec![]; embeddings.len()];
        for (emb, (idx, _)) in embeddings.into_iter().zip(batches.original_ids) {
            sorted_embeddings[idx] = emb;
        }

        Ok(sorted_embeddings)
    }
}
