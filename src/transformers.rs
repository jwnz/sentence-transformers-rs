use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use thiserror::Error;
use tokenizers::{Tokenizer, TruncationParams};

use crate::{
    error::{FastTokenBatchError, LoadConfigError},
    models::{
        bert::{BertConfig, BertModel},
        distilbert::{Config as DistilBertConfig, DistilBertModel},
        mpnet::{Config as MPNetConfig, MPNetModel},
        xlm_roberta::{Config as XLMRobertaConfig, XLMRobertaModel},
    },
    utils::{fast_token_based_batching, load_config, Batch, TokenBatchOutput},
};

#[derive(Debug, Deserialize, Clone)]
pub struct Architectures {
    architectures: Vec<Architecture>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum Architecture {
    BertModel,
    DistilBertModel,
    XLMRobertaModel,
    #[serde(alias = "MPNetForMaskedLM")]
    MPNetModel,
}

pub enum Model {
    Bert(BertModel),
    DistilBert(DistilBertModel),
    XLMRoberta(XLMRobertaModel),
    MPNetModel(MPNetModel),
}

pub struct Transformer {
    model: Model,
    tokenizer: Tokenizer,
}

impl Transformer {
    pub fn load(
        vb: VarBuilder,
        config_filename: &Path,
        tokenizer_filename: &Path,
        max_seq_length: usize,
    ) -> Result<Self, Err> {
        // Load tokenizer
        let strategy = tokenizers::TruncationStrategy::LongestFirst;
        let direction = tokenizers::TruncationDirection::Right;
        let max_length = max_seq_length;
        let truncation_params = TruncationParams {
            max_length,
            strategy,
            direction,
            stride: 0,
        };
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        tokenizer.with_truncation(Some(truncation_params))?;
        tokenizer.with_padding(None);

        // assume only a single model architecture for now
        let model_type = load_config::<Architectures>(config_filename)?.architectures[0];

        let transformer = match model_type {
            Architecture::BertModel => {
                let config = load_config::<BertConfig>(config_filename)?;
                Self {
                    model: Model::Bert(BertModel::load(vb, &config)?),
                    tokenizer: tokenizer,
                }
            }
            Architecture::XLMRobertaModel => {
                let config = load_config::<XLMRobertaConfig>(config_filename)?;
                Self {
                    model: Model::XLMRoberta(XLMRobertaModel::load(vb, &config)?),
                    tokenizer: tokenizer,
                }
            }
            Architecture::DistilBertModel => {
                let config = load_config::<DistilBertConfig>(config_filename)?;
                Self {
                    model: Model::DistilBert(DistilBertModel::load(vb, &config)?),
                    tokenizer: tokenizer,
                }
            }
            Architecture::MPNetModel => {
                let config = load_config::<MPNetConfig>(config_filename)?;
                Self {
                    model: Model::MPNetModel(MPNetModel::load(vb, &config)?),
                    tokenizer: tokenizer,
                }
            }
        };
        Ok(transformer)
    }

    pub fn forward(&self, batch: &Batch) -> Result<Tensor, Err> {
        match &self.model {
            Model::Bert(model) => Ok(model.forward(
                &batch.input_ids,
                &batch.token_type_ids,
                &batch.attention_mask,
            )?),
            Model::XLMRoberta(model) => Ok(model.forward(
                &batch.input_ids,
                &batch.attention_mask,
                &batch.token_type_ids,
            )?),
            Model::DistilBert(model) => Ok(model.forward(&batch.input_ids, &batch.attention_mask)?),
            Model::MPNetModel(model) => Ok(model.forward(&batch.input_ids, &batch.attention_mask)?),
        }
    }

    pub fn tokenize(
        &self,
        lines: &[&str],
        device: &Device,
        pad_token_id: usize,
        batch_size: usize,
    ) -> Result<TokenBatchOutput, Err> {
        match &self.model {
            Model::Bert(_) | Model::DistilBert(_) | Model::XLMRoberta(_) | Model::MPNetModel(_) => {
                let mut token_ids = self
                    .tokenizer
                    .encode_batch(lines.to_vec(), true)?
                    .iter()
                    .map(|enc| enc.get_ids().to_vec())
                    .collect::<Vec<Vec<u32>>>();

                let batches =
                    fast_token_based_batching(&mut token_ids, pad_token_id, batch_size, device)?;

                Ok(TokenBatchOutput {
                    batches: batches.batches,
                    original_ids: batches.original_ids,
                })
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum TransformerError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),

    #[error("SerdeJsonError({0})")]
    SerdeJsonError(#[from] serde_json::Error),

    #[error("IO Error({0})")]
    StdIOError(#[from] std::io::Error),

    #[error("LoadConfigError({0})")]
    LoadConfigError(#[from] LoadConfigError),

    #[error("TokenizersError: {0}")]
    TokenizersError(#[from] tokenizers::Error),

    #[error("FastTokenBatchError: {0}")]
    FastTokenBatchError(#[from] FastTokenBatchError),
}
type Err = TransformerError;
