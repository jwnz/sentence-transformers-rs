use std::path::Path;

use candle_core::Tensor;
use candle_nn::VarBuilder;
use thiserror::Error;
use tokenizers::{Tokenizer, TruncationParams};

use crate::{
    error::LoadConfigError,
    models::{bert::BertModel, distilbert::DistilBertModel, xlm_roberta::XLMRobertaModel},
    utils::load_config,
};

pub trait TransformerLoad: Sized {
    fn load_transformer(
        vb: candle_nn::VarBuilder,
        config_path: &std::path::Path,
    ) -> Result<Self, TLErr>;
}

impl TransformerLoad for BertModel {
    fn load_transformer(vb: VarBuilder, config_filename: &Path) -> Result<Self, TLErr> {
        Ok(BertModel::load(vb, &load_config(config_filename)?)?)
    }
}

impl TransformerLoad for DistilBertModel {
    fn load_transformer(vb: VarBuilder, config_filename: &Path) -> Result<Self, TLErr> {
        Ok(DistilBertModel::load(vb, &load_config(config_filename)?)?)
    }
}

impl TransformerLoad for XLMRobertaModel {
    fn load_transformer(vb: VarBuilder, config_filename: &Path) -> Result<Self, TLErr> {
        Ok(XLMRobertaModel::load(vb, &load_config(config_filename)?)?)
    }
}

pub struct Transformer<T: TransformerLoad> {
    model: T,
    tokenizer: Tokenizer,
}

impl<T> Transformer<T>
where
    T: TransformerLoad,
{
    fn new(model: T, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub fn load(
        vb: VarBuilder,
        config_filename: &Path,
        tokenizer_filename: &Path,
        max_seq_length: usize,
    ) -> Result<Transformer<T>, TLErr> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: max_seq_length,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            direction: tokenizers::TruncationDirection::Right,
            stride: 0,
        }))?;
        tokenizer.with_padding(None);

        let transformer = T::load_transformer(vb, config_filename)?;
        Ok(Transformer::new(transformer, tokenizer))
    }
}

pub trait TransformerOps<T>
where
    T: TransformerLoad,
{
    /// The user will call tokenize outside somewhere and get a tokenized output.
    /// The output will be sorted, padded, and batched.
    fn tokenize(&self, lines: &[&str], batch_size: usize) -> Result<TokenizerOutput, TOTErr>;

    /// The user will take a batch and just do typical forward prop
    fn forward(&self, batch: &Batch) -> Result<Tensor, TOFErr>;
}

impl TransformerOps<BertModel> for Transformer<BertModel> {
    fn tokenize(&self, lines: &[&str], batch_size: usize) -> Result<TokenizerOutput, TOTErr> {
        Ok(TokenizerOutput {
            batches: todo!(),
            original_ids: todo!(),
        })
    }

    fn forward(&self, batch: &Batch) -> Result<Tensor, TOFErr> {
        Ok(self.model.forward(
            &batch.input_ids,
            &batch.token_type_ids,
            Some(&batch.attention_mask),
        )?)
    }
}

impl TransformerOps<DistilBertModel> for Transformer<DistilBertModel> {
    fn tokenize(&self, lines: &[&str], batch_size: usize) -> Result<TokenizerOutput, TOTErr> {
        Ok(TokenizerOutput {
            batches: todo!(),
            original_ids: todo!(),
        })
    }

    fn forward(&self, batch: &Batch) -> Result<Tensor, TOFErr> {
        Ok(self
            .model
            .forward(&batch.input_ids, &batch.attention_mask)?)
    }
}

impl TransformerOps<XLMRobertaModel> for Transformer<XLMRobertaModel> {
    fn tokenize(&self, lines: &[&str], batch_size: usize) -> Result<TokenizerOutput, TOTErr> {
        Ok(TokenizerOutput {
            batches: todo!(),
            original_ids: todo!(),
        })
    }

    fn forward(&self, batch: &Batch) -> Result<Tensor, TOFErr> {
        Ok(self.model.forward(
            &batch.input_ids,
            &batch.attention_mask,
            &batch.token_type_ids,
        )?)
    }
}

pub struct TokenizerOutput {
    pub batches: Vec<Batch>,
    pub original_ids: Vec<(usize, usize)>, // this is currently the idx and the sentence len / token cnt
}
pub struct Batch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub token_type_ids: Tensor,
}

#[derive(Debug, Error)]
pub enum TransformerLoadError {
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
}

#[derive(Debug, Error)]
pub enum TransformerOpsForwardError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),
}

#[derive(Debug, Error)]
pub enum TransformerOpsTokenizeError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),
}

type TLErr = TransformerLoadError;
type TOFErr = TransformerOpsForwardError;
type TOTErr = TransformerOpsTokenizeError;
