pub mod bert;
pub mod distilbert;
pub mod xlm_roberta;

use std::path::Path;

use candle_nn::VarBuilder;
use thiserror::Error;

use crate::{
    error::LoadConfigError,
    models::{distilbert::DistilBertModel, xlm_roberta::XLMRobertaModel},
    utils::load_config,
};

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
}

type Err = TransformerLoadError;

pub trait TransformerLoad: Sized {
    fn load_transformer(
        vb: candle_nn::VarBuilder,
        config_path: &std::path::Path,
    ) -> Result<Self, Err>;
}

impl TransformerLoad for bert::BertModel {
    fn load_transformer(vb: VarBuilder, config_filename: &Path) -> Result<Self, Err> {
        Ok(bert::BertModel::load(vb, &load_config(config_filename)?)?)
    }
}

impl TransformerLoad for DistilBertModel {
    fn load_transformer(vb: VarBuilder, config_filename: &Path) -> Result<Self, Err> {
        Ok(DistilBertModel::load(vb, &load_config(config_filename)?)?)
    }
}

impl TransformerLoad for XLMRobertaModel {
    fn load_transformer(vb: VarBuilder, config_filename: &Path) -> Result<Self, Err> {
        Ok(XLMRobertaModel::load(vb, &load_config(config_filename)?)?)
    }
}
