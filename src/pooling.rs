use core::f32;
use std::path::Path;

use candle_core::{Tensor, D};
use serde::Deserialize;

use crate::{
    error::{PoolerFromConfigError, PoolingError, PoolingStrategyError},
    utils::load_config,
};

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct PoolingConfig {
    word_embedding_dimension: usize,
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
}

pub enum PoolingStrategy {
    Cls,
    Mean,
    Max,
    MeanSqrtLenTokens,
}

impl PoolingStrategy {
    pub fn pool(
        &self,
        token_embeddings: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Tensor, PoolingStrategyError> {
        match self {
            PoolingStrategy::Cls => Ok(token_embeddings.get_on_dim(1, 0)?.contiguous()?),
            PoolingStrategy::Mean => {
                let input_mask_expanded = attn_mask
                    .unsqueeze(D::Minus1)?
                    .expand(token_embeddings.shape())?
                    .to_dtype(token_embeddings.dtype())?;

                let sum_embeddings = token_embeddings
                    .broadcast_mul(&input_mask_expanded)?
                    .sum(1)?;

                let sum_mask = &input_mask_expanded.sum(1)?;

                let sum_mask = sum_mask.clamp(1e-9, f32::INFINITY)?;

                let res = sum_embeddings.broadcast_div(&sum_mask)?;

                Ok(res)
            }
            PoolingStrategy::Max => {
                todo!()
            }
            PoolingStrategy::MeanSqrtLenTokens => {
                todo!()
            }
        }
    }
}

pub struct Pooler {
    // The original repo allows for multiple pooling strategies; the results are concatenated
    strategies: Vec<PoolingStrategy>,
}

impl Pooler {
    pub fn from_config(config_filepath: &Path) -> Result<Self, PoolerFromConfigError> {
        let config = load_config::<PoolingConfig>(config_filepath)?;
        let mut strategies = vec![];

        if config.pooling_mode_cls_token {
            strategies.push(PoolingStrategy::Cls);
        }
        if config.pooling_mode_mean_tokens {
            strategies.push(PoolingStrategy::Mean);
        }
        if config.pooling_mode_max_tokens {
            strategies.push(PoolingStrategy::Max);
        }
        if config.pooling_mode_mean_sqrt_len_tokens {
            strategies.push(PoolingStrategy::MeanSqrtLenTokens);
        }

        if strategies.is_empty() {
            return Err(PoolerFromConfigError::PoolingStrategyNotSpecifiedError {
                config: config.clone(),
            });
        }
        Ok(Pooler { strategies })
    }

    pub fn pool(
        &self,
        token_embeddings: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Tensor, PoolingError> {
        Ok(Tensor::cat(
            &self
                .strategies
                .iter()
                .map(|s| s.pool(token_embeddings, attn_mask))
                .collect::<Result<Vec<Tensor>, PoolingStrategyError>>()?,
            1,
        )?)
    }
}
