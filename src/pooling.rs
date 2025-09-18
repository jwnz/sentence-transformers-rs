use core::f32;

use candle_core::{Tensor, D};
use serde::Deserialize;

use crate::error::PoolingError;

pub enum PoolingStrategy {
    Cls,
    Mean,
    Max,
    MeanSqrtLenTokens,
}

impl PoolingStrategy {
    pub fn from_config(config: PoolingConfig) -> PoolingStrategy {
        if config.pooling_mode_cls_token {
            return PoolingStrategy::Cls;
        } else if config.pooling_mode_mean_tokens {
            return PoolingStrategy::Mean;
        } else if config.pooling_mode_max_tokens {
            return PoolingStrategy::Max;
        } else if config.pooling_mode_mean_sqrt_len_tokens {
            return PoolingStrategy::MeanSqrtLenTokens;
        } else {
            panic!("should have pooling")
        }
    }
}

impl PoolingStrategy {
    pub fn forward(
        &self,
        token_embeddings: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Tensor, PoolingError> {
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

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct PoolingConfig {
    word_embedding_dimension: usize,
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
}
