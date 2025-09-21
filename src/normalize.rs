use candle_core::Tensor;

use crate::error::NormalizeError;

pub enum Normalizer {
    NoNormalization,
    L2,
}

impl Normalizer {
    pub fn normalize(&self, xs: &Tensor) -> Result<Tensor, NormalizeError> {
        let norm = match self {
            Normalizer::NoNormalization => xs.clone(),
            Normalizer::L2 => xs.broadcast_div(&xs.sqr()?.sum_keepdim(1)?.sqrt()?)?,
        };
        Ok(norm)
    }
}
