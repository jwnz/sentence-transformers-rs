use candle_core::Tensor;

use crate::error::NormalizeError;

pub struct Normalize;

impl Normalize {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor, NormalizeError> {
        // Divide each value by the L2 Norm - square root fot he sum of the squares of each component
        Ok(xs.broadcast_div(&xs.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
}
