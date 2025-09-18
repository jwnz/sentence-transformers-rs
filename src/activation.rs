use candle_core::Tensor;
use serde::Deserialize;

use crate::error::ActivationError;

#[derive(Deserialize, Clone)]
pub enum Activation {
    #[serde(alias = "Tanh")]
    #[serde(alias = "torch.nn.modules.activation.Tanh")]
    Tanh,
}
impl Activation {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor, ActivationError> {
        let xs = match self {
            Activation::Tanh => xs.tanh()?,
        };

        Ok(xs)
    }
}
