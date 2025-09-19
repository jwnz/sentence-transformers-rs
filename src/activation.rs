use candle_core::Tensor;
use serde::Deserialize;

use crate::error::ActivationError;

#[derive(Deserialize, Clone)]
pub enum Activation {
    #[serde(alias = "Tanh")]
    #[serde(alias = "torch.nn.modules.activation.Tanh")]
    Tanh,

    #[serde(alias = "Identity")]
    #[serde(alias = "torch.nn.modules.linear.Identity")]
    Identity,
}
impl Activation {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor, ActivationError> {
        let xs = match self {
            Activation::Tanh => xs.tanh()?,
            Activation::Identity => xs.clone(),
        };

        Ok(xs)
    }
}
