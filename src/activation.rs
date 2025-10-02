use candle_core::{Module, Tensor};
use serde::Deserialize;

#[derive(Debug, PartialEq, Deserialize, Clone, Copy)]
pub enum Activation {
    #[serde(alias = "Tanh")]
    #[serde(alias = "torch.nn.modules.activation.Tanh")]
    Tanh,

    #[serde(alias = "Identity")]
    #[serde(alias = "torch.nn.modules.linear.Identity")]
    Identity,

    #[serde(alias = "gelu")]
    Gelu,

    #[serde(alias = "relu")]
    Relu,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::error::Error> {
        let xs = match self {
            Activation::Tanh => xs.tanh()?,
            Activation::Identity => xs.clone(),
            Activation::Gelu => xs.gelu_erf()?,
            Activation::Relu => xs.relu()?,
        };

        Ok(xs)
    }
}
