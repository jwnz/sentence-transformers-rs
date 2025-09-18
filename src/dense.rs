use candle_core::Tensor;
use candle_nn::{linear_b, Linear, Module, VarBuilder};
use serde::Deserialize;

use crate::{activation::Activation, error::DenseError};

pub struct Dense {
    linear: Linear,
    config: DenseConfig,
}

impl Dense {
    /// Create a `Dense` layer from a `DenseConfig`. This takes ownership of the `DenseConfig` object.
    pub fn from_config(vb: VarBuilder, config: DenseConfig) -> Result<Dense, DenseError> {
        Ok(Self {
            linear: linear_b(
                config.in_features,
                config.out_features,
                config.bias,
                vb.pp("linear"),
            )?,
            config,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor, DenseError> {
        Ok(self
            .config
            .activation_function
            .forward(&self.linear.forward(&xs)?)?)
    }
}

#[derive(Deserialize)]
pub struct DenseConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
    activation_function: Activation,
}
