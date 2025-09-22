use std::path::Path;

use candle_core::Tensor;
use candle_nn::{linear_b, Linear, Module, VarBuilder};
use serde::Deserialize;

use crate::{activation::Activation, error::DenseError, utils::load_config};

#[derive(Deserialize)]
pub struct DenseConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
    activation_function: Activation,
}

pub struct Dense {
    linear: Linear,
    config: DenseConfig,
}

impl Dense {
    /// Create a `Dense` layer from a `DenseConfig`. This takes ownership of the `DenseConfig` object.
    pub fn from_config(vb: VarBuilder, config_filename: &Path) -> Result<Dense, DenseError> {
        let config = load_config::<DenseConfig>(config_filename)?;
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
