//! Adapted from Candle (Apache-2.0, MIT License, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/distilbert.rs)

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

use crate::activation::Activation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    n_layers: usize,
    n_heads: usize,
    hidden_dim: usize,
    activation: Activation,
    max_position_embeddings: usize,
    initializer_range: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    model_type: Option<String>,
}

struct Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl Embeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings =
            candle_nn::embedding(config.vocab_size, config.dim, vb.pp("word_embeddings"))?;
        let position_embeddings = candle_nn::embedding(
            config.max_position_embeddings,
            config.dim,
            vb.pp("position_embeddings"),
        )?;
        let layer_norm = layer_norm(config.dim, 1e-12, vb.pp("LayerNorm"))?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
        let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
        let embeddings =
            input_embeddings.broadcast_add(&self.position_embeddings.forward(&position_ids)?)?;

        let embeddings = self.layer_norm.forward(&embeddings)?;
        Ok(embeddings)
    }
}

struct MultiHeadSelfAttention {
    qkv: Linear,
    out_lin: Linear,
    n_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,
}

impl MultiHeadSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.dim / config.n_heads;
        let all_head_size = config.n_heads * attention_head_size;
        let dim = config.dim;

        let query_weight = vb.pp("q_lin").get((all_head_size, dim), "weight")?;
        let query_bias = vb.pp("q_lin").get(all_head_size, "bias")?;

        let key_weight = vb.pp("k_lin").get((all_head_size, dim), "weight")?;
        let key_bias = vb.pp("k_lin").get(all_head_size, "bias")?;

        let value_weight = vb.pp("v_lin").get((all_head_size, dim), "weight")?;
        let value_bias = vb.pp("v_lin").get(all_head_size, "bias")?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_bias = Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?;

        let qkv = Linear::new(qkv_weight, Some(qkv_bias));

        let softmax_scale = 1.0 / ((config.dim / config.n_heads) as f64).sqrt();

        let out_lin = linear(all_head_size, dim, vb.pp("out_lin"))?;
        Ok(Self {
            qkv,
            out_lin,
            n_heads: config.n_heads,
            attention_head_size,
            softmax_scale: softmax_scale,
        })
    }
}

impl MultiHeadSelfAttention {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let qkv = self.qkv.forward(&hidden_states)?;

        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.n_heads * 3);
        new_qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(new_qkv_shape.as_slice())?.transpose(1, 2)?;

        let qkv = qkv.chunk(3, 1)?;
        let query = &qkv[0].contiguous()?;
        let key = &qkv[1].contiguous()?;
        let value = &qkv[2].contiguous()?;

        let attention_scores = query.matmul(&key.t()?)?;
        let attention_scores = (attention_scores * self.softmax_scale)?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;

        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

        let context_layer = attention_probs.matmul(&value)?;
        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;

        let context_layer = self.out_lin.forward(&context_layer)?;

        Ok(context_layer)
    }
}

#[allow(clippy::upper_case_acronyms)]
struct FFN {
    lin1: Linear,
    lin2: Linear,
    activation: Activation,
}

impl FFN {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let lin1 = linear(config.dim, config.hidden_dim, vb.pp("lin1"))?;
        let lin2 = linear(config.hidden_dim, config.dim, vb.pp("lin2"))?;
        Ok(Self {
            lin1,
            lin2,
            activation: config.activation.clone(),
        })
    }
}

impl Module for FFN {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        hidden_states
            .apply(&self.lin1)?
            .apply(&self.activation)?
            .apply(&self.lin2)
    }
}

struct TransformerBlock {
    attention: MultiHeadSelfAttention,
    sa_layer_norm: LayerNorm,
    ffn: FFN,
    output_layer_norm: LayerNorm,
}

impl TransformerBlock {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = MultiHeadSelfAttention::load(vb.pp("attention"), config)?;
        let sa_layer_norm = layer_norm(config.dim, 1e-12, vb.pp("sa_layer_norm"))?;
        let ffn = FFN::load(vb.pp("ffn"), config)?;
        let output_layer_norm = layer_norm(config.dim, 1e-12, vb.pp("output_layer_norm"))?;
        Ok(Self {
            attention,
            sa_layer_norm,
            ffn,
            output_layer_norm,
        })
    }
}

impl TransformerBlock {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let sa_output = self.attention.forward(hidden_states, attention_mask)?;
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let sa_output = sa_output.broadcast_add(hidden_states)?;
        let sa_output = self.sa_layer_norm.forward(&sa_output)?;

        let ffn_output = self.ffn.forward(&sa_output)?;
        let ffn_output = (&ffn_output + sa_output)?;
        let output = self.output_layer_norm.forward(&ffn_output)?;
        Ok(output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
struct Transformer {
    layers: Vec<TransformerBlock>,
}

impl Transformer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.n_layers)
            .map(|index| TransformerBlock::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Transformer { layers })
    }
}

impl Transformer {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

pub struct DistilBertModel {
    embeddings: Embeddings,
    transformer: Transformer,
    pub device: Device,
}

impl DistilBertModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, transformer) = match (
            Embeddings::load(vb.pp("embeddings"), config),
            Transformer::load(vb.pp("transformer"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        Embeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                        Transformer::load(vb.pp(format!("{model_type}.transformer")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            transformer,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        let attention_mask = get_extended_attention_mask(attention_mask, embedding_output.dtype())?;
        let sequence_output = self
            .transformer
            .forward(&embedding_output, &attention_mask)?;
        Ok(sequence_output)
    }
}

fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let attention_mask = match attention_mask.rank() {
        3 => attention_mask.unsqueeze(1)?,
        2 => attention_mask.unsqueeze(1)?.unsqueeze(1)?,
        _ => candle_core::bail!("Wrong shape for input_ids or attention_mask"),
    };
    let attention_mask = attention_mask.to_dtype(dtype)?;
    // torch.finfo(dtype).min
    (attention_mask.ones_like()? - &attention_mask)?.broadcast_mul(
        &Tensor::try_from(f32::MIN)?
            .to_device(attention_mask.device())?
            .to_dtype(dtype)?,
    )
}
