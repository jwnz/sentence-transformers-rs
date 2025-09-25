//! Adapted from Candle (Apache-2.0, MIT License, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/bert.rs)
//! Adapted from Candle (Apache-2.0, MIT License, https://github.com/huggingface/text-embeddings-inference/blob/main/backends/candle/src/models/mpnet.rs)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

#[derive(Clone)]
struct HiddenActLayer {
    act: HiddenAct,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        Self { act }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpnet/configuration_mpnet.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub relative_attention_num_buckets: usize,
}

#[derive(Clone)]
struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let mut embeddings = input_embeddings;

        // positional embeddings x_i = i + 2, where anything is not the pad
        // hard coding the pad token for now
        // needs refactoring
        let pad_token_id = 1u32;
        let mut position_ids = vec![];

        for _ in 0.._bsize {
            let mut pos = vec![];
            for c in 0..seq_len {
                let cur_token_id = input_ids.get(0)?.get(1)?.to_scalar::<u32>()?;
                pos.push(if cur_token_id == pad_token_id {
                    1u32
                } else {
                    (c + 2) as u32
                });
            }
            position_ids.push(pos);
        }

        if let Some(position_embeddings) = &self.position_embeddings {
            // TODO: Proper absolute positions?
            // let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(position_ids, input_ids.device())?;

            let position_embeddings = position_embeddings.forward(&position_ids)?;
            embeddings = embeddings.add(&position_embeddings)?
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    }
}

#[derive(Clone)]
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("q"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("v"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("k"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        attention_bias: &Tensor,
    ) -> Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;

        let attention_scores = attention_scores.add(attention_bias)?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;
        let attention_probs =
            { candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)? };
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}

#[derive(Clone)]
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("attn.o"))?; // replace dense with attn.o
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
#[derive(Clone)]
struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::load(vb.pp("attn"), config)?; // replace self with attn
        let self_output = BertSelfOutput::load(vb.clone(), config)?; // remove output prefix
        Ok(Self {
            self_attention,
            self_output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        attention_bias: &Tensor,
    ) -> Result<Tensor> {
        let self_outputs =
            self.self_attention
                .forward(hidden_states, attention_mask, attention_bias)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
#[derive(Clone)]
struct BertIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
#[derive(Clone)]
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
#[derive(Clone)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = BertOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        attention_bias: &Tensor,
    ) -> Result<Tensor> {
        let attention_output =
            self.attention
                .forward(hidden_states, attention_mask, attention_bias)?;
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
// #[derive(Clone)]
pub struct BertEncoder {
    pub layers: Vec<BertLayer>,
    pub relative_attn_bias: MPNetAttentionBias,
}

impl BertEncoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let relative_attn_bias = MPNetAttentionBias::load(vb.clone(), config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(BertEncoder {
            layers,
            relative_attn_bias,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        let attention_bias = self.relative_attn_bias.forward(&hidden_states)?;
        // let attention_bias = attention_mask.broadcast_as(attention_bias.shape())?;

        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, &attention_mask, &attention_bias)?
        }
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
pub struct MPNetModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pub device: Device,
}

impl MPNetModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                panic!("Some error happened loading mpnet model: {:?}", err)
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        let attention_mask = attention_mask.clone();
        let dtype = embedding_output.dtype();
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L995
        let attention_mask = get_extended_attention_mask(&attention_mask, dtype)?;
        let sequence_output = self.encoder.forward(&embedding_output, &attention_mask)?;
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

struct MPNetAttentionBias {
    relative_attention_bias: Embedding,
    relative_attention_num_buckets: usize,
}

impl MPNetAttentionBias {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let relative_attention_bias = Embedding::new(
            vb.pp("relative_attention_bias").get(
                (
                    config.relative_attention_num_buckets,
                    config.num_attention_heads,
                ),
                "weight",
            )?,
            config.num_attention_heads,
        );

        Ok(MPNetAttentionBias {
            relative_attention_bias,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
        })
    }

    fn relative_position_bucket(
        &self,
        relative_position: &Tensor,
        max_distance: i64,
    ) -> Result<Tensor> {
        let device = relative_position.device();

        let num_buckets = (self.relative_attention_num_buckets / 2) as f64;
        let max_exact = num_buckets / 2.0;
        let max_distance_log = (max_distance as f64 / max_exact).ln();
        let scale = (num_buckets - max_exact) / max_distance_log;

        let mut ret = Tensor::zeros_like(relative_position)?;
        let n = relative_position.to_dtype(DType::F32)?.neg()?;

        ret = ret.add(&(&n.lt(0.0)?.to_dtype(DType::F32)? * num_buckets)?.to_dtype(DType::I64)?)?;
        let n = n.abs()?;

        let is_small = n.lt(max_exact)?;

        let log_val = (n.clone() / max_exact)?.log()?;
        let val_if_large = (max_exact + (log_val * scale)?)?;

        let val_if_large = val_if_large
            .minimum(&Tensor::full(
                (num_buckets - 1.0) as f32,
                val_if_large.shape(),
                device,
            )?)?
            .to_dtype(DType::I64)?;
        ret.add(&is_small.where_cond(&n.clone().to_dtype(DType::I64)?, &val_if_large)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let qlen = x.dim(1)?;
        let klen = x.dim(1)?;

        let context_position = Tensor::arange(0_i64, qlen as i64, x.device())?.unsqueeze(1)?;
        let memory_position = Tensor::arange(0_i64, klen as i64, x.device())?.unsqueeze(0)?;

        let context_position = context_position.broadcast_as((qlen, klen))?;
        let memory_position = memory_position.broadcast_as((qlen, klen))?;

        let relative_position = memory_position.sub(&context_position)?;

        let rp_bucket = self.relative_position_bucket(&relative_position, 128)?;

        let values = self.relative_attention_bias.forward(&rp_bucket)?;
        let values = values.permute([2, 0, 1])?.unsqueeze(0)?;
        let values = values
            .expand(&[bsz, values.dim(1)?, qlen, klen])?
            .contiguous()?;
        Ok(values)
    }
}
