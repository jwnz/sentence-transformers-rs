use candle_core::{DType, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

use candle_nn::Embedding;

use crate::activation::Activation;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub model_type: Option<String>,
}

pub struct BertEmbedding {
    layer_norm: LayerNorm,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    word_embeddings: Embedding,
    span: tracing::Span,
}

impl BertEmbedding {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let word_embedding_shape = (config.vocab_size, config.hidden_size);
        let word_embeddings = Embedding::new(
            vb.pp("word_embeddings")
                .get(word_embedding_shape, "weight")?,
            config.hidden_size,
        );

        let token_type_shape = (config.type_vocab_size, config.hidden_size);
        let token_type_embeddings = Embedding::new(
            vb.pp("token_type_embeddings")
                .get(token_type_shape, "weight")?,
            config.hidden_size,
        );

        let position_embedding_shape = (config.max_position_embeddings, config.hidden_size);
        let position_embeddings = Embedding::new(
            vb.pp("position_embeddings")
                .get(position_embedding_shape, "weight")?,
            config.hidden_size,
        );

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        Ok(Self {
            layer_norm,
            position_embeddings,
            token_type_embeddings,
            word_embeddings,
            span: tracing::span!(tracing::Level::TRACE, "BertEmbedding"),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        // fused layer_norm + residual, kernels can be used here.
        // something like: norm((input + token_type), position)
        // see: https://github.com/huggingface/candle-layer-norm
        let embeddings = (&input_embeddings + token_type_embeddings)?;
        let embeddings = embeddings.broadcast_add(&position_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;

        Ok(embeddings)
    }
}

pub struct BertMultiHeadAttention {
    qkv: Linear,
    dense: Linear,
    layer_norm: LayerNorm,
    softmax_scale: f64,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
}

impl BertMultiHeadAttention {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;

        let query_weight = vb
            .pp("self.query")
            .get((all_head_size, hidden_size), "weight")?;
        let query_bias = vb.pp("self.query").get(all_head_size, "bias")?;

        let key_weight = vb
            .pp("self.key")
            .get((all_head_size, hidden_size), "weight")?;
        let key_bias = vb.pp("self.key").get(all_head_size, "bias")?;

        let value_weight = vb
            .pp("self.value")
            .get((all_head_size, hidden_size), "weight")?;
        let value_bias = vb.pp("self.value").get(all_head_size, "bias")?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_bias = Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?;

        let qkv = Linear::new(qkv_weight, Some(qkv_bias));

        let dense_weight = vb
            .pp("output.dense")
            .get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("output.dense").get(hidden_size, "bias")?;

        let dense = Linear::new(dense_weight, Some(dense_bias));

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("output.LayerNorm"),
        )?;

        let softmax_scale = 1.0 / ((config.hidden_size / config.num_attention_heads) as f64).sqrt();

        Ok(Self {
            qkv,
            dense,
            layer_norm,
            softmax_scale,
            num_attention_heads: config.num_attention_heads,
            attention_head_size: config.hidden_size / config.num_attention_heads,
            span: tracing::span!(tracing::Level::TRACE, "BertMultiHeadAttention"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_bias: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = hidden_states.clone();

        // (bsz, seq_len, 3 * num_heads * head_size) or (4, 56, 2304)
        let qkv = self.qkv.forward(&hidden_states)?;

        // (bsz, 3 * num_heads, seq_len, head_size) becomes (4, 36, 56, 64)
        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(new_qkv_shape.as_slice())?.transpose(1, 2)?;

        // [
        //     (bsz, num_heads, seq_len, head_size)
        //     (bsz, num_heads, seq_len, head_size)
        //     (bsz, num_heads, seq_len, head_size)
        // ]
        let qkv = qkv.chunk(3, 1)?;
        let query = &qkv[0].contiguous()?;
        let key = &qkv[1].contiguous()?;
        let value = &qkv[2].contiguous()?;

        let attention_scores = query.matmul(&key.t()?)?;
        let attention_scores = (attention_scores * self.softmax_scale)?;
        let attention_scores = attention_scores.add(attention_bias)?;

        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

        let context_layer = attention_probs.matmul(&value)?;
        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;

        let hidden_states = self.dense.forward(&context_layer)?;
        let hidden_states = self.layer_norm.forward(&(hidden_states + residual)?)?;

        Ok(hidden_states)
    }
}

pub struct BertIntermediate {
    dense: Linear,
    activation: Activation,
    span: tracing::Span,
}

impl BertIntermediate {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense_weight = vb
            .pp("dense")
            .get((config.intermediate_size, config.hidden_size), "weight")?;
        let dense_bias = vb.pp("dense").get(config.intermediate_size, "bias")?;
        let dense = Linear::new(dense_weight, Some(dense_bias));

        Ok(Self {
            dense,
            activation: config.hidden_act.clone(),
            span: tracing::span!(tracing::Level::TRACE, "BertIntermediate"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.activation.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct BertOutput {
    layer_norm: LayerNorm,
    dense: Linear,
    span: tracing::Span,
}

impl BertOutput {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dense_weight = vb
            .pp("dense")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let dense_bias = vb.pp("dense").get(config.hidden_size, "bias")?;
        let dense = Linear::new(dense_weight, Some(dense_bias));

        Ok(Self {
            layer_norm,
            dense,
            span: tracing::span!(tracing::Level::TRACE, "BertOutput"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.dense.forward(&hidden_states)?;
        // we can use a faster layer norm as described above via candle_layer_norm
        let hidden_states = self.layer_norm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

pub struct BertLayer {
    attention: BertMultiHeadAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
    span: tracing::Span,
}

impl BertLayer {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention = BertMultiHeadAttention::load(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = BertOutput::load(vb.pp("output"), config)?;

        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "BertLayer"),
        })
    }
    pub fn forward(&self, hidden_states: &Tensor, attention_bias: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.attention.forward(hidden_states, attention_bias)?;
        let residual = hidden_states.clone();

        let hidden_states = self.intermediate.forward(&hidden_states)?;
        let hidden_states = self.output.forward(&hidden_states, &residual)?;

        Ok(hidden_states)
    }
}

pub struct BertEncoder {
    layers: Vec<BertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        Ok(Self {
            layers: (0..config.num_hidden_layers)
                .map(|i| BertLayer::load(vb.pp(format!("layer.{i}")), config))
                .collect::<Result<Vec<BertLayer>>>()?,
            span: tracing::span!(tracing::Level::TRACE, "BertEncoder"),
        })
    }
    pub fn forward(&self, hidden_states: &Tensor, attention_bias: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();

        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_bias)?;
        }

        Ok(hidden_states)
    }
}

pub struct BertModel {
    embeddings: BertEmbedding,
    encoder: BertEncoder,
    config: BertConfig,
    span: tracing::Span,
}

impl BertModel {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let embeddings = BertEmbedding::load(vb.pp("embeddings"), config)?;
        let encoder = BertEncoder::load(vb.pp("encoder"), config)?;

        Ok(Self {
            embeddings,
            encoder,
            config: config.clone(),
            span: tracing::span!(tracing::Level::TRACE, "BertModel"),
        })
    }
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (bsz, seq_len) = input_ids.dims2()?;
        let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
        let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;

        let embedding_output =
            self.embeddings
                .forward(&input_ids, &token_type_ids, &position_ids)?;

        let attention_bias =
            get_extended_attention_mask(&attention_mask, embedding_output.dtype())?;
        let attention_bias = attention_bias.broadcast_as((
            bsz,
            self.config.num_attention_heads,
            seq_len,
            seq_len,
        ))?;
        let outputs = self.encoder.forward(&embedding_output, &attention_bias)?;

        Ok(outputs)
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
