use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde_with::skip_serializing_none]
pub struct ModelConfig {
    #[serde(rename = "_name_or_path")]
    pub name_or_path: Option<String>,
    pub architectures: Option<Vec<String>>,
    pub attention_probs_dropout_prob: Option<f32>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub gradient_checkpointing: Option<bool>,
    pub hidden_act: Option<String>,
    pub hidden_dropout_prob: Option<f32>,
    pub hidden_size: Option<usize>,
    pub initializer_range: Option<f32>,
    pub intermediate_size: Option<usize>,
    pub layer_norm_eps: Option<f32>,
    pub max_position_embeddings: Option<usize>,
    pub model_type: Option<String>,
    pub num_attention_heads: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub output_past: Option<bool>,
    pub pad_token_id: Option<usize>,
    pub position_embedding_type: Option<String>,
    pub transformers_version: Option<String>,
    pub type_vocab_size: Option<usize>,
    pub use_cache: Option<bool>,
    pub vocab_size: Option<usize>,
}

#[derive(Deserialize)]
pub struct SentenceBertConfig {
    pub max_seq_length: usize,
    pub do_lower_case: bool,
}
