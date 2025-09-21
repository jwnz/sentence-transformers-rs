use candle_core::Device;
use strum_macros::Display;

use crate::{transformers::Transformer, utils::load_config};

use super::{
    config::{ModelConfig, SentenceBertConfig},
    dense::{Dense, DenseConfig},
    error::{EmbedError, SentenceTransformerBuilderError},
    models::bert::DTYPE,
    normalize::Normalize,
    pooling::{PoolingConfig, PoolingStrategy},
    utils::{download_hf_hub_file, load_safetensors},
};

const DEFAULT_BATCH_SIZE: usize = 2048;
const DEFAULT_WITH_SAFETENSORS: bool = false;
const DEFAULT_WITH_NORMALIZE: bool = false;
const DEFAULT_PAD_TOKEN_ID: usize = 0;

#[derive(Display)]
pub enum Which {
    #[strum(serialize = "sentence-transformers/all-MiniLM-L6-v2")]
    AllMiniLML6v2,
    #[strum(serialize = "sentence-transformers/all-MiniLM-L12-v2")]
    AllMiniLML12v2,
    #[strum(serialize = "sentence-transformers/paraphrase-MiniLM-L6-v2")]
    ParaphraseMiniLML6v2,
    #[strum(serialize = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")]
    ParaphraseMultilingualMiniLML12v2,
    #[strum(serialize = "sentence-transformers/LaBSE")]
    LaBSE,
    #[strum(serialize = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")]
    ParaphraseMultilingualMpnetBaseV2,
    #[strum(serialize = "sentence-transformers/distiluse-base-multilingual-cased-v2")]
    DistiluseBaseMultilingualCasedV2,
}

pub struct SentenceTransformerBuilder {
    model_id: String,
    with_safetensors: bool,
    with_normalization: bool,
    batch_size: usize,
    device: Option<Device>,
    pooling_path: Option<String>,
    dense_paths: Vec<String>,
}

impl SentenceTransformerBuilder {
    pub fn new(model_id: impl AsRef<str>) -> Self {
        Self {
            model_id: model_id.as_ref().to_string(),
            with_safetensors: DEFAULT_WITH_SAFETENSORS,
            with_normalization: DEFAULT_WITH_NORMALIZE,
            batch_size: DEFAULT_BATCH_SIZE,
            device: None,
            pooling_path: None,
            dense_paths: vec![],
        }
    }

    pub fn with_sentence_transformer(model: &Which) -> SentenceTransformerBuilder {
        let model_string = model.to_string();
        match model {
            // Pooling and a single dense with norm
            Which::LaBSE => SentenceTransformerBuilder::new(model_string)
                .with_safetensors()
                .with_normalization()
                .with_pooling("1_Pooling")
                .with_dense("2_Dense"),

            // Pooling and a single dense with no norm
            Which::DistiluseBaseMultilingualCasedV2 => {
                SentenceTransformerBuilder::new(model_string)
                    .with_safetensors()
                    .with_pooling("1_Pooling")
                    .with_dense("2_Dense")
            }

            // Has pooling, but not norm
            Which::ParaphraseMultilingualMiniLML12v2
            | Which::ParaphraseMiniLML6v2
            | Which::ParaphraseMultilingualMpnetBaseV2 => {
                SentenceTransformerBuilder::new(model_string)
                    .with_safetensors()
                    .with_pooling("1_Pooling")
            }

            // Pooling with norm
            Which::AllMiniLML6v2 | Which::AllMiniLML12v2 => {
                SentenceTransformerBuilder::new(model_string)
                    .with_safetensors()
                    .with_normalization()
                    .with_pooling("1_Pooling")
            }
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> SentenceTransformerBuilder {
        self.batch_size = batch_size;
        self
    }

    pub fn with_safetensors(mut self) -> SentenceTransformerBuilder {
        self.with_safetensors = true;
        self
    }

    pub fn with_normalization(mut self) -> SentenceTransformerBuilder {
        self.with_normalization = true;
        self
    }

    pub fn with_device(mut self, device: &Device) -> SentenceTransformerBuilder {
        self.device = Some(device.clone());
        self
    }

    pub fn with_pooling(mut self, pooling: &str) -> SentenceTransformerBuilder {
        self.pooling_path = Some(pooling.to_string());
        self
    }

    pub fn with_dense(mut self, dense_path: &str) -> SentenceTransformerBuilder {
        self.dense_paths.push(dense_path.to_string());
        self
    }

    pub fn build(self) -> Result<SentenceTransformer, SentenceTransformerBuilderError> {
        // Device must be specified
        let device = self
            .device
            .ok_or_else(|| SentenceTransformerBuilderError::DeviceNotSpecified)?;

        // The pooling method must also be specified
        let pooling_method = self
            .pooling_path
            .ok_or_else(|| SentenceTransformerBuilderError::PoolingMethodNotSpecified)?;
        let pooling_method = format!("{pooling_method}/config.json");

        // load the model's hf_hub repo config
        let config_filename = download_hf_hub_file(&self.model_id, "config.json")?;
        let model_config = load_config::<ModelConfig>(&config_filename)?;

        // Load the sbert config
        let sbert_config_filename =
            download_hf_hub_file(&self.model_id, "sentence_bert_config.json")?;
        let sbert_config = load_config::<SentenceBertConfig>(&sbert_config_filename)?;

        // Load the transformer
        let vb = if self.with_safetensors {
            let weights_filename = download_hf_hub_file(&self.model_id, "model.safetensors")?;
            load_safetensors(&[weights_filename], DTYPE, &device)?
        } else {
            let weights_filename = download_hf_hub_file(&self.model_id, "pytorch_model.bin")?;
            candle_nn::VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        };

        let tokenizer_filename = download_hf_hub_file(&self.model_id, "tokenizer.json")?;
        let transformer = Transformer::load(
            vb,
            &config_filename,
            &tokenizer_filename,
            sbert_config.max_seq_length,
        )?;

        // load the pooler
        let pooling_config_filename = download_hf_hub_file(&self.model_id, &pooling_method)?;
        let pooling_config = load_config::<PoolingConfig>(&pooling_config_filename)?;
        let pooler = PoolingStrategy::from_config(pooling_config);

        // Load the dense layers
        let mut dense_layers = vec![];
        for dense_path in self.dense_paths.iter() {
            let dense_config_filename =
                download_hf_hub_file(&self.model_id, &format!("{dense_path}/config.json"))?;
            let dense_config = load_config::<DenseConfig>(&dense_config_filename)?;

            let dense_vb = if self.with_safetensors {
                let weights_filename = download_hf_hub_file(
                    &self.model_id,
                    &format!("{dense_path}/model.safetensors"),
                )?;
                load_safetensors(&[weights_filename], DTYPE, &device)?
            } else {
                let weights_filename = download_hf_hub_file(
                    &self.model_id,
                    &format!("{dense_path}/pytorch_model.bin"),
                )?;
                candle_nn::VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
            };

            let layer = Dense::from_config(dense_vb, dense_config)?;
            dense_layers.push(layer);
        }

        // normalize
        let normalize = if self.with_normalization {
            Some(Normalize)
        } else {
            None
        };

        Ok(SentenceTransformer {
            model_config: model_config,
            batch_size: self.batch_size,
            device: device,
            transformer: transformer,
            pooler: pooler,
            dense_layers: dense_layers,
            normalize: normalize,
        })
    }
}

pub struct SentenceTransformer {
    model_config: ModelConfig,
    batch_size: usize,
    device: Device,
    transformer: Transformer,
    pooler: PoolingStrategy,
    dense_layers: Vec<Dense>,
    normalize: Option<Normalize>,
}

impl SentenceTransformer {
    pub fn embed(&self, lines: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut embeddings: Vec<Vec<f32>> = vec![];

        let batches = self.transformer.tokenize(
            lines,
            &self.device,
            self.model_config
                .pad_token_id
                .unwrap_or(DEFAULT_PAD_TOKEN_ID),
            self.batch_size,
        )?;

        for batch in batches.batches.iter() {
            // transformer
            let mut batch_embeddings = self.transformer.forward(batch)?;

            // pool
            batch_embeddings = self
                .pooler
                .forward(&batch_embeddings, &batch.attention_mask)?;

            // dense
            for dense in self.dense_layers.iter() {
                batch_embeddings = dense.forward(&batch_embeddings)?;
            }

            // norm
            batch_embeddings = if let Some(norm) = &self.normalize {
                norm.forward(&batch_embeddings)
            } else {
                Ok(batch_embeddings)
            }?;

            // add batch embeddings to final embeddings - still unsorted at this point
            for emb in batch_embeddings.to_vec2()?.into_iter() {
                embeddings.push(emb);
            }
        }

        let mut sorted_embeddings = vec![vec![]; embeddings.len()];
        for (emb, (idx, _)) in embeddings.into_iter().zip(batches.original_ids) {
            sorted_embeddings[idx] = emb;
        }

        Ok(sorted_embeddings)
    }
}
