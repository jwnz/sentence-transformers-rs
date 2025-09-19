# Sentence-Transformers-rs

Rust port of [sentence-transformers](https://github.com/UKPLab/sentence-transformers) using the [Candle](https://github.com/huggingface/candle) framework.

## Supported Models

The following models are supported by default:
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
- [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE)

All models not listed above that are based on the `BertModel` architecture should also work with some additional boilerplate; see [Usage: Other Models](#other-models) below.

## Usage

### Supported Models
You can use `with_sentence_transformer` to load any of the supported models:

```Rust
use sentence_transformers_rs::{
    sentence_transformer::{SentenceTransformerBuilder, Which},
    utils::cosine_similarity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = candle_core::Device::new_cuda(0)?;

    let model = SentenceTransformerBuilder::with_sentence_transformer(&Which::AllMiniLML6v2)
        .batch_size(2048)
        .with_device(&device)
        .build()?;

    let sentences = vec!["Hello, World!", "foo bar"];

    let embeddings = model.embed(&sentences)?;

    let sim = cosine_similarity(&embeddings[0], &embeddings[1])?;

    println!("{:?}", sim);

    Ok(())
}
```

### Other Models

You can also use the builder the model yourself if it's not in the supported list of pre-defined models, if the architecture is based on BertModel. If you're not sure, check the `"architectures"` field in the repo's [`config.json`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json) file.

```Rust
use sentence_transformers_rs::{
    sentence_transformer::SentenceTransformerBuilder, utils::cosine_similarity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = candle_core::Device::Cpu;

    let model = SentenceTransformerBuilder::new("sentence-transformers/LaBSE")
        // Specify whether to use safetensors to pytorch checkpoints
        .with_safetensors()
        // [OPTIONAL] Use L2 normalization or not - check the modules.json file to see if
        //     the model uses normalization
        .with_normalization()
        // Must specify the folder on the hub that contains the pooling layer config.json.
        .with_pooling("1_Pooling".to_string())
        // [OPTIONAL] Specify the folder containing the dense layers spec. Some models
        //     have more than one dense layer. See https://huggingface.co/google/embeddinggemma-300m for example.
        .with_dense("2_Dense".to_string())
        // [OPTIONAL] Specify the batch size in tokens.
        .batch_size(2048)
        .with_device(&device)
        .build()?;

    let sentences = vec![
        "To upload your Sentence Transformers models to the Hugging Face Hub",
        "So laden Sie Ihre Sentence Transformers-Modelle zum Hugging Face Hub hoch",
    ];

    let embeddings = model.embed(&sentences)?;

    let sim = cosine_similarity(&embeddings[0], &embeddings[1])?;

    println!("{:?}", sim);

    Ok(())
}
```