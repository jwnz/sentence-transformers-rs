# Sentence-Transformers-rs

Rust port of [sentence-transformers](https://github.com/UKPLab/sentence-transformers) using the [Candle](https://github.com/huggingface/candle) framework.

## Supported Models

The following embedding models are supported by default; see [Usage: Supported Models](#supported-models):

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
- [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE)
- [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [sentence-transformers/distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)

Additionally, any model based on the `BertModel`, `XLMRobertaModel`, or `DistilBertModel` architectures should also work with some additional boilerplate; see [Usage: Other Models](#other-models) below.


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

You can also build models that aren’t in the supported list, as long as the architecture is based on `BertModel`, `XLMRobertaModel`, or `DistilBertModel` . If you're not sure, check the `"architectures"` field in the repo's [`config.json`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json) file.

```Rust
use sentence_transformers_rs::{
    sentence_transformer::SentenceTransformerBuilder, utils::cosine_similarity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = candle_core::Device::Cpu;

    let model = SentenceTransformerBuilder::new("sentence-transformers/LaBSE")
        // Specify whether to use safetensors to pytorch checkpoints
        .with_safetensors()
        // [OPTIONAL] Use normalization or not - check the modules.json file to see if
        //     the model uses normalization
        .with_normalization(Normalizer::L2)
        // Must specify the folder on the hub that contains the pooling layer config.json.
        .with_pooling("1_Pooling")
        // [OPTIONAL] Specify the folder containing the dense layers spec. Some models
        //     have more than one dense layer. See https://huggingface.co/google/embeddinggemma-300m for example.
        .with_dense("2_Dense")
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

## Todo

- [ ] Support `MPNetMaskedLM` architecture
- [ ] Support `NomicBertModel` architecture
- [ ] Support `MPNetMaskedLM` architecture
- [ ] Support `Gemma3TextModel` architecture
