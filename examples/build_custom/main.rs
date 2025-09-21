use sentence_transformers_rs::{
    normalize::Normalizer, sentence_transformer::SentenceTransformerBuilder,
    utils::cosine_similarity,
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
