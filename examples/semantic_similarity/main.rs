use sentence_transformers_rs::{
    sentence_transformer::{SentenceTransformerBuilder, Which},
    utils::cosine_similarity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    let device = candle_core::Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = candle_core::Device::Cpu;

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
