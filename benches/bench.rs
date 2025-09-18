use criterion::{criterion_group, criterion_main, Criterion};
use sentence_transformers_rs::sentence_transformer::{SentenceTransformerBuilder, Which};

use rand::distr::Alphanumeric;
use rand::Rng;

fn random_sentence() -> String {
    let mut rng = rand::rng();
    let len = rng.random_range(64..512);
    rng.sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

fn run_bench(which_model: &Which, c: &mut Criterion) {
    #[cfg(feature = "cuda")]
    let device = candle_core::Device::new_cuda(0).unwrap();
    #[cfg(not(feature = "cuda"))]
    let device = candle_core::Device::Cpu;

    let model = SentenceTransformerBuilder::with_sentence_transformer(which_model)
        .batch_size(2048)
        .with_device(&device)
        .build()
        .unwrap();

    let sentences: Vec<String> = (0..100).map(|_| random_sentence()).collect();
    let sentences: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();

    c.bench_function(which_model.to_string().as_ref(), |b| {
        b.iter(|| {
            let embeddings = model.embed(&sentences).unwrap();
            criterion::black_box(embeddings);
        })
    });
}

fn bench_all_mini_lm_l6_v2(c: &mut Criterion) {
    run_bench(&Which::AllMiniLML6v2, c);
}
fn bench_all_mini_lm_l12_v2(c: &mut Criterion) {
    run_bench(&Which::AllMiniLML12v2, c);
}
fn bench_labse(c: &mut Criterion) {
    run_bench(&Which::LaBSE, c);
}
fn bench_paraphrase_mini_lm_l6_v2(c: &mut Criterion) {
    run_bench(&Which::ParaphraseMiniLML6v2, c);
}
fn bench_paraphrase_multilingual_mini_lm_l12_v2(c: &mut Criterion) {
    run_bench(&Which::ParaphraseMultilingualMiniLML12v2, c);
}

criterion_group!(
    benches,
    bench_all_mini_lm_l6_v2,
    bench_all_mini_lm_l12_v2,
    bench_labse,
    bench_paraphrase_mini_lm_l6_v2,
    bench_paraphrase_multilingual_mini_lm_l12_v2
);
criterion_main!(benches);
