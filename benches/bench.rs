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
fn bench_paraphrase_multilingual_mpnet_base_v2(c: &mut Criterion) {
    run_bench(&Which::ParaphraseMultilingualMpnetBaseV2, c);
}
fn bench_distiluse_base_multilingual_cased_v2(c: &mut Criterion) {
    run_bench(&Which::DistiluseBaseMultilingualCasedV2, c);
}
fn bench_bge_small_en_v1_5(c: &mut Criterion) {
    run_bench(&Which::BgeSmallEnV1_5, c);
}
fn bench_multilingual_e5_large(c: &mut Criterion) {
    run_bench(&Which::MultilingualE5Large, c);
}
fn bench_multilingual_e5_base(c: &mut Criterion) {
    run_bench(&Which::MultilingualE5Base, c);
}
fn bench_multilingual_e5_small(c: &mut Criterion) {
    run_bench(&Which::MultilingualE5Small, c);
}
fn bench_all_mpnet_base_v2(c: &mut Criterion) {
    run_bench(&Which::AllMpnetBaseV2, c);
}
fn bench_paraphrase_mpnet_base_v2(c: &mut Criterion) {
    run_bench(&Which::ParaphraseMpnetBaseV2, c);
}
criterion_group!(
    benches,
    bench_all_mini_lm_l6_v2,
    bench_all_mini_lm_l12_v2,
    bench_labse,
    bench_paraphrase_mini_lm_l6_v2,
    bench_paraphrase_multilingual_mini_lm_l12_v2,
    bench_paraphrase_multilingual_mpnet_base_v2,
    bench_distiluse_base_multilingual_cased_v2,
    bench_bge_small_en_v1_5,
    bench_multilingual_e5_large,
    bench_multilingual_e5_base,
    bench_multilingual_e5_small,
    bench_all_mpnet_base_v2,
    bench_paraphrase_mpnet_base_v2
);
criterion_main!(benches);
