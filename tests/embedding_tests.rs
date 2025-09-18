use sentence_transformers_rs::sentence_transformer::{SentenceTransformerBuilder, Which};
use sentence_transformers_rs::utils::cosine_similarity;

const SENTS_1: &'static [&'static str] = &[
    "Let's explore the key differences and improvements in Gemma 3.",
    "Midnight Commander is a feature-rich, full-screen, text-mode application that allows you to copy, move, and delete files and entire directory trees, search for files, and execute commands in the subshell. Internal viewer, editor and diff viewer are included.",
    "את יכולה לחזור על זה?",
];
const SENTS_2: &'static [&'static str] = &[
    "garblegarble fargle",
    "미드나이트 커맨더(Midnight Commander)는 다양한 기능을 갖춘 풀스크린 텍스트 모드 애플리케이션으로, 파일과 전체 디렉터리 트리를 복사, 이동, 삭제할 수 있으며, 파일 검색과 서브셸에서의 명령 실행을 지원합니다. 또한 내부 뷰어, 편집기, 그리고 차이 비교 뷰어(diff viewer)가 포함되어 있습니다.",
    "Could you repeat that?"
];

fn run_model_test(
    which_model: Which,
    exptected: &'static [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    let device = candle_core::Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = candle_core::Device::Cpu;

    let model = SentenceTransformerBuilder::with_sentence_transformer(&which_model)
        .batch_size(2048)
        .with_device(&device)
        .build()?;

    let sent1_emb = model.embed(&SENTS_1)?;
    let sent2_emb = model.embed(&SENTS_2)?;

    let epsilon = 1e-6;

    for ((s1, s2), expected_sim) in sent1_emb.iter().zip(&sent2_emb).zip(exptected) {
        let sim = cosine_similarity(&s1, &s2)?;
        assert!(
            (sim - expected_sim).abs() < epsilon,
            "Similarities(Rust: {}, Python: {}) do not match",
            sim,
            expected_sim
        );
    }

    Ok(())
}

#[test]
fn test_labse() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[0.10413144528865814, 0.8810012936592102, 0.9385777711868286];

    run_model_test(Which::LaBSE, exptected_values)?;

    Ok(())
}

#[test]
fn test_all_mini_lm_l12_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[0.08386142551898956, 0.6966046094894409, 0.1475667804479599];

    run_model_test(Which::AllMiniLML12v2, exptected_values)?;

    Ok(())
}

#[test]
fn test_all_mini_lm_l6_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.01493704505264759,
        0.39104756712913513,
        0.08823097497224808,
    ];

    run_model_test(Which::AllMiniLML6v2, exptected_values)?;

    Ok(())
}

#[test]
fn test_paraphrase_multilingual_mini_lm_l12_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[0.032717157155275345, 0.8685088753700256, 0.9263136386871338];

    run_model_test(Which::ParaphraseMultilingualMiniLML12v2, exptected_values)?;

    Ok(())
}

#[test]
fn test_paraphrase_mini_lm_l6_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.06461336463689804,
        0.36720430850982666,
        0.12729260325431824,
    ];

    run_model_test(Which::ParaphraseMiniLML6v2, exptected_values)?;

    Ok(())
}
