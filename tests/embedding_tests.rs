use sentence_transformers_rs::error::CosineSimilarityError;
use sentence_transformers_rs::sentence_transformer::{SentenceTransformerBuilder, Which};
use sentence_transformers_rs::utils::cosine_similarity;

const SENTS_1: &'static [&'static str] = &[
    "Let's explore the key differences and improvements in Gemma 3.",
    "Midnight Commander is a feature-rich, full-screen, text-mode application that allows you to copy, move, and delete files and entire directory trees, search for files, and execute commands in the subshell. Internal viewer, editor and diff viewer are included.",
    "את יכולה לחזור על זה?",
    "I can also show a one-liner version that computes the average similarity and propagates errors without creating an intermediate Vec if you want."
];
const SENTS_2: &'static [&'static str] = &[
    "garblegarble fargle",
    "미드나이트 커맨더(Midnight Commander)는 다양한 기능을 갖춘 풀스크린 텍스트 모드 애플리케이션으로, 파일과 전체 디렉터리 트리를 복사, 이동, 삭제할 수 있으며, 파일 검색과 서브셸에서의 명령 실행을 지원합니다. 또한 내부 뷰어, 편집기, 그리고 차이 비교 뷰어(diff viewer)가 포함되어 있습니다.",
    "Could you repeat that?",
    "Ich kann Ihnen auch eine Einzeiler-Version zeigen, die die durchschnittliche Ähnlichkeit berechnet und Fehler weitergibt, ohne einen Zwischen-Vec zu erstellen, wenn Sie möchten."
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

    let sims = sent1_emb
        .iter()
        .zip(&sent2_emb)
        .map(|(a, b)| cosine_similarity(a, b))
        .collect::<Result<Vec<f32>, CosineSimilarityError>>()?;

    for (sim, expected_sim) in sims.iter().zip(exptected) {
        assert!(
            (sim - expected_sim).abs() < epsilon,
            "Similarities(Rust: {:?}, Python: {:?}) do not match",
            sims,
            exptected
        );
    }

    Ok(())
}

#[test]
fn test_labse() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.10413145273923874,
        0.8810012936592102,
        0.9385778903961182,
        0.8646234273910522,
    ];

    run_model_test(Which::LaBSE, exptected_values)?;

    Ok(())
}

#[test]
fn test_all_mini_lm_l12_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.08386149257421494,
        0.6966043710708618,
        0.14756658673286438,
        0.19373546540737152,
    ];

    run_model_test(Which::AllMiniLML12v2, exptected_values)?;

    Ok(())
}

#[test]
fn test_all_mini_lm_l6_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.014937047846615314,
        0.39104771614074707,
        0.08823097497224808,
        0.32795071601867676,
    ];

    run_model_test(Which::AllMiniLML6v2, exptected_values)?;

    Ok(())
}

#[test]
fn test_paraphrase_multilingual_mini_lm_l12_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.03271716460585594,
        0.86850905418396,
        0.9263136386871338,
        0.8704717755317688,
    ];

    run_model_test(Which::ParaphraseMultilingualMiniLML12v2, exptected_values)?;

    Ok(())
}

#[test]
fn test_paraphrase_mini_lm_l6_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.06461339443922043,
        0.36720430850982666,
        0.12729252874851227,
        0.23750200867652893,
    ];

    run_model_test(Which::ParaphraseMiniLML6v2, exptected_values)?;

    Ok(())
}

#[test]
fn test_paraphrase_multilingual_mpnet_base_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.1816011220216751,
        0.8937023282051086,
        0.9709388017654419,
        0.8515507578849792,
    ];

    run_model_test(Which::ParaphraseMultilingualMpnetBaseV2, exptected_values)?;

    Ok(())
}

#[test]
fn test_distiluse_base_multilingual_cased_v2() -> Result<(), Box<dyn std::error::Error>> {
    let exptected_values = &[
        0.011966128833591938,
        0.8534433841705322,
        0.9190109372138977,
        0.9041813015937805,
    ];

    run_model_test(Which::DistiluseBaseMultilingualCasedV2, exptected_values)?;

    Ok(())
}
