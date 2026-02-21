use deepseek_ocr_pipeline::OcrModelId;

#[test]
fn rejects_empty_model_id() {
    let err = OcrModelId::try_from("").expect_err("empty model_id should error");
    assert!(err.to_string().contains("non-empty"));
}

#[test]
fn rejects_whitespace_model_id() {
    let err = OcrModelId::try_from("   \t").expect_err("whitespace-only model_id should error");
    assert!(err.to_string().contains("non-empty"));
}

#[test]
fn rejects_model_id_with_inner_whitespace() {
    let err = OcrModelId::try_from("abc def").expect_err("model_id with spaces should error");
    assert!(err.to_string().contains("whitespace"));
}

#[test]
fn accepts_valid_model_id() {
    let id = OcrModelId::try_from("deepseek-ocr-q4k").expect("valid model_id should parse");
    assert_eq!(id.as_str(), "deepseek-ocr-q4k");
}

#[test]
fn known_models_matches_expected() {
    let expected = vec![
        "deepseek-ocr",
        "deepseek-ocr-q4k",
        "deepseek-ocr-q6k",
        "deepseek-ocr-q8k",
        "paddleocr-vl",
        "paddleocr-vl-q4k",
        "paddleocr-vl-q6k",
        "paddleocr-vl-q8k",
        "dots-ocr",
        "dots-ocr-q4k",
        "dots-ocr-q6k",
        "dots-ocr-q8k",
        "glm-ocr",
    ];

    let models = OcrModelId::known_models();
    assert_eq!(models, expected);
}
