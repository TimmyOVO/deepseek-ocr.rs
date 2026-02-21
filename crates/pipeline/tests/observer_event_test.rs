use std::time::Duration;

use deepseek_ocr_pipeline::{ModelKind, OcrModelId, OcrPipelineEvent};

fn model(id: &str) -> OcrModelId {
    OcrModelId::try_from(id).expect("valid model id")
}

#[test]
fn serialize_includes_all_fields_and_formats_duration() {
    let event = OcrPipelineEvent::ModelLoadFinished {
        model_id: model("deepseek-ocr"),
        kind: ModelKind::Deepseek,
        flash_attention: true,
        duration: Duration::from_millis(1234),
    };

    let json = serde_json::to_string(&event).expect("serialize");

    assert!(json.contains("\"kind\":\"model_load_finished\""));
    assert!(json.contains("\"model_id\":\"deepseek-ocr\""));
    assert!(json.contains("\"model_kind\":\"deepseek\""));
    assert!(json.contains("\"flash_attention\":true"));
    assert!(json.contains("\"duration_s\":\"1.234s\""));
}

#[test]
fn display_is_human_readable() {
    let event = OcrPipelineEvent::GenerationFinished {
        model_id: model("paddleocr-vl"),
        prompt_tokens: 10,
        response_tokens: 20,
        duration: Duration::from_micros(5_550_000),
    };

    let text = event.to_string();

    assert!(text.starts_with("GenerationFinished paddleocr-vl"));
    assert!(text.contains("prompt_tokens=10"));
    assert!(text.contains("response_tokens=20"));
    assert!(text.contains("5.550s"));
}

#[test]
fn duration_format_matches_three_decimal_seconds() {
    let event = OcrPipelineEvent::ModelLoadFinished {
        model_id: model("dots-ocr"),
        kind: ModelKind::DotsOcr,
        flash_attention: false,
        duration: Duration::from_nanos(987_654_321),
    };

    let json = serde_json::to_string(&event).expect("serialize");

    assert!(json.contains("\"duration_s\":\"0.988s\""));
}
