use std::sync::{Arc, Mutex};

use anyhow::Result;
use candle_core::{DType, Device};
use deepseek_ocr_core::{inference::StreamCallback, DecodeOutcome, OcrEngine, OcrInferenceEngine};
use deepseek_ocr_pipeline::{
    DecodeParameters, ModelKind, OcrModelId, OcrPipeline, OcrPipelineHandle, OcrPrompt, OcrRequest,
    VisionSettings,
};
use image::DynamicImage;
use tokenizers::Tokenizer;

struct DummyEngine {
    device: Device,
    text: String,
    generated_tokens: Vec<i64>,
}

impl DummyEngine {
    fn new(text: &str, generated_tokens: Vec<i64>) -> Self {
        Self {
            device: Device::Cpu,
            text: text.to_string(),
            generated_tokens,
        }
    }
}

impl OcrEngine for DummyEngine {
    fn kind(&self) -> ModelKind {
        ModelKind::Deepseek
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        DType::F32
    }

    fn decode(
        &self,
        _tokenizer: &Tokenizer,
        _prompt: &str,
        _images: &[DynamicImage],
        _vision: VisionSettings,
        _params: &DecodeParameters,
        stream: StreamCallback,
    ) -> Result<DecodeOutcome> {
        if let Some(callback) = stream {
            callback(1, &self.generated_tokens);
        }

        Ok(DecodeOutcome {
            text: self.text.clone(),
            prompt_tokens: 3,
            response_tokens: self.generated_tokens.len(),
            generated_tokens: self.generated_tokens.clone(),
        })
    }
}

fn build_tokenizer() -> Tokenizer {
    let tokenizer_json = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]": 0},
        "unk_token": "[UNK]"
      }
    }"#;
    Tokenizer::from_bytes(tokenizer_json.as_bytes()).expect("tokenizer json should parse")
}

fn base_request() -> OcrRequest {
    OcrRequest {
        prompt: OcrPrompt::Raw("<image>\nhello".to_string()),
        template: "plain".to_string(),
        system_prompt: String::new(),
        images: vec![DynamicImage::new_rgba8(1, 1)],
        vision: VisionSettings {
            base_size: 1024,
            image_size: 640,
            crop_mode: true,
        },
        decode: DecodeParameters::with_sampling_defaults(8),
    }
}

fn assert_send_sync<T: Send + Sync>() {}

fn assert_clone<T: Clone>() {}

fn assert_as_ref<T: AsRef<OcrPipeline>>() {}

#[test]
fn pipeline_handle_is_send_sync_clone() {
    assert_send_sync::<OcrPipelineHandle>();
    assert_clone::<OcrPipelineHandle>();
    assert_as_ref::<OcrPipelineHandle>();
}

#[test]
fn handle_generate_delegates_to_inner_pipeline() -> Result<()> {
    let pipeline = OcrPipeline::from_loaded(
        OcrModelId::try_from("deepseek-ocr")?,
        Arc::new(Mutex::new(Box::new(DummyEngine::new(
            "handle ok",
            vec![11, 22],
        )))),
        Arc::new(build_tokenizer()),
        Arc::new(OcrInferenceEngine::with_default_semantics(
            ModelKind::Deepseek,
        )),
    );
    let handle = OcrPipelineHandle::new(pipeline);

    let stream_invocations = Arc::new(Mutex::new(Vec::<(usize, Vec<i64>)>::new()));
    let stream_capture = Arc::clone(&stream_invocations);
    let stream = move |count: usize, tokens: &[i64]| {
        stream_capture
            .lock()
            .expect("stream capture mutex should not be poisoned")
            .push((count, tokens.to_vec()));
    };

    let response = handle.generate(&base_request(), None, Some(&stream))?;

    assert_eq!(response.text, "handle ok");
    assert_eq!(response.generated_tokens, vec![11, 22]);
    assert_eq!(handle.as_ref().tokenizer()?.get_vocab_size(false), 1);

    let stream_events = stream_invocations
        .lock()
        .expect("stream capture mutex should not be poisoned");
    assert_eq!(stream_events.len(), 1);
    assert_eq!(stream_events[0].0, 1);
    assert_eq!(stream_events[0].1, vec![11, 22]);

    Ok(())
}
