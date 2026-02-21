use std::sync::{Arc, Mutex};

use anyhow::Result;
use candle_core::{DType, Device};
use deepseek_ocr_core::{inference::StreamCallback, DecodeOutcome, OcrEngine, OcrInferenceEngine};
use deepseek_ocr_pipeline::{
    DecodeParameters, ModelKind, OcrMessage, OcrModelId, OcrPipeline, OcrPipelineEvent, OcrPrompt,
    OcrRequest, OcrRole, VisionSettings,
};
use image::DynamicImage;
use tokenizers::Tokenizer;

#[derive(Default)]
struct CapturingObserver {
    events: Mutex<Vec<OcrPipelineEvent>>,
}

impl deepseek_ocr_pipeline::OcrPipelineObserver for CapturingObserver {
    fn on_event(&self, event: &OcrPipelineEvent) {
        self.events
            .lock()
            .expect("observer events mutex should not be poisoned")
            .push(event.clone());
    }
}

struct DummyEngine {
    device: Device,
    text: String,
    prompt_tokens: usize,
    response_tokens: usize,
    generated_tokens: Vec<i64>,
    last_prompt: Arc<Mutex<Option<String>>>,
}

impl DummyEngine {
    fn new(
        text: &str,
        prompt_tokens: usize,
        response_tokens: usize,
        generated_tokens: Vec<i64>,
    ) -> Self {
        Self {
            device: Device::Cpu,
            text: text.to_string(),
            prompt_tokens,
            response_tokens,
            generated_tokens,
            last_prompt: Arc::new(Mutex::new(None)),
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
        prompt: &str,
        _images: &[DynamicImage],
        _vision: VisionSettings,
        _params: &DecodeParameters,
        stream: StreamCallback,
    ) -> Result<DecodeOutcome> {
        *self
            .last_prompt
            .lock()
            .expect("prompt capture mutex should not be poisoned") = Some(prompt.to_string());

        if let Some(callback) = stream {
            callback(1, &self.generated_tokens);
        }

        Ok(DecodeOutcome {
            text: self.text.clone(),
            prompt_tokens: self.prompt_tokens,
            response_tokens: self.response_tokens,
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

fn base_request(prompt: OcrPrompt) -> OcrRequest {
    OcrRequest {
        prompt,
        template: "plain".to_string(),
        system_prompt: String::new(),
        images: vec![DynamicImage::new_rgba8(1, 1)],
        vision: VisionSettings {
            base_size: 1024,
            image_size: 640,
            crop_mode: true,
        },
        decode: DecodeParameters::with_sampling_defaults(16),
    }
}

#[test]
fn generate_raw_prompt_streams_and_emits_observer_events() -> Result<()> {
    let dummy = DummyEngine::new("decoded text", 12, 3, vec![101, 102, 103]);
    let observer = Arc::new(CapturingObserver::default());

    let pipeline = OcrPipeline::from_loaded(
        OcrModelId::try_from("deepseek-ocr")?,
        Arc::new(Mutex::new(Box::new(dummy))),
        Arc::new(build_tokenizer()),
        Arc::new(OcrInferenceEngine::with_default_semantics(
            ModelKind::Deepseek,
        )),
    )
    .with_observer(observer.clone());

    let req = base_request(OcrPrompt::Raw("<image>\nextract text".to_string()));

    let stream_invocations = Arc::new(Mutex::new(Vec::<(usize, Vec<i64>)>::new()));
    let stream_capture = Arc::clone(&stream_invocations);
    let stream = move |count: usize, tokens: &[i64]| {
        stream_capture
            .lock()
            .expect("stream capture mutex should not be poisoned")
            .push((count, tokens.to_vec()));
    };

    let response = pipeline.generate(&req, None, Some(&stream))?;

    assert_eq!(response.text, "decoded text");
    assert_eq!(response.prompt_tokens, 12);
    assert_eq!(response.response_tokens, 3);
    assert_eq!(response.generated_tokens, vec![101, 102, 103]);
    assert!(!response.rendered_prompt.is_empty());

    let stream_events = stream_invocations
        .lock()
        .expect("stream capture mutex should not be poisoned");
    assert_eq!(stream_events.len(), 1);
    assert_eq!(stream_events[0].0, 1);
    assert_eq!(stream_events[0].1, vec![101, 102, 103]);

    let events = observer
        .events
        .lock()
        .expect("observer events mutex should not be poisoned");
    assert_eq!(events.len(), 2);
    match &events[0] {
        OcrPipelineEvent::GenerationStarted {
            model_id,
            max_new_tokens,
        } => {
            assert_eq!(model_id.as_str(), "deepseek-ocr");
            assert_eq!(*max_new_tokens, req.decode.max_new_tokens);
        }
        other => panic!("unexpected first event: {other:?}"),
    }
    match &events[1] {
        OcrPipelineEvent::GenerationFinished {
            model_id,
            prompt_tokens,
            response_tokens,
            duration,
        } => {
            assert_eq!(model_id.as_str(), "deepseek-ocr");
            assert_eq!(*prompt_tokens, 12);
            assert_eq!(*response_tokens, 3);
            assert!(*duration >= std::time::Duration::ZERO);
        }
        other => panic!("unexpected second event: {other:?}"),
    }

    Ok(())
}

#[test]
fn generate_messages_prompt_maps_roles_and_succeeds() -> Result<()> {
    let dummy = DummyEngine::new("messages ok", 7, 2, vec![201, 202]);
    let prompt_capture = Arc::clone(&dummy.last_prompt);

    let pipeline = OcrPipeline::from_loaded(
        OcrModelId::try_from("deepseek-ocr")?,
        Arc::new(Mutex::new(Box::new(dummy))),
        Arc::new(build_tokenizer()),
        Arc::new(OcrInferenceEngine::with_default_semantics(
            ModelKind::Deepseek,
        )),
    );

    let req = base_request(OcrPrompt::Messages(vec![
        OcrMessage {
            role: OcrRole::System,
            content: "system-rule".to_string(),
        },
        OcrMessage {
            role: OcrRole::User,
            content: "<image>\nuser-question".to_string(),
        },
        OcrMessage {
            role: OcrRole::Assistant,
            content: "assistant-context".to_string(),
        },
    ]));

    let response = pipeline.generate(&req, None, None)?;

    assert_eq!(response.text, "messages ok");
    assert_eq!(response.prompt_tokens, 7);
    assert_eq!(response.response_tokens, 2);
    assert_eq!(response.generated_tokens, vec![201, 202]);

    let rendered = prompt_capture
        .lock()
        .expect("prompt capture mutex should not be poisoned")
        .clone()
        .expect("dummy engine should capture rendered prompt");
    assert!(rendered.contains("<image>"));
    assert!(rendered.contains("user-question"));
    assert!(rendered.contains("assistant-context"));

    Ok(())
}

#[test]
fn generate_returns_error_when_pipeline_not_initialized() {
    let req = base_request(OcrPrompt::Raw("<image>\nanything".to_string()));
    let pipeline = OcrPipeline::new();

    let err = pipeline
        .generate(&req, None, None)
        .expect_err("uninitialized pipeline should return an error");

    assert!(err
        .to_string()
        .contains("pipeline model_id is not initialized"));
}
