use std::{borrow::Cow, sync::Arc};

use anyhow::{Result, ensure};
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::{
    inference::{
        DecodeParameters, DecodeOutcome, ModelKind, OcrEngine, StreamCallback, VisionSettings,
        normalize_text, render_prompt,
    },
    runtime_backend::{RuntimeState, RuntimeStateGuard},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrPromptRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct OcrPromptMessage {
    pub role: OcrPromptRole,
    pub content: String,
}

impl OcrPromptMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: OcrPromptRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: OcrPromptRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: OcrPromptRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OcrPromptInput<'a> {
    Raw(&'a str),
    Rendered(&'a str),
    Messages(&'a [OcrPromptMessage]),
}

#[derive(Debug, Clone, Copy)]
pub struct OcrInferenceRequest<'a> {
    pub prompt: OcrPromptInput<'a>,
    pub template: &'a str,
    pub system_prompt: &'a str,
    pub images: &'a [DynamicImage],
    pub vision: VisionSettings,
    pub decode: &'a DecodeParameters,
}

#[derive(Debug, Clone)]
pub struct OcrInferenceResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub response_tokens: usize,
    pub generated_tokens: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct RenderedPrompt {
    pub text: String,
    pub image_slots: usize,
}

impl RenderedPrompt {
    pub fn validate_image_count(&self, image_count: usize) -> Result<()> {
        ensure!(
            self.image_slots == image_count,
            "rendered prompt expects {} <image> slots but got {} images",
            self.image_slots,
            image_count
        );
        Ok(())
    }
}

pub struct PreparedInputs<'a> {
    pub prompt: Cow<'a, str>,
    pub images: &'a [DynamicImage],
    pub vision: VisionSettings,
    pub decode: &'a DecodeParameters,
}

pub trait ModelSemantics: Send + Sync {
    fn kind(&self) -> ModelKind;

    fn render_prompt(&self, req: &OcrInferenceRequest<'_>) -> Result<RenderedPrompt>;

    fn prepare_inputs<'a>(
        &self,
        req: &'a OcrInferenceRequest<'a>,
        rendered: &RenderedPrompt,
        state: Option<&mut RuntimeState>,
    ) -> Result<PreparedInputs<'a>>;

    fn postprocess_text(&self, decoded: &str) -> String {
        normalize_text(decoded)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DefaultModelSemantics {
    kind: ModelKind,
}

impl DefaultModelSemantics {
    pub fn new(kind: ModelKind) -> Self {
        Self { kind }
    }

    fn collapse_messages(messages: &[OcrPromptMessage]) -> String {
        let mut sections = Vec::new();
        for message in messages {
            let text = message.content.trim();
            if text.is_empty() {
                continue;
            }
            sections.push(text.to_owned());
        }
        sections.join("\n\n")
    }
}

impl ModelSemantics for DefaultModelSemantics {
    fn kind(&self) -> ModelKind {
        self.kind
    }

    fn render_prompt(&self, req: &OcrInferenceRequest<'_>) -> Result<RenderedPrompt> {
        let rendered = match req.prompt {
            OcrPromptInput::Raw(raw) => render_prompt(req.template, req.system_prompt, raw)?,
            OcrPromptInput::Rendered(rendered) => rendered.to_owned(),
            OcrPromptInput::Messages(messages) => {
                let collapsed = Self::collapse_messages(messages);
                render_prompt(req.template, req.system_prompt, &collapsed)?
            }
        };
        let image_slots = rendered.matches("<image>").count();
        let prompt = RenderedPrompt {
            text: rendered,
            image_slots,
        };
        prompt.validate_image_count(req.images.len())?;
        Ok(prompt)
    }

    fn prepare_inputs<'a>(
        &self,
        req: &'a OcrInferenceRequest<'a>,
        rendered: &RenderedPrompt,
        state: Option<&mut RuntimeState>,
    ) -> Result<PreparedInputs<'a>> {
        let _ = state;
        Ok(PreparedInputs {
            prompt: Cow::Owned(rendered.text.clone()),
            images: req.images,
            vision: req.vision,
            decode: req.decode,
        })
    }

    fn postprocess_text(&self, decoded: &str) -> String {
        normalize_text(decoded)
    }
}

pub struct OcrInferenceEngine {
    semantics: Arc<dyn ModelSemantics>,
}

impl OcrInferenceEngine {
    pub fn new(semantics: Arc<dyn ModelSemantics>) -> Self {
        Self { semantics }
    }

    pub fn with_default_semantics(kind: ModelKind) -> Self {
        Self::new(Arc::new(DefaultModelSemantics::new(kind)))
    }

    pub fn semantics(&self) -> &dyn ModelSemantics {
        self.semantics.as_ref()
    }

    pub fn prompt_scope_guard<'a>(&self, state: &'a mut RuntimeState) -> RuntimeStateGuard<'a> {
        state.prompt_scope_guard()
    }

    pub fn generate(
        &self,
        model: &dyn OcrEngine,
        tokenizer: &Tokenizer,
        req: &OcrInferenceRequest<'_>,
        state: Option<&mut RuntimeState>,
        stream: StreamCallback<'_>,
    ) -> Result<OcrInferenceResult> {
        ensure!(
            model.kind() == self.semantics.kind(),
            "model kind {:?} does not match semantics {:?}",
            model.kind(),
            self.semantics.kind()
        );

        let rendered = self.semantics.render_prompt(req)?;
        let prepared = self.semantics.prepare_inputs(req, &rendered, state)?;
        let outcome = model.decode(
            tokenizer,
            prepared.prompt.as_ref(),
            prepared.images,
            prepared.vision,
            prepared.decode,
            stream,
        )?;
        Ok(self.postprocess(outcome))
    }

    fn postprocess(&self, outcome: DecodeOutcome) -> OcrInferenceResult {
        OcrInferenceResult {
            text: self.semantics.postprocess_text(&outcome.text),
            prompt_tokens: outcome.prompt_tokens,
            response_tokens: outcome.response_tokens,
            generated_tokens: outcome.generated_tokens,
        }
    }
}
