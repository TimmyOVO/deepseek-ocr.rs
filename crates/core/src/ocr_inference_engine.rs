use std::{borrow::Cow, sync::Arc};

use anyhow::{Result, ensure};
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::{
    inference::{
        DecodeParameters, DecodeOutcome, ModelKind, OcrEngine, StreamCallback, VisionSettings,
        normalize_text, render_conversation, render_prompt,
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
pub struct OcrPromptRenderRequest<'a> {
    pub prompt: OcrPromptInput<'a>,
    pub template: &'a str,
    pub system_prompt: &'a str,
    pub image_count: usize,
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
    pub rendered_prompt: String,
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

    fn render_prompt(&self, req: &OcrPromptRenderRequest<'_>) -> Result<RenderedPrompt>;

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

    fn split_system_messages(
        messages: &[OcrPromptMessage],
    ) -> (Vec<String>, Vec<(String, Option<String>)>) {
        let mut system_sections = Vec::new();
        let mut payload = Vec::new();

        for message in messages {
            let text = message.content.trim();
            if text.is_empty() {
                continue;
            }

            match message.role {
                OcrPromptRole::System => system_sections.push(text.to_owned()),
                OcrPromptRole::User => payload.push(("User".to_string(), Some(text.to_owned()))),
                OcrPromptRole::Assistant => {
                    payload.push(("Assistant".to_string(), Some(text.to_owned())))
                }
            }
        }

        // Mirror Transformers `add_generation_prompt=True` by ensuring the
        // rendered prompt ends with an assistant slot.
        let needs_generation_prompt = payload.last().is_none_or(|(role, content)| {
            role != "Assistant" || content.as_ref().is_some_and(|value| !value.trim().is_empty())
        });
        if needs_generation_prompt {
            payload.push(("Assistant".to_string(), None));
        }

        (system_sections, payload)
    }
}

impl ModelSemantics for DefaultModelSemantics {
    fn kind(&self) -> ModelKind {
        self.kind
    }

    fn render_prompt(&self, req: &OcrPromptRenderRequest<'_>) -> Result<RenderedPrompt> {
        let rendered = match req.prompt {
            OcrPromptInput::Raw(raw) => render_prompt(req.template, req.system_prompt, raw)?,
            OcrPromptInput::Rendered(rendered) => rendered.to_owned(),
            OcrPromptInput::Messages(messages) => {
                let (system_sections, payload) = Self::split_system_messages(messages);
                let mut system_prompt = String::new();
                if !req.system_prompt.trim().is_empty() {
                    system_prompt.push_str(req.system_prompt.trim());
                }
                for section in system_sections {
                    if system_prompt.is_empty() {
                        system_prompt.push_str(&section);
                    } else {
                        system_prompt.push_str("\n\n");
                        system_prompt.push_str(&section);
                    }
                }
                render_conversation(req.template, &system_prompt, &payload)?
            }
        };
        let image_slots = rendered.matches("<image>").count();
        let prompt = RenderedPrompt {
            text: rendered,
            image_slots,
        };
        prompt.validate_image_count(req.image_count)?;
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

        let rendered = self.semantics.render_prompt(&OcrPromptRenderRequest {
            prompt: req.prompt,
            template: req.template,
            system_prompt: req.system_prompt,
            image_count: req.images.len(),
        })?;
        let prepared = self.semantics.prepare_inputs(req, &rendered, state)?;
        let outcome = model.decode(
            tokenizer,
            prepared.prompt.as_ref(),
            prepared.images,
            prepared.vision,
            prepared.decode,
            stream,
        )?;
        Ok(self.postprocess(rendered, outcome))
    }

    fn postprocess(&self, rendered: RenderedPrompt, outcome: DecodeOutcome) -> OcrInferenceResult {
        OcrInferenceResult {
            text: self.semantics.postprocess_text(&outcome.text),
            rendered_prompt: rendered.text,
            prompt_tokens: outcome.prompt_tokens,
            response_tokens: outcome.response_tokens,
            generated_tokens: outcome.generated_tokens,
        }
    }
}
