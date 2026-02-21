use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use deepseek_ocr_core::{
    ocr_inference_engine::{OcrPromptMessage, OcrPromptRole},
    OcrEngine, OcrInferenceEngine, OcrInferenceRequest, OcrPromptInput,
};
use tokenizers::Tokenizer;

use crate::{
    observer::{NoopObserver, OcrPipelineEvent},
    OcrModelId, OcrPipelineObserver, OcrRequest, OcrResponse, OcrStreamCallback,
};

/// An assembled OCR pipeline (model + tokenizer + engine + semantics).
///
/// 设计目标：
/// - 类似 Transformers 的 `pipeline`/`AutoModel` 使用体验：调用者拿到一个对象，
///   直接 `generate(...)`，无需自己拼装 tokenizer/model/engine。
/// - 上层不感知 assets/config/core/infer-* 的细节。
///
/// 当前阶段仅定义 API；实现将在评审后逐步迁移。
#[derive(Clone)]
pub struct OcrPipeline {
    observer: Arc<dyn OcrPipelineObserver>,
    model_id: Option<OcrModelId>,
    model: Option<Arc<Mutex<Box<dyn OcrEngine>>>>,
    tokenizer: Option<Arc<Tokenizer>>,
    engine: Option<Arc<OcrInferenceEngine>>,
}

impl OcrPipeline {
    pub fn new() -> Self {
        Self {
            observer: Arc::new(NoopObserver),
            model_id: None,
            model: None,
            tokenizer: None,
            engine: None,
        }
    }

    pub fn from_loaded(
        model_id: OcrModelId,
        model: Arc<Mutex<Box<dyn OcrEngine>>>,
        tokenizer: Arc<Tokenizer>,
        engine: Arc<OcrInferenceEngine>,
    ) -> Self {
        Self {
            observer: Arc::new(NoopObserver),
            model_id: Some(model_id),
            model: Some(model),
            tokenizer: Some(tokenizer),
            engine: Some(engine),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn OcrPipelineObserver>) -> Self {
        self.observer = observer;
        self
    }

    pub fn observer(&self) -> &dyn OcrPipelineObserver {
        self.observer.as_ref()
    }

    pub fn tokenizer(&self) -> Result<&Tokenizer> {
        let Some(tokenizer) = self.tokenizer.as_ref() else {
            anyhow::bail!("pipeline tokenizer is not initialized")
        };
        Ok(tokenizer.as_ref())
    }

    pub fn generate(
        &self,
        req: &OcrRequest,
        state: Option<&mut deepseek_ocr_core::runtime_backend::RuntimeState>,
        stream: OcrStreamCallback<'_>,
    ) -> Result<OcrResponse> {
        let model_id = self
            .model_id
            .as_ref()
            .cloned()
            .context("pipeline model_id is not initialized")?;
        let model = self
            .model
            .as_ref()
            .context("pipeline model is not initialized")?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("pipeline tokenizer is not initialized")?;
        let engine = self
            .engine
            .as_ref()
            .context("pipeline inference engine is not initialized")?;

        let prompt_messages;
        let prompt = match &req.prompt {
            crate::OcrPrompt::Raw(raw) => OcrPromptInput::Raw(raw.as_str()),
            crate::OcrPrompt::Rendered(rendered) => OcrPromptInput::Rendered(rendered.as_str()),
            crate::OcrPrompt::Messages(messages) => {
                prompt_messages = messages
                    .iter()
                    .map(|message| OcrPromptMessage {
                        role: match message.role {
                            crate::OcrRole::System => OcrPromptRole::System,
                            crate::OcrRole::User => OcrPromptRole::User,
                            crate::OcrRole::Assistant => OcrPromptRole::Assistant,
                        },
                        content: message.content.clone(),
                    })
                    .collect::<Vec<_>>();
                OcrPromptInput::Messages(prompt_messages.as_slice())
            }
        };

        let core_req = OcrInferenceRequest {
            prompt,
            template: req.template.as_str(),
            system_prompt: req.system_prompt.as_str(),
            images: req.images.as_slice(),
            vision: req.vision,
            decode: &req.decode,
        };

        self.observer
            .on_event(&OcrPipelineEvent::GenerationStarted {
                model_id: model_id.clone(),
                max_new_tokens: req.decode.max_new_tokens,
            });

        let start = Instant::now();
        let model_guard = model
            .lock()
            .map_err(|_| anyhow::anyhow!("pipeline model mutex is poisoned"))?;
        let outcome = engine.generate(
            model_guard.as_ref(),
            tokenizer.as_ref(),
            &core_req,
            state,
            stream,
        )?;
        let duration = start.elapsed();

        self.observer
            .on_event(&OcrPipelineEvent::GenerationFinished {
                model_id,
                prompt_tokens: outcome.prompt_tokens,
                response_tokens: outcome.response_tokens,
                duration,
            });

        Ok(OcrResponse {
            text: outcome.text,
            rendered_prompt: outcome.rendered_prompt,
            prompt_tokens: outcome.prompt_tokens,
            response_tokens: outcome.response_tokens,
            generated_tokens: outcome.generated_tokens,
        })
    }
}

impl Default for OcrPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// A cheap-to-clone handle intended for server-style sharing.
///
/// 设计目标：
/// - Server 可能需要跨请求共享已加载的模型；
/// - CLI 则可以直接使用 `OcrPipeline`。
#[derive(Clone)]
pub struct OcrPipelineHandle {
    inner: Arc<OcrPipeline>,
}

impl OcrPipelineHandle {
    pub fn new(pipeline: OcrPipeline) -> Self {
        Self {
            inner: Arc::new(pipeline),
        }
    }

    pub fn pipeline(&self) -> &OcrPipeline {
        self.inner.as_ref()
    }

    pub fn observer(&self) -> &dyn OcrPipelineObserver {
        self.inner.observer()
    }

    pub fn generate(
        &self,
        req: &OcrRequest,
        state: Option<&mut deepseek_ocr_core::runtime_backend::RuntimeState>,
        stream: OcrStreamCallback<'_>,
    ) -> Result<OcrResponse> {
        self.inner.generate(req, state, stream)
    }
}

impl AsRef<OcrPipeline> for OcrPipelineHandle {
    fn as_ref(&self) -> &OcrPipeline {
        self.inner.as_ref()
    }
}
