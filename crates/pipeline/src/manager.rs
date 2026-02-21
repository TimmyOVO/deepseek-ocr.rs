use std::{
    collections::HashMap,
    sync::{Arc, Condvar, Mutex},
    time::Instant,
};

use anyhow::{Context, Result};
use candle_core::DType;
use deepseek_ocr_config::{prepare_model_paths, AppConfig, LocalFileSystem};
use deepseek_ocr_core::{ModelLoadArgs, OcrInferenceEngine};
use deepseek_ocr_infer_deepseek::load_model as load_deepseek_model;
use deepseek_ocr_infer_dots::load_model as load_dots_model;
use deepseek_ocr_infer_glm::load_model as load_glm_model;
use deepseek_ocr_infer_paddleocr::load_model as load_paddle_model;
use tokenizers::Tokenizer;

use crate::{
    observer::{NoopObserver, OcrPipelineEvent},
    pipeline::{OcrPipeline, OcrPipelineHandle},
    DeviceKind, ModelKind, OcrModelId, OcrPipelineObserver, Precision,
};

/// Simple model listing for API/UI purposes.
///
/// 说明：对于 DeepSeek OCR1/OCR2，这两者在系统里是两个不同的 `model_id`
///（例如 `deepseek-ocr` / `deepseek-ocr-2`），因此无需额外 `variant` 字段。
#[derive(Debug, Clone)]
pub struct OcrModelListing {
    pub id: OcrModelId,
    pub kind: ModelKind,
}

/// Model manager for server-style multi-model hosting.
///
/// 设计目标：
/// - 统一管理配置文件与模型资源准备；
/// - 统一缓存已加载模型（避免每请求重复 load）；
/// - 对外只暴露 `OcrPipelineHandle`，不暴露底层 infer/config/assets 细节。
pub struct OcrModelManager {
    fs: LocalFileSystem,
    config: Arc<AppConfig>,
    device: DeviceKind,
    precision: Option<Precision>,
    dtype: DType,
    observer: Arc<dyn OcrPipelineObserver>,
    models: Vec<OcrModelListing>,
    active_model: Option<OcrModelId>,
    cache: Mutex<HashMap<OcrModelId, OcrPipelineHandle>>,
    inflight: Mutex<HashMap<OcrModelId, Arc<LoadEntry>>>,
}

#[derive(Clone)]
enum LoadState {
    InProgress,
    Ready(OcrPipelineHandle),
    Failed(String),
}

struct LoadEntry {
    state: Mutex<LoadState>,
    cv: Condvar,
}

impl LoadEntry {
    fn new() -> Self {
        Self {
            state: Mutex::new(LoadState::InProgress),
            cv: Condvar::new(),
        }
    }
}

impl OcrModelManager {
    pub fn new(
        fs: LocalFileSystem,
        config: Arc<AppConfig>,
        device: DeviceKind,
        precision: Option<Precision>,
        dtype: DType,
    ) -> Self {
        let models = config
            .models
            .entries
            .iter()
            .filter_map(|(id, entry)| {
                OcrModelId::try_from(id.as_str())
                    .ok()
                    .map(|id| OcrModelListing {
                        id,
                        kind: entry.kind,
                    })
            })
            .collect();
        let active_model = OcrModelId::try_from(config.models.active.as_str()).ok();

        Self {
            fs,
            config,
            device,
            precision,
            dtype,
            observer: Arc::new(NoopObserver),
            models,
            active_model,
            cache: Mutex::new(HashMap::new()),
            inflight: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn OcrPipelineObserver>) -> Self {
        self.observer = observer;
        self
    }

    pub fn device_kind(&self) -> DeviceKind {
        self.device
    }

    pub fn precision(&self) -> Option<Precision> {
        self.precision
    }

    pub fn observer(&self) -> &dyn OcrPipelineObserver {
        self.observer.as_ref()
    }

    pub fn available_models(&self) -> &[OcrModelListing] {
        &self.models
    }

    pub fn active_model_id(&self) -> Option<&OcrModelId> {
        self.active_model.as_ref()
    }

    pub fn set_active_model_id(&mut self, model_id: Option<OcrModelId>) {
        self.active_model = model_id;
    }

    pub fn load(&self, model_id: &OcrModelId) -> Result<OcrPipelineHandle> {
        if let Some(handle) = self
            .cache
            .lock()
            .expect("model cache mutex poisoning detected")
            .get(model_id)
            .cloned()
        {
            return Ok(handle);
        }

        let (entry, is_loader) = {
            let mut inflight = self
                .inflight
                .lock()
                .expect("inflight map mutex poisoning detected");
            if let Some(existing) = inflight.get(model_id) {
                (Arc::clone(existing), false)
            } else {
                let entry = Arc::new(LoadEntry::new());
                inflight.insert(model_id.clone(), Arc::clone(&entry));
                (entry, true)
            }
        };

        if !is_loader {
            let mut state = entry
                .state
                .lock()
                .expect("singleflight entry mutex poisoning detected");
            while matches!(*state, LoadState::InProgress) {
                state = entry
                    .cv
                    .wait(state)
                    .expect("singleflight entry wait failed");
            }
            return match &*state {
                LoadState::Ready(handle) => Ok(handle.clone()),
                LoadState::Failed(message) => Err(anyhow::anyhow!(message.clone())),
                LoadState::InProgress => unreachable!("inflight state cannot remain in progress"),
            };
        }

        let load_result = self.load_uncached(model_id);

        let mut state = entry
            .state
            .lock()
            .expect("singleflight entry mutex poisoning detected");
        match &load_result {
            Ok(handle) => {
                self.cache
                    .lock()
                    .expect("model cache mutex poisoning detected")
                    .insert(model_id.clone(), handle.clone());
                *state = LoadState::Ready(handle.clone());
            }
            Err(err) => {
                *state = LoadState::Failed(err.to_string());
            }
        }
        self.inflight
            .lock()
            .expect("inflight map mutex poisoning detected")
            .remove(model_id);
        entry.cv.notify_all();

        load_result
    }

    fn load_uncached(&self, model_id: &OcrModelId) -> Result<OcrPipelineHandle> {
        self.observer.on_event(&OcrPipelineEvent::ModelLoadStarted {
            model_id: model_id.clone(),
        });

        let resources = self
            .config
            .model_resources(&self.fs, model_id.as_str())
            .with_context(|| format!("model `{model_id}` not found in configuration"))?;

        let prepared = prepare_model_paths(
            &self.fs,
            &resources.id,
            &resources.config,
            &resources.tokenizer,
            &resources.weights,
            resources.snapshot.as_ref(),
        )?;

        self.observer
            .on_event(&OcrPipelineEvent::ResourcesPrepared {
                model_id: model_id.clone(),
                config: prepared.config.display().to_string(),
                tokenizer: prepared.tokenizer.display().to_string(),
                weights: prepared.weights.display().to_string(),
                snapshot: prepared.snapshot.as_ref().map(|p| p.display().to_string()),
            });

        let (device, _dtype) =
            deepseek_ocr_core::runtime::prepare_device_and_dtype(self.device, self.precision)?;

        let load_args = ModelLoadArgs {
            kind: resources.kind,
            config_path: Some(&prepared.config),
            weights_path: Some(&prepared.weights),
            snapshot_path: prepared.snapshot.as_deref(),
            device,
            dtype: self.dtype,
        };

        let start = Instant::now();
        let model = match resources.kind {
            ModelKind::Deepseek => load_deepseek_model(load_args)?,
            ModelKind::PaddleOcrVl => load_paddle_model(load_args)?,
            ModelKind::DotsOcr => load_dots_model(load_args)?,
            ModelKind::GlmOcr => load_glm_model(load_args)?,
        };
        let duration = start.elapsed();
        let flash_attention = model.flash_attention_enabled();

        let tokenizer = Arc::new(Tokenizer::from_file(&prepared.tokenizer).map_err(|err| {
            anyhow::anyhow!(
                "failed to load tokenizer from {}: {err}",
                prepared.tokenizer.display()
            )
        })?);
        let engine = Arc::new(OcrInferenceEngine::with_default_semantics(resources.kind));

        let pipeline = OcrPipeline::from_loaded(
            model_id.clone(),
            Arc::new(Mutex::new(model)),
            tokenizer,
            engine,
        )
        .with_observer(Arc::clone(&self.observer));

        self.observer
            .on_event(&OcrPipelineEvent::ModelLoadFinished {
                model_id: model_id.clone(),
                kind: resources.kind,
                flash_attention,
                duration,
            });

        Ok(OcrPipelineHandle::new(pipeline))
    }

    pub fn load_active(&self) -> Result<OcrPipelineHandle> {
        let Some(model_id) = self.active_model_id() else {
            anyhow::bail!("active model is not configured")
        };
        self.load(model_id)
    }

    pub fn build_pipeline(&self) -> Result<OcrPipeline> {
        let handle = self.load_active()?;
        Ok(handle.pipeline().clone())
    }
}
