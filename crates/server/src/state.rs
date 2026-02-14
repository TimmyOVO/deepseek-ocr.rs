use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use tokenizers::Tokenizer;
use tracing::info;

use deepseek_ocr_config::{
    AppConfig, InferenceOverride, InferenceSettings, LocalFileSystem, prepare_model_paths,
};
use deepseek_ocr_core::{
    DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, OcrInferenceEngine, VisionSettings,
};
use deepseek_ocr_infer_deepseek::load_model as load_deepseek_model;
use deepseek_ocr_infer_dots::load_model as load_dots_model;
use deepseek_ocr_infer_glm::load_model as load_glm_model;
use deepseek_ocr_infer_paddleocr::load_model as load_paddle_model;

use crate::error::ApiError;

pub type SharedModel = Arc<Mutex<Box<dyn OcrEngine>>>;
type LoadedModelHandles = (
    SharedModel,
    Arc<OcrInferenceEngine>,
    Arc<Tokenizer>,
    String,
);

#[derive(Clone)]
pub struct ModelListing {
    pub id: String,
    pub kind: ModelKind,
}

pub struct AppState {
    manager: ModelManager,
    current: Mutex<Option<LoadedModel>>,
    base_inference: InferenceSettings,
    inference_overrides: InferenceOverride,
    available_models: Vec<ModelListing>,
}

#[derive(Clone)]
pub struct GenerationInputs {
    pub model: SharedModel,
    pub engine: Arc<OcrInferenceEngine>,
    pub tokenizer: Arc<Tokenizer>,
    pub template: String,
    pub vision: VisionSettings,
    pub defaults: DecodeParameters,
}

impl AppState {
    pub fn bootstrap(
        fs: LocalFileSystem,
        config: Arc<AppConfig>,
        device: Device,
        dtype: DType,
        base_inference: InferenceSettings,
        inference_overrides: InferenceOverride,
    ) -> Result<Self> {
        let available_models = config
            .models
            .entries
            .iter()
            .map(|(id, entry)| ModelListing {
                id: id.clone(),
                kind: entry.kind,
            })
            .collect::<Vec<_>>();

        let manager = ModelManager::new(fs, config, device, dtype);

        Ok(Self {
            manager,
            current: Mutex::new(None),
            base_inference,
            inference_overrides,
            available_models,
        })
    }

    pub fn available_models(&self) -> &[ModelListing] {
        &self.available_models
    }

    pub fn per_model_inference_settings(
        &self,
        model_id: &str,
    ) -> Result<(VisionSettings, DecodeParameters, String), ApiError> {
        let base_config = self.manager.config.as_ref();
        let effective = base_config
            .effective_inference_for_model(
                model_id,
                &self.base_inference,
                &self.inference_overrides,
            )
            .map_err(|err| ApiError::BadRequest(err.to_string()))?;

        let decode = effective.decode.clone();
        let vision = effective.to_vision_settings();
        let template = effective.template.clone();

        Ok((vision, decode, template))
    }

    pub fn prepare_generation(
        &self,
        requested_model: &str,
    ) -> Result<(GenerationInputs, String), ApiError> {
        self.validate_model(requested_model)?;
        let (shared_model, engine, tokenizer, model_id) = self.ensure_model_loaded(requested_model)?;
        let (vision, defaults, template) = self.per_model_inference_settings(requested_model)?;
        let inputs = GenerationInputs {
            model: shared_model,
            engine,
            tokenizer,
            template,
            vision,
            defaults,
        };
        Ok((inputs, model_id))
    }

    fn validate_model(&self, requested: &str) -> Result<(), ApiError> {
        if self
            .available_models
            .iter()
            .any(|entry| entry.id == requested)
        {
            Ok(())
        } else {
            Err(ApiError::BadRequest(format!(
                "requested model `{requested}` is not available"
            )))
        }
    }

    fn ensure_model_loaded(
        &self,
        model_id: &str,
    ) -> Result<LoadedModelHandles, ApiError> {
        {
            if let Ok(guard) = self.current.lock()
                && let Some(loaded) = guard.as_ref()
                && loaded.id == model_id
            {
                return Ok((
                    Arc::clone(&loaded.model),
                    Arc::clone(&loaded.engine),
                    Arc::clone(&loaded.tokenizer),
                    loaded.id.clone(),
                ));
            }
        }

        let loaded = self
            .manager
            .load_model(model_id)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
        let mut guard = self.current.lock().expect("model mutex poisoning detected");
        *guard = Some(loaded);
        let loaded = guard
            .as_ref()
            .expect("loaded model missing after assignment");
        Ok((
            Arc::clone(&loaded.model),
            Arc::clone(&loaded.engine),
            Arc::clone(&loaded.tokenizer),
            loaded.id.clone(),
        ))
    }
}

struct LoadedModel {
    id: String,
    model: SharedModel,
    engine: Arc<OcrInferenceEngine>,
    tokenizer: Arc<Tokenizer>,
}

struct ModelManager {
    fs: LocalFileSystem,
    config: Arc<AppConfig>,
    device: Device,
    dtype: DType,
}

impl ModelManager {
    fn new(fs: LocalFileSystem, config: Arc<AppConfig>, device: Device, dtype: DType) -> Self {
        Self {
            fs,
            config,
            device,
            dtype,
        }
    }

    fn load_model(&self, model_id: &str) -> Result<LoadedModel> {
        let resources = self
            .config
            .model_resources(&self.fs, model_id)
            .with_context(|| format!("model `{model_id}` not found in configuration"))?;
        let prepared = prepare_model_paths(
            &self.fs,
            &resources.id,
            &resources.config,
            &resources.tokenizer,
            &resources.weights,
            resources.snapshot.as_ref(),
        )?;
        let config_path = prepared.config;
        let tokenizer_path = prepared.tokenizer;
        let weights_path = prepared.weights;
        let snapshot_path = prepared.snapshot;

        let load_args = ModelLoadArgs {
            kind: resources.kind,
            config_path: Some(&config_path),
            weights_path: Some(&weights_path),
            snapshot_path: snapshot_path.as_deref(),
            device: self.device.clone(),
            dtype: self.dtype,
        };
        let start = Instant::now();
        let model = match resources.kind {
            ModelKind::Deepseek => load_deepseek_model(load_args)?,
            ModelKind::PaddleOcrVl => load_paddle_model(load_args)?,
            ModelKind::DotsOcr => load_dots_model(load_args)?,
            ModelKind::GlmOcr => load_glm_model(load_args)?,
        };
        info!(
            "Model `{}` loaded in {:.2?} (kind={:?}, flash-attn: {}, weights={})",
            model_id,
            start.elapsed(),
            model.kind(),
            model.flash_attention_enabled(),
            weights_path.display()
        );
        let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow::anyhow!(
                "failed to load tokenizer from {}: {err}",
                tokenizer_path.display()
            )
        })?);
        let engine = Arc::new(OcrInferenceEngine::with_default_semantics(resources.kind));
        Ok(LoadedModel {
            id: model_id.to_string(),
            model: Arc::new(Mutex::new(model)),
            engine,
            tokenizer,
        })
    }
}
