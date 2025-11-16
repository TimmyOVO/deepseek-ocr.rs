use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use tokenizers::Tokenizer;
use tracing::info;

use deepseek_ocr_config::{AppConfig, LocalFileSystem};
use deepseek_ocr_core::{DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings};
use deepseek_ocr_infer_deepseek::load_model as load_deepseek_model;
use deepseek_ocr_infer_dots::load_model as load_dots_model;
use deepseek_ocr_infer_paddleocr::load_model as load_paddle_model;

use crate::{
    error::ApiError,
    resources::{
        ensure_config_file, ensure_tokenizer_file, prepare_snapshot_path, prepare_weights_path,
    },
};

pub type SharedModel = Arc<Mutex<Box<dyn OcrEngine>>>;

#[derive(Clone)]
pub struct ModelListing {
    pub id: String,
    pub kind: ModelKind,
}

pub struct AppState {
    manager: ModelManager,
    current: Mutex<Option<LoadedModel>>,
    vision: VisionSettings,
    decode_defaults: DecodeParameters,
    available_models: Vec<ModelListing>,
}

#[derive(Clone)]
pub struct GenerationInputs {
    pub kind: ModelKind,
    pub model: SharedModel,
    pub tokenizer: Arc<Tokenizer>,
    pub vision: VisionSettings,
    pub defaults: DecodeParameters,
}

impl AppState {
    pub fn bootstrap(
        fs: LocalFileSystem,
        config: Arc<AppConfig>,
        device: Device,
        dtype: DType,
        vision: VisionSettings,
        decode_defaults: DecodeParameters,
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
            vision,
            decode_defaults,
            available_models,
        })
    }

    pub fn available_models(&self) -> &[ModelListing] {
        &self.available_models
    }

    pub fn default_max_new_tokens(&self) -> usize {
        self.decode_defaults.max_new_tokens
    }

    pub fn prepare_generation(
        &self,
        requested_model: &str,
    ) -> Result<(GenerationInputs, String), ApiError> {
        self.validate_model(requested_model)?;
        let (shared_model, tokenizer, model_id, kind) =
            self.ensure_model_loaded(requested_model)?;
        let inputs = GenerationInputs {
            kind,
            model: shared_model,
            tokenizer,
            vision: self.vision,
            defaults: self.decode_defaults.clone(),
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
    ) -> Result<(SharedModel, Arc<Tokenizer>, String, ModelKind), ApiError> {
        {
            if let Ok(guard) = self.current.lock() {
                if let Some(loaded) = guard.as_ref() {
                    if loaded.id == model_id {
                        return Ok((
                            Arc::clone(&loaded.model),
                            Arc::clone(&loaded.tokenizer),
                            loaded.id.clone(),
                            loaded.kind,
                        ));
                    }
                }
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
            Arc::clone(&loaded.tokenizer),
            loaded.id.clone(),
            loaded.kind,
        ))
    }
}

struct LoadedModel {
    id: String,
    kind: ModelKind,
    model: SharedModel,
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
        let config_path = ensure_config_file(&self.fs, &resources.id, &resources.config)?;
        let tokenizer_path = ensure_tokenizer_file(&self.fs, &resources.id, &resources.tokenizer)?;
        let weights_path = prepare_weights_path(&self.fs, &resources.id, &resources.weights)?;
        let snapshot_path =
            prepare_snapshot_path(&self.fs, &resources.id, resources.snapshot.as_ref())?;

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
        Ok(LoadedModel {
            id: model_id.to_string(),
            kind: resources.kind,
            model: Arc::new(Mutex::new(model)),
            tokenizer,
        })
    }
}
