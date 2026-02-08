use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use tokenizers::Tokenizer;
use tracing::info;

use deepseek_ocr_assets as assets;
use deepseek_ocr_config::{AppConfig, InferenceOverride, InferenceSettings, LocalFileSystem};
use deepseek_ocr_core::{DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings};
use deepseek_ocr_infer_deepseek::load_model as load_deepseek_model;
use deepseek_ocr_infer_dots::load_model as load_dots_model;
use deepseek_ocr_infer_glm::load_model as load_glm_model;
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
    base_inference: InferenceSettings,
    inference_overrides: InferenceOverride,
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

    fn per_model_inference_settings(
        &self,
        model_id: &str,
    ) -> Result<(VisionSettings, DecodeParameters), ApiError> {
        let base_config = self.manager.config.as_ref();
        let mut effective = self.base_inference.clone();
        let Some(entry) = base_config.models.entries.get(model_id) else {
            return Err(ApiError::BadRequest(format!(
                "requested model `{model_id}` is not available"
            )));
        };

        // Resolve requested model defaults first, then overlay process-level
        // inference overrides captured at bootstrap.
        entry.defaults.inference.apply_to(&mut effective);
        self.inference_overrides.apply_to(&mut effective);

        let model_defaults = DecodeParameters {
            max_new_tokens: effective.max_new_tokens,
            do_sample: effective.do_sample,
            temperature: effective.temperature,
            top_p: if effective.top_p < 1.0 {
                Some(effective.top_p)
            } else {
                None
            },
            top_k: effective.top_k,
            repetition_penalty: effective.repetition_penalty,
            no_repeat_ngram_size: effective.no_repeat_ngram_size,
            seed: effective.seed,
            use_cache: effective.use_cache,
        };

        let decode = model_defaults;

        let vision = VisionSettings {
            base_size: effective.base_size,
            image_size: effective.image_size,
            crop_mode: effective.crop_mode,
        };

        Ok((vision, decode))
    }

    pub fn prepare_generation(
        &self,
        requested_model: &str,
    ) -> Result<(GenerationInputs, String), ApiError> {
        self.validate_model(requested_model)?;
        let (shared_model, tokenizer, model_id, kind) =
            self.ensure_model_loaded(requested_model)?;
        let (vision, defaults) = self.per_model_inference_settings(requested_model)?;
        let inputs = GenerationInputs {
            kind,
            model: shared_model,
            tokenizer,
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
    ) -> Result<(SharedModel, Arc<Tokenizer>, String, ModelKind), ApiError> {
        {
            if let Ok(guard) = self.current.lock()
                && let Some(loaded) = guard.as_ref()
                && loaded.id == model_id
            {
                return Ok((
                    Arc::clone(&loaded.model),
                    Arc::clone(&loaded.tokenizer),
                    loaded.id.clone(),
                    loaded.kind,
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

        // Ensure any model-specific preprocessor config is present (e.g. dots.ocr).
        let _ = assets::ensure_model_preprocessor_for(&resources.id, &config_path)?;

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
        Ok(LoadedModel {
            id: model_id.to_string(),
            kind: resources.kind,
            model: Arc::new(Mutex::new(model)),
            tokenizer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_state(
        base_inference: InferenceSettings,
        inference_overrides: InferenceOverride,
    ) -> AppState {
        let fs = LocalFileSystem::new("deepseek-ocr-server-tests");
        let config = Arc::new(AppConfig::default());
        AppState::bootstrap(
            fs,
            config,
            Device::Cpu,
            DType::F32,
            base_inference,
            inference_overrides,
        )
        .expect("bootstrap state")
    }

    #[test]
    fn ocr2_uses_its_model_default_image_size() {
        let mut base = InferenceSettings::default();
        base.image_size = 640;
        let state = build_state(base, InferenceOverride::default());

        let (vision, _) = state
            .per_model_inference_settings("deepseek-ocr-2")
            .expect("resolve ocr2 settings");
        assert_eq!(vision.base_size, 1024);
        assert_eq!(vision.image_size, 768);
        assert!(vision.crop_mode);
    }

    #[test]
    fn server_cli_overrides_model_defaults() {
        let base = InferenceSettings::default();
        let mut overrides = InferenceOverride::default();
        overrides.image_size = Some(896);
        overrides.base_size = Some(960);
        let state = build_state(base, overrides);

        let (vision, _) = state
            .per_model_inference_settings("deepseek-ocr-2")
            .expect("resolve overridden settings");
        assert_eq!(vision.base_size, 960);
        assert_eq!(vision.image_size, 896);
    }
}
