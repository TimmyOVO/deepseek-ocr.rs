use std::{
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use tokenizers::Tokenizer;
use deepseek_ocr_pipeline::{
    DecodeParameters, DecodeParametersPatch, ModelKind, OcrConfig, OcrConfigPatch,
    OcrConfigResolver, OcrConfigSource, OcrInferencePatch, OcrModelId, OcrPatchLayer,
    OcrPipelineHandle, OcrRuntime, OcrRuntimeBuilder, VisionSettings,
};

use crate::args::Args;
use crate::error::ApiError;

pub type SharedModel = Arc<Mutex<Box<dyn deepseek_ocr_pipeline::deepseek_ocr_core::OcrEngine>>>;

type LoadedModelHandles = (
    OcrPipelineHandle,
    SharedModel,
    Arc<deepseek_ocr_pipeline::deepseek_ocr_core::OcrInferenceEngine>,
    Arc<Tokenizer>,
    String,
);

#[derive(Clone)]
pub struct ModelListing {
    pub id: String,
    pub kind: ModelKind,
}

pub struct AppState {
    runtime: OcrRuntime,
    current: Mutex<Option<LoadedModel>>,
    base_config: Arc<OcrConfig>,
    runtime_config: Arc<OcrConfig>,
    runtime_inference_patch: OcrInferencePatch,
    available_models: Vec<ModelListing>,
}

#[derive(Clone)]
pub struct GenerationInputs {
    pub handle: OcrPipelineHandle,
    pub model: SharedModel,
    pub engine: Arc<deepseek_ocr_pipeline::deepseek_ocr_core::OcrInferenceEngine>,
    pub tokenizer: Arc<Tokenizer>,
    pub template: String,
    pub vision: VisionSettings,
    pub defaults: DecodeParameters,
}

impl AppState {
    pub fn bootstrap(args: &Args, defaults_layer: Option<OcrConfigPatch>) -> Result<Self> {
        let config_file_layer = OcrConfigPatch {
            config_path: args.model.config.clone(),
            ..Default::default()
        };
        let cli_args_layer = cli_patch_from_args(args)?;

        let mut base_resolver = OcrConfigResolver::new();
        if let Some(defaults) = defaults_layer.clone() {
            base_resolver.push_layer(OcrPatchLayer::new(OcrConfigSource::Defaults, defaults));
        }
        base_resolver.push_layer(OcrPatchLayer::new(
            OcrConfigSource::ConfigFile,
            config_file_layer.clone(),
        ));
        let base_config = Arc::new(base_resolver.resolve()?);

        let mut runtime_resolver = OcrConfigResolver::new();
        if let Some(defaults) = defaults_layer.clone() {
            runtime_resolver.push_layer(OcrPatchLayer::new(OcrConfigSource::Defaults, defaults));
        }
        runtime_resolver.push_layer(OcrPatchLayer::new(
            OcrConfigSource::ConfigFile,
            config_file_layer.clone(),
        ));
        runtime_resolver.push_layer(OcrPatchLayer::new(
            OcrConfigSource::CliArgs,
            cli_args_layer.clone(),
        ));
        let runtime_config = Arc::new(runtime_resolver.resolve()?);

        let mut runtime_builder = OcrRuntimeBuilder::new();
        if let Some(defaults) = defaults_layer {
            runtime_builder = runtime_builder.with_defaults_layer(defaults);
        }
        runtime_builder = runtime_builder
            .with_config_file_layer(config_file_layer)
            .with_cli_args_layer(cli_args_layer.clone());
        let runtime = runtime_builder.build()?;

        let available_models = runtime_config
            .models
            .entries
            .iter()
            .map(|(id, entry)| ModelListing {
                id: id.clone(),
                kind: entry.kind,
            })
            .collect::<Vec<_>>();

        Ok(Self {
            runtime,
            current: Mutex::new(None),
            base_config,
            runtime_config,
            runtime_inference_patch: cli_args_layer.inference,
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
        let entry = self
            .runtime_config
            .models
            .entries
            .get(model_id)
            .ok_or_else(|| {
                ApiError::BadRequest(format!("requested model `{model_id}` is not available"))
            })?;

        let mut effective = self.base_config.inference.clone();
        effective += &entry.defaults.inference;

        if let Some(template) = self.runtime_inference_patch.template.as_ref() {
            effective.template = template.clone();
        }
        if let Some(base_size) = self.runtime_inference_patch.vision.base_size {
            effective.base_size = base_size;
        }
        if let Some(image_size) = self.runtime_inference_patch.vision.image_size {
            effective.image_size = image_size;
        }
        if let Some(crop_mode) = self.runtime_inference_patch.vision.crop_mode {
            effective.crop_mode = crop_mode;
        }
        effective.decode = effective.decode.clone()
            + &DecodeParametersPatch {
                max_new_tokens: self.runtime_inference_patch.decode.max_new_tokens,
                do_sample: self.runtime_inference_patch.decode.do_sample,
                temperature: self.runtime_inference_patch.decode.temperature,
                top_p: self.runtime_inference_patch.decode.top_p,
                top_k: self.runtime_inference_patch.decode.top_k,
                repetition_penalty: self.runtime_inference_patch.decode.repetition_penalty,
                no_repeat_ngram_size: self.runtime_inference_patch.decode.no_repeat_ngram_size,
                seed: self.runtime_inference_patch.decode.seed,
                use_cache: self.runtime_inference_patch.decode.use_cache,
            };

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
        let (handle, model, engine, tokenizer, model_id) =
            self.ensure_model_loaded(requested_model)?;
        let (vision, defaults, template) = self.per_model_inference_settings(requested_model)?;
        let inputs = GenerationInputs {
            handle,
            model,
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
                    loaded.handle.clone(),
                    Arc::clone(&loaded.model),
                    Arc::clone(&loaded.engine),
                    Arc::clone(&loaded.tokenizer),
                    loaded.id.clone(),
                ));
            }
        }

        let typed_model_id =
            OcrModelId::try_from(model_id).map_err(|err| ApiError::BadRequest(err.to_string()))?;

        let loaded = self
            .runtime
            .manager()
            .load(&typed_model_id)
            .map_err(|err| ApiError::Internal(err.to_string()))?;

        let tokenizer = Arc::new(
            loaded
                .pipeline()
                .tokenizer()
                .map_err(|err| ApiError::Internal(err.to_string()))?
                .clone(),
        );
        let kind = self
            .available_models
            .iter()
            .find(|entry| entry.id == model_id)
            .map(|entry| entry.kind)
            .ok_or_else(|| {
                ApiError::BadRequest(format!("requested model `{model_id}` is not available"))
            })?;
        let (model, engine) = compatibility_handles(kind);

        let mut guard = self.current.lock().expect("model mutex poisoning detected");
        *guard = Some(LoadedModel {
            id: model_id.to_string(),
            handle: loaded.clone(),
            model: Arc::clone(&model),
            engine: Arc::clone(&engine),
            tokenizer: Arc::clone(&tokenizer),
        });
        let loaded = guard
            .as_ref()
            .expect("loaded model missing after assignment");
        Ok((
            loaded.handle.clone(),
            Arc::clone(&loaded.model),
            Arc::clone(&loaded.engine),
            Arc::clone(&loaded.tokenizer),
            loaded.id.clone(),
        ))
    }
}

struct LoadedModel {
    id: String,
    handle: OcrPipelineHandle,
    model: SharedModel,
    engine: Arc<deepseek_ocr_pipeline::deepseek_ocr_core::OcrInferenceEngine>,
    tokenizer: Arc<Tokenizer>,
}

#[derive(Debug)]
struct PipelineHandleEngine {
    kind: ModelKind,
    device: Device,
    dtype: DType,
}

impl deepseek_ocr_pipeline::deepseek_ocr_core::OcrEngine for PipelineHandleEngine {
    fn kind(&self) -> ModelKind {
        self.kind
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn decode(
        &self,
        _tokenizer: &Tokenizer,
        _prompt: &str,
        _images: &[image::DynamicImage],
        _vision: VisionSettings,
        _params: &DecodeParameters,
        _stream: deepseek_ocr_pipeline::deepseek_ocr_core::inference::StreamCallback,
    ) -> Result<deepseek_ocr_pipeline::deepseek_ocr_core::DecodeOutcome> {
        anyhow::bail!(
            "legacy engine path is unavailable after server state migration; use OcrPipelineHandle"
        )
    }
}

fn compatibility_handles(
    kind: ModelKind,
) -> (
    SharedModel,
    Arc<deepseek_ocr_pipeline::deepseek_ocr_core::OcrInferenceEngine>,
) {
    let model: SharedModel = Arc::new(Mutex::new(Box::new(PipelineHandleEngine {
        kind,
        device: Device::Cpu,
        dtype: DType::F32,
    })));
    let engine = Arc::new(
        deepseek_ocr_pipeline::deepseek_ocr_core::OcrInferenceEngine::with_default_semantics(kind),
    );
    (model, engine)
}

fn cli_patch_from_args(args: &Args) -> Result<OcrConfigPatch> {
    let model_id = match args.model.model.as_deref() {
        Some(raw) => Some(
            OcrModelId::try_from(raw)
                .map_err(|err| anyhow!("invalid --model `{raw}`: {err}"))?,
        ),
        None => None,
    };

    Ok(OcrConfigPatch {
        model: deepseek_ocr_pipeline::OcrModelPatch {
            id: model_id,
            config: args.model.model_config.clone(),
            tokenizer: args.model.tokenizer.clone(),
            weights: args.model.weights.clone(),
            snapshot: None,
        },
        inference: OcrInferencePatch {
            device: args.inference.device,
            precision: args.inference.dtype,
            template: args.inference.template.clone(),
            vision: deepseek_ocr_pipeline::OcrVisionPatch {
                base_size: args.inference.base_size,
                image_size: args.inference.image_size,
                crop_mode: args.inference.crop_mode,
            },
            decode: DecodeParametersPatch {
                max_new_tokens: args.inference.max_new_tokens,
                do_sample: args.inference.do_sample,
                temperature: args.inference.temperature,
                top_p: args.inference.top_p,
                top_k: args.inference.top_k,
                repetition_penalty: args.inference.repetition_penalty,
                no_repeat_ngram_size: args.inference.no_repeat_ngram_size,
                seed: args.inference.seed,
                use_cache: args.inference.no_cache.then_some(false),
            },
        },
        server: deepseek_ocr_pipeline::OcrServerPatch {
            host: args.bind.host.clone(),
            port: args.bind.port,
        },
        ..Default::default()
    })
}
