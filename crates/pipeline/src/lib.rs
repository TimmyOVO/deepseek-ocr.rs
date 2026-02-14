use std::path::Path;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use deepseek_ocr_config::{ModelResources, PreparedModelPaths, VirtualFileSystem, prepare_model_paths};
use deepseek_ocr_core::{ModelKind, ModelLoadArgs, OcrEngine, ocr_inference_engine::OcrInferenceEngine};
use tokenizers::Tokenizer;

pub struct LoadedOcrModel {
    pub resources: ModelResources,
    pub prepared: PreparedModelPaths,
    pub model: Box<dyn OcrEngine>,
    pub tokenizer: Tokenizer,
    pub engine: OcrInferenceEngine,
}

pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|err| anyhow!("failed to load tokenizer from {}: {err}", path.display()))
}

pub fn load_engine(kind: ModelKind) -> OcrInferenceEngine {
    OcrInferenceEngine::with_default_semantics(kind)
}

pub fn load_model(args: ModelLoadArgs<'_>) -> Result<Box<dyn OcrEngine>> {
    match args.kind {
        ModelKind::Deepseek => deepseek_ocr_infer_deepseek::load_model(args),
        ModelKind::PaddleOcrVl => deepseek_ocr_infer_paddleocr::load_model(args),
        ModelKind::DotsOcr => deepseek_ocr_infer_dots::load_model(args),
        ModelKind::GlmOcr => deepseek_ocr_infer_glm::load_model(args),
    }
}

pub fn load_model_with_resources(
    fs: &impl VirtualFileSystem,
    resources: ModelResources,
    device: Device,
    dtype: DType,
) -> Result<LoadedOcrModel> {
    let prepared = prepare_model_paths(
        fs,
        &resources.id,
        &resources.config,
        &resources.tokenizer,
        &resources.weights,
        resources.snapshot.as_ref(),
    )?;

    let load_args = ModelLoadArgs {
        kind: resources.kind,
        config_path: Some(&prepared.config),
        weights_path: Some(&prepared.weights),
        snapshot_path: prepared.snapshot.as_deref(),
        device,
        dtype,
    };

    let model = load_model(load_args)?;
    let tokenizer = load_tokenizer(&prepared.tokenizer)?;
    let engine = load_engine(resources.kind);

    Ok(LoadedOcrModel {
        resources,
        prepared,
        model,
        tokenizer,
        engine,
    })
}
