use std::{fmt, path::PathBuf};

use anyhow::{Context, Result};

use crate::{DecodeParametersPatch, DeviceKind, OcrFsOptions, OcrModelId, Precision};

/// Pipeline-level configuration schema.
///
/// 当前阶段先通过 `type` 复用 `deepseek-ocr-config` 的配置结构，避免在 pipeline
/// crate 中复制一份 schema 造成漂移。
///
/// 后续当 pipeline API 完全稳定、需要对外发布独立 SDK 时，再考虑引入
/// pipeline 自己的 `OcrConfig` 定义与适配层。
pub type OcrConfig = deepseek_ocr_config::AppConfig;

/// Optional configuration patch used by individual layers.
///
/// 注意：这不是“覆盖流程”，只是“某一层提供的可选字段集”。
#[derive(Debug, Clone, Default)]
pub struct OcrConfigPatch {
    /// Optional path to a configuration file.
    pub config_path: Option<PathBuf>,

    /// File-system roots.
    pub fs: Option<OcrFsOptions>,

    pub model: OcrModelPatch,
    pub inference: OcrInferencePatch,
    pub server: OcrServerPatch,
}

impl OcrConfigPatch {
    pub fn merge_from(&mut self, rhs: OcrConfigPatch) {
        if rhs.config_path.is_some() {
            self.config_path = rhs.config_path;
        }

        if rhs.fs.is_some() {
            self.fs = rhs.fs;
        }

        self.model.merge_from(rhs.model);
        self.inference.merge_from(rhs.inference);
        self.server.merge_from(rhs.server);
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrModelPatch {
    pub id: Option<OcrModelId>,

    pub config: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub weights: Option<PathBuf>,
    pub snapshot: Option<PathBuf>,
}

impl OcrModelPatch {
    pub fn merge_from(&mut self, rhs: OcrModelPatch) {
        if rhs.id.is_some() {
            self.id = rhs.id;
        }
        if rhs.config.is_some() {
            self.config = rhs.config;
        }
        if rhs.tokenizer.is_some() {
            self.tokenizer = rhs.tokenizer;
        }
        if rhs.weights.is_some() {
            self.weights = rhs.weights;
        }
        if rhs.snapshot.is_some() {
            self.snapshot = rhs.snapshot;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrInferencePatch {
    pub device: Option<DeviceKind>,
    pub precision: Option<Precision>,

    pub template: Option<String>,

    pub vision: OcrVisionPatch,

    /// Decoding overrides (patch semantics).
    pub decode: DecodeParametersPatch,
}

impl OcrInferencePatch {
    pub fn merge_from(&mut self, rhs: OcrInferencePatch) {
        if rhs.device.is_some() {
            self.device = rhs.device;
        }
        if rhs.precision.is_some() {
            self.precision = rhs.precision;
        }
        if rhs.template.is_some() {
            self.template = rhs.template;
        }

        self.vision.merge_from(rhs.vision);
        merge_decode_patch(&mut self.decode, rhs.decode);
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrVisionPatch {
    pub base_size: Option<u32>,
    pub image_size: Option<u32>,
    pub crop_mode: Option<bool>,
}

impl OcrVisionPatch {
    pub fn merge_from(&mut self, rhs: OcrVisionPatch) {
        if rhs.base_size.is_some() {
            self.base_size = rhs.base_size;
        }
        if rhs.image_size.is_some() {
            self.image_size = rhs.image_size;
        }
        if rhs.crop_mode.is_some() {
            self.crop_mode = rhs.crop_mode;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrServerPatch {
    pub host: Option<String>,
    pub port: Option<u16>,
}

impl OcrServerPatch {
    pub fn merge_from(&mut self, rhs: OcrServerPatch) {
        if rhs.host.is_some() {
            self.host = rhs.host;
        }
        if rhs.port.is_some() {
            self.port = rhs.port;
        }
    }
}

fn merge_decode_patch(target: &mut DecodeParametersPatch, rhs: DecodeParametersPatch) {
    if rhs.max_new_tokens.is_some() {
        target.max_new_tokens = rhs.max_new_tokens;
    }
    if rhs.do_sample.is_some() {
        target.do_sample = rhs.do_sample;
    }
    if rhs.temperature.is_some() {
        target.temperature = rhs.temperature;
    }
    if rhs.top_p.is_some() {
        target.top_p = rhs.top_p;
    }
    if rhs.top_k.is_some() {
        target.top_k = rhs.top_k;
    }
    if rhs.repetition_penalty.is_some() {
        target.repetition_penalty = rhs.repetition_penalty;
    }
    if rhs.no_repeat_ngram_size.is_some() {
        target.no_repeat_ngram_size = rhs.no_repeat_ngram_size;
    }
    if rhs.seed.is_some() {
        target.seed = rhs.seed;
    }
    if rhs.use_cache.is_some() {
        target.use_cache = rhs.use_cache;
    }
}

/// Origin of a configuration layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OcrConfigSource {
    Defaults,
    ConfigFile,
    CliArgs,
    Request,
}

impl fmt::Display for OcrConfigSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            OcrConfigSource::Defaults => "defaults",
            OcrConfigSource::ConfigFile => "config file",
            OcrConfigSource::CliArgs => "cli args",
            OcrConfigSource::Request => "request",
        };
        f.write_str(label)
    }
}

/// Configuration layer that can contribute a patch.
///
/// 覆盖优先级由 resolver 决定（而不是 layer 自己写死）。
pub trait OcrConfigLayer: Send + Sync {
    fn source(&self) -> OcrConfigSource;
    fn load_patch(&self) -> Result<OcrConfigPatch>;
}

/// Resolver for `OcrConfig`.
///
/// 覆盖语义：按 layer 添加顺序从低优先级到高优先级叠加。
#[derive(Default)]
pub struct OcrConfigResolver {
    layers: Vec<Box<dyn OcrConfigLayer>>,
}

impl OcrConfigResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_layer<L>(&mut self, layer: L)
    where
        L: OcrConfigLayer + 'static,
    {
        self.layers.push(Box::new(layer));
    }

    pub fn merged_patch(&self) -> Result<OcrConfigPatch> {
        let mut merged = OcrConfigPatch::default();
        for layer in &self.layers {
            let source = layer.source();
            let patch = layer
                .load_patch()
                .with_context(|| format!("failed to load config patch from {source}"))?;
            merged.merge_from(patch);
        }
        Ok(merged)
    }

    pub fn resolve(&self) -> Result<OcrConfig> {
        use crate::ext::IntoConfigOverrides;

        let merged = self.merged_patch()?;
        let overrides = merged
            .clone()
            .into_config_overrides()
            .context("failed to build config overrides from merged patch")?;
        let fs = build_local_fs(merged.fs);

        let (mut config, _descriptor) =
            deepseek_ocr_config::AppConfig::load_or_init(&fs, overrides.config_path.as_deref())
                .with_context(|| {
                    let path = overrides
                        .config_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "virtual config".to_string());
                    format!("failed to load or init config from {path}")
                })?;
        config.apply_overrides(&overrides);
        config
            .normalise(&fs)
            .context("failed to normalise config after overrides")?;
        Ok(config)
    }
}

pub(crate) fn build_local_fs(fs: Option<OcrFsOptions>) -> deepseek_ocr_config::LocalFileSystem {
    let opts = fs.unwrap_or_default();

    match (opts.config_dir.clone(), opts.cache_dir.clone()) {
        (Some(config_dir), Some(cache_dir)) => {
            deepseek_ocr_config::LocalFileSystem::with_directories(
                opts.app_name,
                config_dir,
                cache_dir,
            )
        }
        _ => deepseek_ocr_config::LocalFileSystem::new(opts.app_name),
    }
}

/// Convenience layer that wraps an already-built patch.
#[derive(Debug, Clone)]
pub struct OcrPatchLayer {
    source: OcrConfigSource,
    patch: OcrConfigPatch,
}

impl OcrPatchLayer {
    pub fn new(source: OcrConfigSource, patch: OcrConfigPatch) -> Self {
        Self { source, patch }
    }
}

impl OcrConfigLayer for OcrPatchLayer {
    fn source(&self) -> OcrConfigSource {
        self.source
    }

    fn load_patch(&self) -> Result<OcrConfigPatch> {
        Ok(self.patch.clone())
    }
}
