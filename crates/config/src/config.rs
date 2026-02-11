use std::{
    collections::BTreeMap,
    fs,
    ops::AddAssign,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use deepseek_ocr_core::{
    ModelKind,
    runtime::{DeviceKind, Precision},
};
use serde::{Deserialize, Serialize};

use crate::fs::{VirtualFileSystem, VirtualPath};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct AppConfig {
    pub models: ModelRegistry,
    pub inference: InferenceSettings,
    pub server: ServerSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelRegistry {
    pub active: String,
    pub entries: BTreeMap<String, ModelEntry>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let mut entries = BTreeMap::new();
        ensure_default_model_entries(&mut entries);
        Self {
            active: "deepseek-ocr".to_string(),
            entries,
        }
    }
}

fn ensure_default_model_entries(entries: &mut BTreeMap<String, ModelEntry>) {
    entries
        .entry("deepseek-ocr".to_string())
        .or_insert_with(deepseek_ocr1_entry);
    entries
        .entry("deepseek-ocr-2".to_string())
        .or_insert_with(deepseek_ocr2_entry);
    entries
        .entry("paddleocr-vl".to_string())
        .or_insert_with(|| ModelEntry {
            kind: ModelKind::PaddleOcrVl,
            ..ModelEntry::default()
        });
    entries
        .entry("dots-ocr".to_string())
        .or_insert_with(|| ModelEntry {
            kind: ModelKind::DotsOcr,
            ..ModelEntry::default()
        });
    entries
        .entry("glm-ocr".to_string())
        .or_insert_with(glm_ocr_entry);
    entries
        .entry("deepseek-ocr-q4k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::Deepseek, "Q4_K", "deepseek-ocr"));
    entries
        .entry("deepseek-ocr-q6k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::Deepseek, "Q6_K", "deepseek-ocr"));
    entries
        .entry("deepseek-ocr-q8k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::Deepseek, "Q8_0", "deepseek-ocr"));
    entries
        .entry("paddleocr-vl-q4k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::PaddleOcrVl, "Q4_K", "paddleocr-vl"));
    entries
        .entry("paddleocr-vl-q6k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::PaddleOcrVl, "Q6_K", "paddleocr-vl"));
    entries
        .entry("paddleocr-vl-q8k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::PaddleOcrVl, "Q8_0", "paddleocr-vl"));
    entries
        .entry("dots-ocr-q4k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::DotsOcr, "Q4_K", "dots-ocr"));
    entries
        .entry("dots-ocr-q6k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::DotsOcr, "Q6_K", "dots-ocr"));
    entries
        .entry("dots-ocr-q8k".to_string())
        .or_insert_with(|| quantized_entry(ModelKind::DotsOcr, "Q8_0", "dots-ocr"));

    ensure_model_defaults(entries);
}

fn deepseek_ocr1_entry() -> ModelEntry {
    let mut entry = ModelEntry::default();
    entry.defaults.inference.base_size = Some(1024);
    entry.defaults.inference.image_size = Some(640);
    entry.defaults.inference.crop_mode = Some(true);
    entry
}

fn deepseek_ocr2_entry() -> ModelEntry {
    let mut entry = ModelEntry::default();
    entry.defaults.inference.base_size = Some(1024);
    entry.defaults.inference.image_size = Some(768);
    entry.defaults.inference.crop_mode = Some(true);
    entry
}

fn glm_ocr_entry() -> ModelEntry {
    let mut entry = ModelEntry {
        kind: ModelKind::GlmOcr,
        ..ModelEntry::default()
    };
    entry.defaults.inference.template = Some("plain".to_string());
    entry.defaults.inference.base_size = Some(336);
    entry.defaults.inference.image_size = Some(336);
    entry.defaults.inference.crop_mode = Some(false);
    entry.defaults.inference.max_new_tokens = Some(8192);
    entry.defaults.inference.do_sample = Some(false);
    entry.defaults.inference.temperature = Some(0.0);
    entry.defaults.inference.top_p = Some(1.0);
    entry.defaults.inference.repetition_penalty = Some(1.0);
    entry.defaults.inference.no_repeat_ngram_size = None;
    entry.defaults.inference.seed = Some(0);
    entry
}

fn ensure_model_defaults(entries: &mut BTreeMap<String, ModelEntry>) {
    let ocr1_defaults = deepseek_ocr1_entry().defaults;
    let ocr2_defaults = deepseek_ocr2_entry().defaults;

    if let Some(entry) = entries.get_mut("deepseek-ocr") {
        fill_missing_model_defaults(entry, &ocr1_defaults);
    }
    if let Some(entry) = entries.get_mut("deepseek-ocr-2") {
        fill_missing_model_defaults(entry, &ocr2_defaults);
    }

    let quantized_deepseek_ids = ["deepseek-ocr-q4k", "deepseek-ocr-q6k", "deepseek-ocr-q8k"];
    for model_id in quantized_deepseek_ids {
        if let Some(entry) = entries.get_mut(model_id) {
            fill_missing_model_defaults(entry, &ocr1_defaults);
        }
    }
}

fn fill_missing_model_defaults(entry: &mut ModelEntry, defaults: &ModelDefaults) {
    fill_missing_inference_override(&mut entry.defaults.inference, &defaults.inference);
}

fn fill_missing_inference_override(target: &mut InferenceOverride, defaults: &InferenceOverride) {
    if target.base_size.is_none() {
        target.base_size = defaults.base_size;
    }
    if target.image_size.is_none() {
        target.image_size = defaults.image_size;
    }
    if target.crop_mode.is_none() {
        target.crop_mode = defaults.crop_mode;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelEntry {
    pub kind: ModelKind,
    pub config: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub weights: Option<PathBuf>,
    pub snapshot: Option<SnapshotEntry>,
    pub defaults: ModelDefaults,
}

impl Default for ModelEntry {
    fn default() -> Self {
        Self {
            kind: ModelKind::Deepseek,
            config: None,
            tokenizer: None,
            weights: None,
            snapshot: None,
            defaults: ModelDefaults::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SnapshotEntry {
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceSettings {
    pub device: DeviceKind,
    pub precision: Option<Precision>,
    pub template: String,
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: bool,
    pub max_new_tokens: usize,
    pub use_cache: bool,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: Option<usize>,
    pub seed: Option<u64>,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            device: DeviceKind::Cpu,
            precision: None,
            template: "plain".to_string(),
            base_size: 1024,
            image_size: 640,
            crop_mode: true,
            max_new_tokens: 512,
            use_cache: true,
            do_sample: false,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: Some(20),
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ResourceLocation {
    Virtual(VirtualPath),
    Physical(PathBuf),
}

impl ResourceLocation {
    pub fn display_with(&self, fs: &impl VirtualFileSystem) -> Result<String> {
        match self {
            ResourceLocation::Virtual(path) => {
                fs.with_physical_path(path, |p| Ok(p.display().to_string()))
            }
            ResourceLocation::Physical(path) => Ok(path.display().to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelResources {
    pub id: String,
    pub config: ResourceLocation,
    pub tokenizer: ResourceLocation,
    pub weights: ResourceLocation,
    pub kind: ModelKind,
    pub snapshot: Option<SnapshotResources>,
}

#[derive(Debug, Clone)]
pub struct SnapshotResources {
    pub location: ResourceLocation,
    pub dtype: String,
}

pub struct ConfigDescriptor {
    pub location: ResourceLocation,
}

impl AppConfig {
    pub fn load_or_init(
        fs: &impl VirtualFileSystem,
        override_path: Option<&Path>,
    ) -> Result<(Self, ConfigDescriptor)> {
        match override_path {
            Some(path) => load_physical_config(fs, path),
            None => load_virtual_config(fs),
        }
    }

    pub fn load_with_overrides(
        fs: &impl VirtualFileSystem,
        overrides: ConfigOverrides,
    ) -> Result<(Self, ConfigDescriptor, ModelResources)> {
        let config_path_override = overrides.config_path.clone();
        let (mut config, descriptor) = Self::load_or_init(fs, config_path_override.as_deref())?;
        config += overrides;
        config.normalise(fs)?;
        let resources = config.active_model_resources(fs)?;
        Ok((config, descriptor, resources))
    }

    pub fn normalise(&mut self, fs: &impl VirtualFileSystem) -> Result<()> {
        ensure_default_model_entries(&mut self.models.entries);
        if !self.models.entries.contains_key(&self.models.active) {
            self.models
                .entries
                .insert(self.models.active.clone(), ModelEntry::default());
        }

        for (model_id, entry) in self.models.entries.iter_mut() {
            entry.normalise(fs, model_id)?;
        }
        Ok(())
    }

    pub fn active_model_resources(&self, fs: &impl VirtualFileSystem) -> Result<ModelResources> {
        self.model_resources(fs, &self.models.active)
    }

    pub fn model_resources(
        &self,
        _fs: &impl VirtualFileSystem,
        model_id: &str,
    ) -> Result<ModelResources> {
        let entry = self
            .models
            .entries
            .get(model_id)
            .ok_or_else(|| anyhow!("model `{model_id}` not found in configuration"))?;
        Ok(entry.resolved(model_id))
    }

    pub fn apply_overrides(&mut self, overrides: &ConfigOverrides) {
        if let Some(model_id) = overrides.model_id.as_ref() {
            self.models.active = model_id.clone();
            self.models.entries.entry(model_id.clone()).or_default();
        }

        if let Some(entry) = self.models.entries.get_mut(&self.models.active) {
            if let Some(path) = overrides.model_config.as_ref() {
                entry.config = Some(path.clone());
            }
            if let Some(path) = overrides.tokenizer.as_ref() {
                entry.tokenizer = Some(path.clone());
            }
            if let Some(path) = overrides.weights.as_ref() {
                entry.weights = Some(path.clone());
            }

            // Per-model inference defaults apply before CLI/runtime overrides.
            entry.defaults.inference.apply_to(&mut self.inference);
        }

        overrides.inference.apply_to(&mut self.inference);

        if let Some(host) = overrides.server.host.as_ref() {
            self.server.host = host.clone();
        }
        if let Some(port) = overrides.server.port {
            self.server.port = port;
        }
    }
}

impl ModelEntry {
    fn normalise(&mut self, fs: &impl VirtualFileSystem, model_id: &str) -> Result<()> {
        let model_dir = VirtualPath::model_dir(model_id.to_string());
        fs.ensure_dir(&model_dir)?;
        fs.ensure_parent(&VirtualPath::model_config(model_id.to_string()))?;
        fs.ensure_parent(&VirtualPath::model_tokenizer(model_id.to_string()))?;
        fs.ensure_parent(&VirtualPath::model_weights(model_id.to_string()))?;
        if self.snapshot.is_some() {
            fs.ensure_parent(&VirtualPath::model_snapshot(model_id.to_string()))?;
        }
        Ok(())
    }

    fn resolved(&self, model_id: &str) -> ModelResources {
        let config = match &self.config {
            Some(path) => ResourceLocation::Physical(path.clone()),
            None => ResourceLocation::Virtual(VirtualPath::model_config(model_id.to_string())),
        };
        let tokenizer = match &self.tokenizer {
            Some(path) => ResourceLocation::Physical(path.clone()),
            None => ResourceLocation::Virtual(VirtualPath::model_tokenizer(model_id.to_string())),
        };
        let weights = match &self.weights {
            Some(path) => ResourceLocation::Physical(path.clone()),
            None => ResourceLocation::Virtual(VirtualPath::model_weights(model_id.to_string())),
        };
        let snapshot = self
            .snapshot
            .as_ref()
            .map(|entry| entry.to_resources(model_id));
        ModelResources {
            id: model_id.to_string(),
            config,
            tokenizer,
            weights,
            kind: self.kind,
            snapshot,
        }
    }
}

impl SnapshotEntry {
    fn to_resources(&self, model_id: &str) -> SnapshotResources {
        SnapshotResources {
            location: ResourceLocation::Virtual(VirtualPath::model_snapshot(model_id.to_string())),
            dtype: self.dtype.clone(),
        }
    }
}

fn quantized_entry(kind: ModelKind, dtype: &str, baseline_id: &str) -> ModelEntry {
    let defaults = match baseline_id {
        "deepseek-ocr" => deepseek_ocr1_entry().defaults,
        "deepseek-ocr-2" => deepseek_ocr2_entry().defaults,
        _ => ModelDefaults::default(),
    };
    ModelEntry {
        kind,
        snapshot: Some(SnapshotEntry {
            dtype: dtype.to_string(),
        }),
        defaults,
        ..ModelEntry::default()
    }
}

fn load_virtual_config(fs: &impl VirtualFileSystem) -> Result<(AppConfig, ConfigDescriptor)> {
    let path = VirtualPath::config_file();
    if !fs.exists(&path)? {
        let mut cfg = AppConfig::default();
        cfg.normalise(fs)?;
        let serialized = toml::to_string_pretty(&cfg)?;
        fs.write(&path, serialized.as_bytes())?;
        return Ok((
            cfg,
            ConfigDescriptor {
                location: ResourceLocation::Virtual(path),
            },
        ));
    }

    let bytes = fs.read(&path)?;
    let contents = String::from_utf8(bytes).context("configuration file is not valid UTF-8")?;
    let mut cfg: AppConfig =
        toml::from_str(&contents).context("failed to parse configuration file")?;
    cfg.normalise(fs)?;
    Ok((
        cfg,
        ConfigDescriptor {
            location: ResourceLocation::Virtual(path),
        },
    ))
}

fn load_physical_config(
    fs: &impl VirtualFileSystem,
    path: &Path,
) -> Result<(AppConfig, ConfigDescriptor)> {
    let path_buf = path.to_path_buf();
    if !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        let mut cfg = AppConfig::default();
        cfg.normalise(fs)?;
        let serialized = toml::to_string_pretty(&cfg)?;
        fs::write(&path_buf, serialized)
            .with_context(|| format!("failed to write configuration to {}", path_buf.display()))?;
        return Ok((
            cfg,
            ConfigDescriptor {
                location: ResourceLocation::Physical(path_buf),
            },
        ));
    }

    let contents = fs::read_to_string(&path_buf)
        .with_context(|| format!("failed to read configuration from {}", path_buf.display()))?;
    let mut cfg: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("failed to parse configuration at {}", path_buf.display()))?;
    cfg.normalise(fs)?;
    Ok((
        cfg,
        ConfigDescriptor {
            location: ResourceLocation::Physical(path_buf),
        },
    ))
}

#[derive(Debug, Default, Clone)]
pub struct ConfigOverrides {
    pub config_path: Option<PathBuf>,
    pub model_id: Option<String>,
    pub model_config: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub weights: Option<PathBuf>,
    pub inference: InferenceOverride,
    pub server: ServerOverride,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceOverride {
    pub device: Option<DeviceKind>,
    pub precision: Option<Precision>,
    pub template: Option<String>,
    pub base_size: Option<u32>,
    pub image_size: Option<u32>,
    pub crop_mode: Option<bool>,
    pub max_new_tokens: Option<usize>,
    pub use_cache: Option<bool>,
    pub do_sample: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub no_repeat_ngram_size: Option<usize>,
    pub seed: Option<u64>,
}

impl InferenceOverride {
    pub fn apply_to(&self, inference: &mut InferenceSettings) {
        if let Some(device) = self.device {
            inference.device = device;
        }
        if self.precision.is_some() {
            inference.precision = self.precision;
        }
        if let Some(template) = self.template.as_ref() {
            inference.template = template.clone();
        }
        if let Some(base_size) = self.base_size {
            inference.base_size = base_size;
        }
        if let Some(image_size) = self.image_size {
            inference.image_size = image_size;
        }
        if let Some(crop_mode) = self.crop_mode {
            inference.crop_mode = crop_mode;
        }
        if let Some(max_new_tokens) = self.max_new_tokens {
            inference.max_new_tokens = max_new_tokens;
        }
        if let Some(use_cache) = self.use_cache {
            inference.use_cache = use_cache;
        }
        if let Some(do_sample) = self.do_sample {
            inference.do_sample = do_sample;
        }
        if let Some(temperature) = self.temperature {
            inference.temperature = temperature;
        }
        if let Some(top_p) = self.top_p {
            inference.top_p = top_p;
        }
        if let Some(top_k) = self.top_k {
            inference.top_k = Some(top_k);
        }
        if let Some(repetition_penalty) = self.repetition_penalty {
            inference.repetition_penalty = repetition_penalty;
        }
        if let Some(no_repeat) = self.no_repeat_ngram_size {
            inference.no_repeat_ngram_size = Some(no_repeat);
        }
        if self.seed.is_some() {
            inference.seed = self.seed;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelDefaults {
    pub inference: InferenceOverride,
}

#[derive(Debug, Default, Clone)]
pub struct ServerOverride {
    pub host: Option<String>,
    pub port: Option<u16>,
}

pub trait ConfigOverride {
    fn apply(self, config: &mut AppConfig);
}

impl ConfigOverride for ConfigOverrides {
    fn apply(self, config: &mut AppConfig) {
        config.apply_overrides(&self);
    }
}

impl ConfigOverride for &ConfigOverrides {
    fn apply(self, config: &mut AppConfig) {
        config.apply_overrides(self);
    }
}

impl<O: ConfigOverride> AddAssign<O> for AppConfig {
    fn add_assign(&mut self, rhs: O) {
        rhs.apply(self);
    }
}

pub fn save_config(
    fs: &impl VirtualFileSystem,
    descriptor: &ConfigDescriptor,
    config: &AppConfig,
) -> Result<()> {
    let serialized = toml::to_string_pretty(config)?;
    match &descriptor.location {
        ResourceLocation::Virtual(path) => fs.write(path, serialized.as_bytes()),
        ResourceLocation::Physical(path) => fs::write(path, serialized)
            .with_context(|| format!("failed to write configuration to {}", path.display())),
    }
}
