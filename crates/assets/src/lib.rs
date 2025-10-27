use once_cell::sync::OnceCell;
use std::{
    collections::VecDeque,
    fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use hf_hub::api::sync::Api;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle, style::ProgressTracker};
use reqwest::blocking::Client;
use serde::Deserialize;

pub const DEFAULT_REPO_ID: &str = "deepseek-ai/DeepSeek-OCR";
pub const DEFAULT_CONFIG_PATH: &str = "DeepSeek-OCR/config.json";
pub const DEFAULT_CONFIG_FILENAME: &str = "config.json";
pub const DEFAULT_TOKENIZER_PATH: &str = "DeepSeek-OCR/tokenizer.json";
pub const DEFAULT_TOKENIZER_FILENAME: &str = "tokenizer.json";
pub const DEFAULT_WEIGHTS_PATH: &str = deepseek_ocr_core::model::DEFAULT_WEIGHTS_PATH;
pub const DEFAULT_WEIGHTS_FILENAME: &str = "model-00001-of-000001.safetensors";

const DEFAULT_MODELSCOPE_ID: &str = "deepseek-ai/DeepSeek-OCR";
const MODELSCOPE_FILES_URL: &str =
    "https://modelscope.cn/api/v1/models/{model_id}/repo/files?Recursive=true";
const MODELSCOPE_DOWNLOAD_URL: &str =
    "https://modelscope.cn/models/{model_id}/resolve/master/{path}";
const HTTP_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0";
const PROGRESS_TEMPLATE: &str =
    "{msg} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {bytes_per_sec_smoothed} ({eta})";
const PROGRESS_LABEL_MAX: usize = 30;

static HTTP_CLIENT: OnceCell<Client> = OnceCell::new();
static MODELSCOPE_MANIFEST: OnceCell<Vec<ModelScopeFile>> = OnceCell::new();
static PROVIDER_CHOICE: OnceCell<Provider> = OnceCell::new();

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Provider {
    HuggingFace,
    ModelScope,
}

#[derive(Debug, Deserialize)]
struct ModelScopeResponse {
    #[serde(rename = "Code")]
    code: i32,
    #[serde(rename = "Data")]
    data: ModelScopeData,
}

#[derive(Debug, Deserialize)]
struct ModelScopeData {
    #[serde(rename = "Files")]
    files: Vec<ModelScopeFile>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelScopeFile {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size")]
    size: u64,
    #[serde(rename = "Type")]
    kind: String,
}

#[derive(Clone)]
struct SmoothedRate {
    window: Duration,
    samples: VecDeque<(Instant, u64)>,
}

impl Default for Provider {
    fn default() -> Self {
        Provider::HuggingFace
    }
}

impl Default for SmoothedRate {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(1),
            samples: VecDeque::new(),
        }
    }
}

impl ProgressTracker for SmoothedRate {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(self.clone())
    }

    fn tick(&mut self, state: &indicatif::ProgressState, now: Instant) {
        if let Some((last, _)) = self.samples.back() {
            if now.duration_since(*last) < Duration::from_millis(20) {
                return;
            }
        }

        self.samples.push_back((now, state.pos()));
        while let Some((time, _)) = self.samples.front() {
            if now.duration_since(*time) > self.window {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    fn reset(&mut self, _state: &indicatif::ProgressState, _now: Instant) {
        self.samples.clear();
    }

    fn write(&self, _state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write) {
        if let (Some((t0, p0)), Some((t1, p1))) = (self.samples.front(), self.samples.back()) {
            if self.samples.len() > 1 && t1 > t0 {
                let elapsed = t1.duration_since(*t0).as_millis() as f64 / 1000.0;
                let bytes = (p1 - p0) as f64;
                let rate = if elapsed > 0.0 { bytes / elapsed } else { 0.0 };
                let _ = write!(w, "{}/s", HumanBytes(rate as u64));
                return;
            }
        }

        let _ = write!(w, "-");
    }
}

pub fn ensure_config() -> Result<PathBuf> {
    ensure_config_at(Path::new(DEFAULT_CONFIG_PATH))
}

pub fn ensure_config_at(target: &Path) -> Result<PathBuf> {
    if target.exists() {
        return Ok(target.to_path_buf());
    }

    download_asset(DEFAULT_CONFIG_FILENAME, target)
}

pub fn ensure_tokenizer(path: &Path) -> Result<PathBuf> {
    ensure_tokenizer_at(path)
}

pub fn ensure_tokenizer_at(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    if path != Path::new(DEFAULT_TOKENIZER_PATH) {
        ensure_parent(path)?;
        return download_asset(DEFAULT_TOKENIZER_FILENAME, path);
    }

    download_asset(DEFAULT_TOKENIZER_FILENAME, path)
}

pub fn resolve_weights(custom: Option<&Path>) -> Result<PathBuf> {
    resolve_weights_with_default(custom, Path::new(DEFAULT_WEIGHTS_PATH))
}

pub fn resolve_weights_with_default(custom: Option<&Path>, default_path: &Path) -> Result<PathBuf> {
    if let Some(path) = custom {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        return Err(anyhow!(
            "weights not found at custom path {}",
            path.display()
        ));
    }

    if default_path.exists() {
        return Ok(default_path.to_path_buf());
    }

    download_asset(DEFAULT_WEIGHTS_FILENAME, default_path)
}

fn download_asset(remote_name: &str, target: &Path) -> Result<PathBuf> {
    let provider = preferred_provider();
    let mut candidates = vec![provider];
    let alt = alternate(provider);
    if alt != provider {
        candidates.push(alt);
    }

    let mut last_err: Option<anyhow::Error> = None;
    for candidate in candidates {
        announce_provider(candidate, remote_name, target);
        match candidate {
            Provider::HuggingFace => match download_from_huggingface(remote_name, target) {
                Ok(path) => return Ok(path),
                Err(err) => last_err = Some(err),
            },
            Provider::ModelScope => match download_from_modelscope(remote_name, target) {
                Ok(path) => return Ok(path),
                Err(err) => last_err = Some(err),
            },
        }
    }

    Err(last_err.unwrap_or_else(|| {
        anyhow!(
            "failed to download {} using any configured provider",
            remote_name
        )
    }))
}

fn download_from_huggingface(remote_name: &str, target: &Path) -> Result<PathBuf> {
    let api = Api::new().context("failed to initialise Hugging Face API client")?;
    let repo = api.model(DEFAULT_REPO_ID.to_string());
    let cached = repo
        .get(remote_name)
        .with_context(|| format!("failed to download {remote_name} from Hugging Face"))?;

    copy_to_target(&cached, target)?;
    Ok(target.to_path_buf())
}

fn download_from_modelscope(remote_name: &str, target: &Path) -> Result<PathBuf> {
    let entries = modelscope_manifest()?;
    let entry = entries
        .iter()
        .find(|file| match_path(file, remote_name))
        .ok_or_else(|| {
            anyhow!(
                "file {remote_name} was not found in ModelScope manifest for {}",
                DEFAULT_MODELSCOPE_ID
            )
        })?;

    if target.exists() {
        if !target.is_file() {
            bail!(
                "download target {} exists but is not a file",
                target.display()
            );
        }
        if fs::metadata(target)?.len() == entry.size {
            return Ok(target.to_path_buf());
        }
    }

    ensure_parent(target)?;

    let url = MODELSCOPE_DOWNLOAD_URL
        .replace("{model_id}", DEFAULT_MODELSCOPE_ID)
        .replace("{path}", &entry.path);

    let response = http_client()
        .get(url)
        .send()
        .context("failed to request ModelScope download")?;

    if !response.status().is_success() {
        bail!(
            "ModelScope returned HTTP {} for {}",
            response.status(),
            remote_name
        );
    }

    let tmp_path = target.with_extension("download");
    let file = fs::File::create(&tmp_path)
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut reader = response;

    let bar = create_progress_bar(entry.size, remote_name);
    let mut downloaded = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .context("failed to read data from ModelScope")?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
        downloaded += read as u64;
        bar.inc(read as u64);
    }

    writer.flush()?;

    if downloaded != entry.size {
        bar.abandon();
        bail!(
            "downloaded {} bytes but expected {} for {}",
            downloaded,
            entry.size,
            remote_name
        );
    }

    bar.finish();

    if target.exists() {
        fs::remove_file(target)?;
    }
    fs::rename(&tmp_path, target)?;

    Ok(target.to_path_buf())
}

fn copy_to_target(cached: &Path, target: &Path) -> Result<()> {
    ensure_parent(target)?;

    if target.exists() && !target.is_file() {
        bail!(
            "download target {} exists but is not a file",
            target.display()
        );
    }

    if !target.exists() || target != cached {
        fs::copy(cached, target).with_context(|| {
            format!(
                "failed to copy cached file {} to {}",
                cached.display(),
                target.display()
            )
        })?;
    }

    Ok(())
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    Ok(())
}

fn preferred_provider() -> Provider {
    *PROVIDER_CHOICE.get_or_init(determine_preferred_provider)
}

fn determine_preferred_provider() -> Provider {
    let hf_latency = measure_hf_latency();
    let ms_latency = measure_modelscope_latency();

    match (hf_latency, ms_latency) {
        (Some(hf), Some(ms)) => {
            if ms < hf {
                Provider::ModelScope
            } else {
                Provider::HuggingFace
            }
        }
        (Some(_), None) => Provider::HuggingFace,
        (None, Some(_)) => Provider::ModelScope,
        (None, None) => Provider::default(),
    }
}

fn measure_hf_latency() -> Option<Duration> {
    let start = Instant::now();
    let api = Api::new().ok()?;
    let repo = api.model(DEFAULT_REPO_ID.to_string());
    repo.info().ok()?;
    Some(start.elapsed())
}

fn measure_modelscope_latency() -> Option<Duration> {
    if MODELSCOPE_MANIFEST.get().is_some() {
        return Some(Duration::ZERO);
    }

    let start = Instant::now();
    modelscope_manifest().ok()?;
    Some(start.elapsed())
}

fn modelscope_manifest() -> Result<&'static [ModelScopeFile]> {
    MODELSCOPE_MANIFEST
        .get_or_try_init(fetch_modelscope_manifest)
        .map(|entries| entries.as_slice())
}

fn fetch_modelscope_manifest() -> Result<Vec<ModelScopeFile>> {
    let url = MODELSCOPE_FILES_URL.replace("{model_id}", DEFAULT_MODELSCOPE_ID);
    let response = http_client()
        .get(url)
        .send()
        .context("failed to request ModelScope manifest")?;

    if !response.status().is_success() {
        bail!(
            "ModelScope returned HTTP {} while fetching manifest",
            response.status()
        );
    }

    let payload: ModelScopeResponse = response
        .json()
        .context("failed to deserialize ModelScope manifest")?;

    if payload.code != 200 {
        bail!(
            "ModelScope manifest request failed with code {}",
            payload.code
        );
    }

    let files = payload
        .data
        .files
        .into_iter()
        .filter(|file| file.kind == "blob")
        .collect();

    Ok(files)
}

fn http_client() -> &'static Client {
    HTTP_CLIENT.get_or_init(|| {
        Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent(HTTP_USER_AGENT)
            .build()
            .expect("failed to build HTTP client")
    })
}

fn alternate(provider: Provider) -> Provider {
    match provider {
        Provider::HuggingFace => Provider::ModelScope,
        Provider::ModelScope => Provider::HuggingFace,
    }
}

fn create_progress_bar(total: u64, label: &str) -> ProgressBar {
    let bar = ProgressBar::new(total);
    let style = ProgressStyle::with_template(PROGRESS_TEMPLATE)
        .expect("progress template should be valid")
        .with_key("bytes_per_sec_smoothed", SmoothedRate::default());
    bar.set_style(style);
    let message = if label.len() > PROGRESS_LABEL_MAX {
        format!("..{}", &label[label.len() - PROGRESS_LABEL_MAX..])
    } else {
        label.to_string()
    };
    bar.set_message(message);
    bar
}

fn match_path(file: &ModelScopeFile, remote_path: &str) -> bool {
    let normalized = remote_path.trim_start_matches('/');
    let path = file.path.trim_start_matches('/');
    path == normalized || file.name == normalized || path.ends_with(normalized)
}

fn announce_provider(provider: Provider, remote: &str, target: &Path) {
    println!(
        "Downloading {remote} via {} -> {}",
        provider_name(provider),
        target.display()
    );
}

fn provider_name(provider: Provider) -> &'static str {
    match provider {
        Provider::HuggingFace => "Hugging Face Hub",
        Provider::ModelScope => "ModelScope",
    }
}
