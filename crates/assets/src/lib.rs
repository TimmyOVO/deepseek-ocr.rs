mod progress;
mod providers;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use deepseek_ocr_core::ModelKind;
use once_cell::sync::OnceCell;
use reqwest::blocking::Client;

use providers::providers_in_download_order;

pub const DEFAULT_REPO_ID: &str = "deepseek-ai/DeepSeek-OCR";
pub const DEFAULT_CONFIG_PATH: &str = "DeepSeek-OCR/config.json";
pub const DEFAULT_CONFIG_FILENAME: &str = "config.json";
pub const DEFAULT_TOKENIZER_PATH: &str = "DeepSeek-OCR/tokenizer.json";
pub const DEFAULT_TOKENIZER_FILENAME: &str = "tokenizer.json";
pub const DEFAULT_WEIGHTS_PATH: &str = deepseek_ocr_infer_deepseek::model::DEFAULT_WEIGHTS_PATH;
pub const DEFAULT_WEIGHTS_FILENAME: &str = "model-00001-of-000001.safetensors";

const PADDLE_REPO_ID: &str = "PaddlePaddle/PaddleOCR-VL";
const PADDLE_CONFIG_FILENAME: &str = "config.json";
const PADDLE_TOKENIZER_FILENAME: &str = "tokenizer.json";
const PADDLE_WEIGHTS_FILENAME: &str = "model.safetensors";

const HTTP_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0";

static HTTP_CLIENT: OnceCell<Client> = OnceCell::new();

pub fn ensure_config() -> Result<PathBuf> {
    ensure_config_at(Path::new(DEFAULT_CONFIG_PATH))
}

pub fn ensure_config_at(target: &Path) -> Result<PathBuf> {
    ensure_model_config(ModelKind::Deepseek, target)
}

pub fn ensure_tokenizer(path: &Path) -> Result<PathBuf> {
    ensure_tokenizer_at(path)
}

pub fn ensure_tokenizer_at(path: &Path) -> Result<PathBuf> {
    ensure_model_tokenizer(ModelKind::Deepseek, path)
}

pub fn resolve_weights(custom: Option<&Path>) -> Result<PathBuf> {
    resolve_weights_with_default(custom, Path::new(DEFAULT_WEIGHTS_PATH))
}

pub fn resolve_weights_with_default(custom: Option<&Path>, default_path: &Path) -> Result<PathBuf> {
    resolve_weights_with_kind(custom, default_path, ModelKind::Deepseek)
}

pub fn resolve_weights_with_kind(
    custom: Option<&Path>,
    default_path: &Path,
    kind: ModelKind,
) -> Result<PathBuf> {
    if let Some(path) = custom {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        return Err(anyhow!(
            "weights not found at custom path {}",
            path.display()
        ));
    }

    ensure_model_weights(kind, default_path)
}

pub fn ensure_model_config(kind: ModelKind, target: &Path) -> Result<PathBuf> {
    ensure_asset(kind, target, AssetFile::Config)
}

pub fn ensure_model_tokenizer(kind: ModelKind, target: &Path) -> Result<PathBuf> {
    ensure_asset(kind, target, AssetFile::Tokenizer)
}

pub fn ensure_model_weights(kind: ModelKind, target: &Path) -> Result<PathBuf> {
    ensure_asset(kind, target, AssetFile::Weights)
}

fn download_asset(repo_id: &str, remote_name: &str, target: &Path) -> Result<PathBuf> {
    let mut last_err: Option<anyhow::Error> = None;
    for provider in providers_in_download_order() {
        providers::announce_provider(provider, repo_id, remote_name, target);
        match provider.download(repo_id, remote_name, target) {
            Ok(path) => return Ok(path),
            Err(err) => last_err = Some(err),
        }
    }

    Err(last_err.unwrap_or_else(|| {
        anyhow!(
            "failed to download {} using any configured provider",
            remote_name
        )
    }))
}

pub(crate) fn copy_to_target(cached: &Path, target: &Path) -> Result<()> {
    ensure_parent(target)?;

    if target.exists() && !target.is_file() {
        bail!(
            "download target {} exists but is not a file",
            target.display()
        );
    }

    if !target.exists() || target != cached {
        std::fs::copy(cached, target).with_context(|| {
            format!(
                "failed to copy cached file {} to {}",
                cached.display(),
                target.display()
            )
        })?;
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum AssetFile {
    Config,
    Tokenizer,
    Weights,
}

struct ModelAssetSpec {
    repo_id: &'static str,
    config: &'static str,
    tokenizer: &'static str,
    weights: &'static str,
}

fn asset_spec(kind: ModelKind) -> ModelAssetSpec {
    match kind {
        ModelKind::Deepseek => ModelAssetSpec {
            repo_id: DEFAULT_REPO_ID,
            config: DEFAULT_CONFIG_FILENAME,
            tokenizer: DEFAULT_TOKENIZER_FILENAME,
            weights: DEFAULT_WEIGHTS_FILENAME,
        },
        ModelKind::PaddleOcrVl => ModelAssetSpec {
            repo_id: PADDLE_REPO_ID,
            config: PADDLE_CONFIG_FILENAME,
            tokenizer: PADDLE_TOKENIZER_FILENAME,
            weights: PADDLE_WEIGHTS_FILENAME,
        },
    }
}

fn ensure_asset(kind: ModelKind, target: &Path, file: AssetFile) -> Result<PathBuf> {
    if target.exists() {
        return Ok(target.to_path_buf());
    }

    let spec = asset_spec(kind);
    let remote = match file {
        AssetFile::Config => spec.config,
        AssetFile::Tokenizer => spec.tokenizer,
        AssetFile::Weights => spec.weights,
    };

    download_asset(spec.repo_id, remote, target)
}

pub(crate) fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    Ok(())
}

pub(crate) fn http_client() -> &'static Client {
    HTTP_CLIENT.get_or_init(|| {
        Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent(HTTP_USER_AGENT)
            .build()
            .expect("failed to build HTTP client")
    })
}
