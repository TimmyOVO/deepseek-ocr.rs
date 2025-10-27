mod progress;
mod providers;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use once_cell::sync::OnceCell;
use reqwest::blocking::Client;

use providers::providers_in_download_order;

pub const DEFAULT_REPO_ID: &str = "deepseek-ai/DeepSeek-OCR";
pub const DEFAULT_CONFIG_PATH: &str = "DeepSeek-OCR/config.json";
pub const DEFAULT_CONFIG_FILENAME: &str = "config.json";
pub const DEFAULT_TOKENIZER_PATH: &str = "DeepSeek-OCR/tokenizer.json";
pub const DEFAULT_TOKENIZER_FILENAME: &str = "tokenizer.json";
pub const DEFAULT_WEIGHTS_PATH: &str = deepseek_ocr_core::model::DEFAULT_WEIGHTS_PATH;
pub const DEFAULT_WEIGHTS_FILENAME: &str = "model-00001-of-000001.safetensors";

const HTTP_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0";

static HTTP_CLIENT: OnceCell<Client> = OnceCell::new();

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
    let mut last_err: Option<anyhow::Error> = None;
    for provider in providers_in_download_order() {
        providers::announce_provider(provider, remote_name, target);
        match provider.download(remote_name, target) {
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
