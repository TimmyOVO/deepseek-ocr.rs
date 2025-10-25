use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use hf_hub::api::sync::Api;

pub const DEFAULT_REPO_ID: &str = "deepseek-ai/DeepSeek-OCR";
pub const DEFAULT_CONFIG_PATH: &str = "DeepSeek-OCR/config.json";
pub const DEFAULT_CONFIG_FILENAME: &str = "config.json";
pub const DEFAULT_TOKENIZER_PATH: &str = "DeepSeek-OCR/tokenizer.json";
pub const DEFAULT_TOKENIZER_FILENAME: &str = "tokenizer.json";
pub const DEFAULT_WEIGHTS_PATH: &str = deepseek_ocr_core::model::DEFAULT_WEIGHTS_PATH;
pub const DEFAULT_WEIGHTS_FILENAME: &str = "model-00001-of-000001.safetensors";

pub fn ensure_config() -> Result<PathBuf> {
    let default_path = PathBuf::from(DEFAULT_CONFIG_PATH);
    if default_path.exists() {
        return Ok(default_path);
    }

    let fallback = PathBuf::from(DEFAULT_CONFIG_FILENAME);
    if fallback.exists() {
        return Ok(fallback);
    }

    download_from_hub(
        DEFAULT_CONFIG_FILENAME,
        Some(Path::new(DEFAULT_CONFIG_PATH)),
    )
}

pub fn ensure_tokenizer(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    if path != Path::new(DEFAULT_TOKENIZER_PATH) {
        return Err(anyhow!("tokenizer file not found at {}", path.display()));
    }

    download_from_hub(
        DEFAULT_TOKENIZER_FILENAME,
        Some(Path::new(DEFAULT_TOKENIZER_PATH)),
    )
}

pub fn resolve_weights(custom: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = custom {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        return Err(anyhow!(
            "weights not found at custom path {}",
            path.display()
        ));
    }

    if Path::new(DEFAULT_WEIGHTS_PATH).exists() {
        return Ok(PathBuf::from(DEFAULT_WEIGHTS_PATH));
    }

    download_from_hub(
        DEFAULT_WEIGHTS_FILENAME,
        Some(Path::new(DEFAULT_WEIGHTS_PATH)),
    )
}

fn download_from_hub(filename: &str, target: Option<&Path>) -> Result<PathBuf> {
    let api = Api::new().context("failed to initialise Hugging Face API client")?;
    let repo = api.model(DEFAULT_REPO_ID.to_string());
    let cached = repo
        .get(filename)
        .with_context(|| format!("failed to download {filename} from Hugging Face"))?;

    if let Some(target_path) = target {
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }

        if target_path.exists() && !target_path.is_file() {
            return Err(anyhow!(
                "download target {} exists but is not a file",
                target_path.display()
            ));
        }

        if !target_path.exists() || target_path != cached {
            fs::copy(&cached, target_path).with_context(|| {
                format!(
                    "failed to copy cached file {} to {}",
                    cached.display(),
                    target_path.display()
                )
            })?;
        }

        Ok(target_path.to_path_buf())
    } else {
        Ok(cached)
    }
}
