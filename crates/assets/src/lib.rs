use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use hf_hub::api::sync::Api;

// ModelScope configuration
pub const MODELSCOPE_BASE_URL: &str = "https://www.modelscope.cn";
pub const MODELSCOPE_REPO_ID: &str = "deepseek-ai/DeepSeek-OCR";
pub const MODELSCOPE_FILES_PATH: &str = "master";

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

    download_file(
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

    download_file(
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

    download_file(
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

/// Download file from ModelScope
fn download_from_modelscope(filename: &str, target: Option<&Path>) -> Result<PathBuf> {
    // Build download URL
    let url = format!(
        "{}/{}/resolve/{}/{}",
        MODELSCOPE_BASE_URL, MODELSCOPE_REPO_ID, MODELSCOPE_FILES_PATH, filename
    );

    println!("Downloading {} from ModelScope", filename);
    println!("Download URL: {}", url);

    // Send HTTP request, allowing redirects
    let response = ureq::get(&url)
        .set("User-Agent", "deepseek-ocr-rs/0.1.0")
        .call()
        .with_context(|| format!("failed to fetch {} from ModelScope", filename))?;

    if response.status() != 200 {
        return Err(anyhow!(
            "ModelScope API returned status {} for file {}",
            response.status(),
            filename
        ));
    }

    // Determine target path
    let target_path = if let Some(path) = target {
        path.to_path_buf()
    } else {
        // If no target path specified, use current directory + filename
        PathBuf::from(filename)
    };

    // Create directory
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    // Download file
    let mut file = fs::File::create(&target_path)
        .with_context(|| format!("failed to create file {}", target_path.display()))?;

    std::io::copy(&mut response.into_reader(), &mut file)
        .with_context(|| format!("failed to write file {}", target_path.display()))?;

    println!("Download completed: {}", target_path.display());
    Ok(target_path)
}

/// Download file with fallback: try ModelScope first, fallback to Hugging Face
fn download_file(filename: &str, target: Option<&Path>) -> Result<PathBuf> {
    // Try downloading from ModelScope first
    match download_from_modelscope(filename, target) {
        Ok(path) => Ok(path),
        Err(e) => {
            println!(
                "ModelScope download failed: {}, falling back to Hugging Face",
                e
            );
            download_from_hub(filename, target)
        }
    }
}
