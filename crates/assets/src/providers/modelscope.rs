use std::{
    fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use once_cell::sync::OnceCell;
use serde::Deserialize;

use super::AssetProvider;
use crate::{ensure_parent, http_client, progress::create_progress_bar};

const DEFAULT_MODELSCOPE_ID: &str = "deepseek-ai/DeepSeek-OCR";
const MODELSCOPE_FILES_URL: &str =
    "https://modelscope.cn/api/v1/models/{model_id}/repo/files?Recursive=true";
const MODELSCOPE_DOWNLOAD_URL: &str =
    "https://modelscope.cn/models/{model_id}/resolve/master/{path}";

static MODELSCOPE_MANIFEST: OnceCell<Vec<ModelScopeFile>> = OnceCell::new();

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

pub(crate) struct ModelScopeProvider;

impl AssetProvider for ModelScopeProvider {
    fn display_name(&self) -> &'static str {
        "ModelScope"
    }

    fn download(&self, remote_name: &str, target: &Path) -> Result<PathBuf> {
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

    fn benchmark(&self) -> Option<Duration> {
        if MODELSCOPE_MANIFEST.get().is_some() {
            return Some(Duration::ZERO);
        }

        let start = Instant::now();
        modelscope_manifest().ok()?;
        Some(start.elapsed())
    }
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

fn match_path(file: &ModelScopeFile, remote_path: &str) -> bool {
    let normalized = remote_path.trim_start_matches('/');
    let path = file.path.trim_start_matches('/');
    path == normalized || file.name == normalized || path.ends_with(normalized)
}
