use std::{
    collections::HashMap,
    fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
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

static MODELSCOPE_MANIFESTS: OnceCell<Mutex<HashMap<String, Arc<Vec<ModelScopeFile>>>>> =
    OnceCell::new();

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
    _name: String,
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

    fn download(&self, repo_id: &str, remote_name: &str, target: &Path) -> Result<PathBuf> {
        let entries = modelscope_manifest(repo_id)?;
        let entry = entries
            .iter()
            .find(|file| match_path(file, remote_name))
            .ok_or_else(|| {
                anyhow!(
                    "file {remote_name} was not found in ModelScope manifest for {}",
                    repo_id
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
            .replace("{model_id}", repo_id)
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
        if manifest_exists(DEFAULT_MODELSCOPE_ID) {
            return Some(Duration::ZERO);
        }

        let start = Instant::now();
        modelscope_manifest(DEFAULT_MODELSCOPE_ID).ok()?;
        Some(start.elapsed())
    }
}

fn manifest_cache() -> &'static Mutex<HashMap<String, Arc<Vec<ModelScopeFile>>>> {
    MODELSCOPE_MANIFESTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn manifest_exists(repo_id: &str) -> bool {
    if let Some(cache) = MODELSCOPE_MANIFESTS.get() {
        if let Ok(guard) = cache.lock() {
            return guard.contains_key(repo_id);
        }
    }
    false
}

fn modelscope_manifest(repo_id: &str) -> Result<Arc<Vec<ModelScopeFile>>> {
    {
        let cache = manifest_cache();
        if let Some(entries) = cache
            .lock()
            .expect("ModelScope manifest cache poisoned")
            .get(repo_id)
        {
            return Ok(entries.clone());
        }
    }

    let entries = Arc::new(fetch_modelscope_manifest(repo_id)?);
    let cache = manifest_cache();
    let mut guard = cache.lock().expect("ModelScope manifest cache poisoned");
    Ok(guard
        .entry(repo_id.to_string())
        .or_insert_with(|| entries.clone())
        .clone())
}

fn fetch_modelscope_manifest(repo_id: &str) -> Result<Vec<ModelScopeFile>> {
    let url = MODELSCOPE_FILES_URL.replace("{model_id}", repo_id);
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
    let normalized = remote_path.trim_start_matches('/').replace('\\', "/");
    if file.path == normalized {
        return true;
    }
    false
}
