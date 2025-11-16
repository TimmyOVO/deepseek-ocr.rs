mod progress;
mod providers;

use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use deepseek_ocr_core::ModelKind;
use once_cell::sync::OnceCell;
use reqwest::blocking::Client;
use serde::Deserialize;

use providers::providers_in_download_order;

// Primary DeepSeek repo used for benchmarking provider connectivity.
pub const DEFAULT_REPO_ID: &str = "deepseek-ai/DeepSeek-OCR";

const HTTP_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0";

static HTTP_CLIENT: OnceCell<Client> = OnceCell::new();

#[derive(Clone, Copy)]
pub struct SnapshotAsset {
    pub dtype: &'static str,
    pub repo_id: &'static str,
    pub filename: &'static str,
}

#[derive(Clone, Copy)]
pub struct ModelAsset {
    pub id: &'static str,
    pub kind: ModelKind,
    pub repo_id: &'static str,
    pub config: &'static str,
    pub tokenizer: &'static str,
    pub weights: &'static str,
}

#[derive(Clone, Copy)]
pub struct QuantizedModelAsset {
    pub id: &'static str,
    pub kind: ModelKind,
    pub baseline_id: &'static str,
    pub snapshot: SnapshotAsset,
}

pub const MODEL_ASSETS: &[ModelAsset] = &[
    ModelAsset {
        id: "deepseek-ocr",
        kind: ModelKind::Deepseek,
        repo_id: DEFAULT_REPO_ID,
        config: "config.json",
        tokenizer: "tokenizer.json",
        weights: "model-00001-of-000001.safetensors",
    },
    ModelAsset {
        id: "paddleocr-vl",
        kind: ModelKind::PaddleOcrVl,
        repo_id: "PaddlePaddle/PaddleOCR-VL",
        config: "config.json",
        tokenizer: "tokenizer.json",
        weights: "model.safetensors",
    },
    ModelAsset {
        id: "dots-ocr",
        kind: ModelKind::DotsOcr,
        repo_id: "rednote-hilab/dots.ocr",
        config: "config.json",
        tokenizer: "tokenizer.json",
        weights: "model.safetensors.index.json",
    },
];

pub const QUANTIZED_MODEL_ASSETS: &[QuantizedModelAsset] = &[
    QuantizedModelAsset {
        id: "deepseek-ocr-q4k",
        kind: ModelKind::Deepseek,
        baseline_id: "deepseek-ocr",
        snapshot: SnapshotAsset {
            dtype: "Q4_K",
            repo_id: "TimmyOVO/deepseek-ocr.rs",
            filename: "DeepSeek-OCR.Q4_K.dsq",
        },
    },
    QuantizedModelAsset {
        id: "deepseek-ocr-q6k",
        kind: ModelKind::Deepseek,
        baseline_id: "deepseek-ocr",
        snapshot: SnapshotAsset {
            dtype: "Q6_K",
            repo_id: "TimmyOVO/deepseek-ocr.rs",
            filename: "DeepSeek-OCR.Q6_K.dsq",
        },
    },
    QuantizedModelAsset {
        id: "deepseek-ocr-q8k",
        kind: ModelKind::Deepseek,
        baseline_id: "deepseek-ocr",
        snapshot: SnapshotAsset {
            dtype: "Q8_0",
            repo_id: "TimmyOVO/deepseek-ocr.rs",
            filename: "DeepSeek-OCR.Q8_0.dsq",
        },
    },
    QuantizedModelAsset {
        id: "paddleocr-vl-q4k",
        kind: ModelKind::PaddleOcrVl,
        baseline_id: "paddleocr-vl",
        snapshot: SnapshotAsset {
            dtype: "Q4_K",
            repo_id: "TimmyOVO/PaddleOCR-VL-Quantization",
            filename: "PaddleOCR-VL.Q4_K.dsq",
        },
    },
    QuantizedModelAsset {
        id: "paddleocr-vl-q6k",
        kind: ModelKind::PaddleOcrVl,
        baseline_id: "paddleocr-vl",
        snapshot: SnapshotAsset {
            dtype: "Q6_K",
            repo_id: "TimmyOVO/PaddleOCR-VL-Quantization",
            filename: "PaddleOCR-VL.Q6_K.dsq",
        },
    },
    QuantizedModelAsset {
        id: "paddleocr-vl-q8k",
        kind: ModelKind::PaddleOcrVl,
        baseline_id: "paddleocr-vl",
        snapshot: SnapshotAsset {
            dtype: "Q8_0",
            repo_id: "TimmyOVO/PaddleOCR-VL-Quantization",
            filename: "PaddleOCR-VL.Q8_0.dsq",
        },
    },
];

pub fn baseline_model_id(model_id: &str) -> String {
    if let Some(q) = quantized_asset_profile(model_id) {
        q.baseline_id.to_string()
    } else {
        model_id.to_string()
    }
}

pub fn ensure_model_config_for(model_id: &str, target: &Path) -> Result<PathBuf> {
    let profile = asset_profile(model_id)?;
    ensure_remote_file(profile.repo_id, profile.config, target)
}

pub fn ensure_model_tokenizer_for(model_id: &str, target: &Path) -> Result<PathBuf> {
    let profile = asset_profile(model_id)?;
    ensure_remote_file(profile.repo_id, profile.tokenizer, target)
}

pub fn ensure_model_weights_for(model_id: &str, target: &Path) -> Result<PathBuf> {
    let profile = asset_profile(model_id)?;
    let path = ensure_remote_file(profile.repo_id, profile.weights, target)?;
    if profile.weights.ends_with(".index.json") {
        ensure_index_shards(profile.repo_id, &path)?;
    }
    Ok(path)
}

pub fn ensure_model_snapshot_for(model_id: &str, dtype: &str, target: &Path) -> Result<PathBuf> {
    let snapshot = snapshot_profile(model_id, dtype)?;
    ensure_remote_file(snapshot.repo_id, snapshot.filename, target)
}

fn ensure_remote_file(repo_id: &str, remote_name: &str, target: &Path) -> Result<PathBuf> {
    if target.exists() {
        return Ok(target.to_path_buf());
    }

    download_asset(repo_id, remote_name, target)
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

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

fn ensure_index_shards(repo_id: &str, index_path: &Path) -> Result<()> {
    let bytes = fs::read(index_path).with_context(|| {
        format!(
            "failed to read downloaded index at {}",
            index_path.display()
        )
    })?;
    let index: WeightIndex = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to parse safetensors index at {}",
            index_path.display()
        )
    })?;
    ensure!(
        !index.weight_map.is_empty(),
        "index {} lists no shards",
        index_path.display()
    );
    let parent = index_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut shards: BTreeSet<String> = BTreeSet::new();
    for value in index.weight_map.values() {
        shards.insert(value.clone());
    }
    for shard in &shards {
        let local = parent.join(shard);
        ensure_remote_file(repo_id, shard, &local)?;
    }
    Ok(())
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

fn asset_profile(model_id: &str) -> Result<&'static ModelAsset> {
    let baseline = baseline_model_id(model_id);
    MODEL_ASSETS
        .iter()
        .find(|asset| asset.id == baseline)
        .ok_or_else(|| anyhow!("unknown model id `{model_id}` (baseline `{baseline}`)"))
}

fn quantized_asset_profile(model_id: &str) -> Option<&'static QuantizedModelAsset> {
    QUANTIZED_MODEL_ASSETS
        .iter()
        .find(|asset| asset.id == model_id)
}

fn snapshot_profile(model_id: &str, dtype: &str) -> Result<&'static SnapshotAsset> {
    let asset = quantized_asset_profile(model_id)
        .ok_or_else(|| anyhow!("model `{}` has no quantized snapshot configured", model_id))?;
    let snapshot = &asset.snapshot;
    if snapshot.dtype.eq_ignore_ascii_case(dtype) {
        Ok(snapshot)
    } else {
        Err(anyhow!(
            "snapshot dtype `{}` not configured for model `{}`",
            dtype,
            model_id
        ))
    }
}
