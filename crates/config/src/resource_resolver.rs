use std::path::{Path, PathBuf};

use anyhow::Result;
use deepseek_ocr_assets as assets;

use crate::{ResourceLocation, SnapshotResources, VirtualFileSystem, VirtualPath};

#[derive(Debug, Clone)]
pub struct PreparedModelPaths {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
    pub snapshot: Option<PathBuf>,
    pub preprocessor: Option<PathBuf>,
}

pub fn prepare_model_paths(
    fs: &impl VirtualFileSystem,
    model_id: &str,
    config_location: &ResourceLocation,
    tokenizer_location: &ResourceLocation,
    weights_location: &ResourceLocation,
    snapshot: Option<&SnapshotResources>,
) -> Result<PreparedModelPaths> {
    let config = ensure_config_file(fs, model_id, config_location)?;
    let tokenizer = ensure_tokenizer_file(fs, model_id, tokenizer_location)?;
    let weights = prepare_weights_path(fs, model_id, weights_location)?;
    let snapshot = prepare_snapshot_path(fs, model_id, snapshot)?;
    let preprocessor = ensure_model_preprocessor_file(model_id, &config)?;

    Ok(PreparedModelPaths {
        config,
        tokenizer,
        weights,
        snapshot,
        preprocessor,
    })
}

fn ensure_config_file(
    fs: &impl VirtualFileSystem,
    model_id: &str,
    location: &ResourceLocation,
) -> Result<PathBuf> {
    match location {
        ResourceLocation::Physical(path) => assets::ensure_model_config_for(model_id, path),
        ResourceLocation::Virtual(_) => {
            let baseline_id = assets::baseline_model_id(model_id);
            if baseline_id == model_id {
                ensure_resource(fs, location, |path| {
                    assets::ensure_model_config_for(&baseline_id, path)
                })
            } else {
                let vpath = VirtualPath::model_config(baseline_id.clone());
                fs.with_physical_path(&vpath, |physical| {
                    assets::ensure_model_config_for(&baseline_id, physical)
                })
            }
        }
    }
}

fn ensure_tokenizer_file(
    fs: &impl VirtualFileSystem,
    model_id: &str,
    location: &ResourceLocation,
) -> Result<PathBuf> {
    match location {
        ResourceLocation::Physical(path) => assets::ensure_model_tokenizer_for(model_id, path),
        ResourceLocation::Virtual(_) => {
            let baseline_id = assets::baseline_model_id(model_id);
            if baseline_id == model_id {
                ensure_resource(fs, location, |path| {
                    assets::ensure_model_tokenizer_for(&baseline_id, path)
                })
            } else {
                let vpath = VirtualPath::model_tokenizer(baseline_id.clone());
                fs.with_physical_path(&vpath, |physical| {
                    assets::ensure_model_tokenizer_for(&baseline_id, physical)
                })
            }
        }
    }
}

fn prepare_weights_path(
    fs: &impl VirtualFileSystem,
    model_id: &str,
    location: &ResourceLocation,
) -> Result<PathBuf> {
    let baseline_id = assets::baseline_model_id(model_id);
    if baseline_id == model_id {
        ensure_resource(fs, location, |path| {
            assets::ensure_model_weights_for(model_id, path)
        })
    } else {
        let vpath = VirtualPath::model_weights(baseline_id.clone());
        fs.with_physical_path(&vpath, |physical| {
            assets::ensure_model_weights_for(&baseline_id, physical)
        })
    }
}

fn prepare_snapshot_path(
    fs: &impl VirtualFileSystem,
    model_id: &str,
    snapshot: Option<&SnapshotResources>,
) -> Result<Option<PathBuf>> {
    // Debug-only escape hatch: when set, this environment variable forces a
    // specific snapshot path regardless of the configured model entry. This
    // is primarily intended for local testing of freshly exported `.dsq`
    // artifacts.
    if let Ok(path_str) = std::env::var("DEEPSEEK_OCR_SNAPSHOT_OVERRIDE")
        && !path_str.trim().is_empty()
    {
        return Ok(Some(PathBuf::from(path_str)));
    }

    let Some(entry) = snapshot else {
        return Ok(None);
    };
    ensure_resource(fs, &entry.location, |path| {
        assets::ensure_model_snapshot_for(model_id, &entry.dtype, path)
    })
    .map(Some)
}

fn ensure_model_preprocessor_file(model_id: &str, config_path: &Path) -> Result<Option<PathBuf>> {
    assets::ensure_model_preprocessor_for(model_id, config_path)
}

fn ensure_resource<F>(
    fs: &impl VirtualFileSystem,
    location: &ResourceLocation,
    ensure_fn: F,
) -> Result<PathBuf>
where
    F: Fn(&Path) -> Result<PathBuf>,
{
    match location {
        ResourceLocation::Physical(path) => ensure_fn(path),
        ResourceLocation::Virtual(vpath) => {
            fs.with_physical_path(vpath, |physical| ensure_fn(physical))
        }
    }
}
