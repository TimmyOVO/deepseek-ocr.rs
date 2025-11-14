use std::path::{Path, PathBuf};

use anyhow::Result;
use deepseek_ocr_assets as assets;
use deepseek_ocr_config::{
    LocalFileSystem, ResourceLocation, SnapshotResources, VirtualFileSystem, VirtualPath,
};

pub fn ensure_config_file(
    fs: &LocalFileSystem,
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

pub fn ensure_tokenizer_file(
    fs: &LocalFileSystem,
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

pub fn prepare_weights_path(
    fs: &LocalFileSystem,
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

pub fn prepare_snapshot_path(
    fs: &LocalFileSystem,
    model_id: &str,
    snapshot: Option<&SnapshotResources>,
) -> Result<Option<PathBuf>> {
    let Some(entry) = snapshot else {
        return Ok(None);
    };
    ensure_resource(fs, &entry.location, |path| {
        assets::ensure_model_snapshot_for(model_id, &entry.dtype, path)
    })
    .map(Some)
}

fn ensure_resource<F>(
    fs: &LocalFileSystem,
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
