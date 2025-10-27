use std::path::{Path, PathBuf};

use anyhow::Result;
use deepseek_ocr_assets as assets;
use deepseek_ocr_config::{LocalFileSystem, ResourceLocation, VirtualFileSystem};

pub fn ensure_config_file(fs: &LocalFileSystem, location: &ResourceLocation) -> Result<PathBuf> {
    ensure_resource(fs, location, |path| assets::ensure_config_at(path))
}

pub fn ensure_tokenizer_file(fs: &LocalFileSystem, location: &ResourceLocation) -> Result<PathBuf> {
    ensure_resource(fs, location, |path| assets::ensure_tokenizer_at(path))
}

pub fn prepare_weights_path(fs: &LocalFileSystem, location: &ResourceLocation) -> Result<PathBuf> {
    ensure_resource(fs, location, |path| {
        assets::resolve_weights_with_default(None, path)
    })
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
