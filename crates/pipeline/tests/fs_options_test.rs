use std::path::PathBuf;

use deepseek_ocr_config::{LocalFileSystem, VirtualFileSystem};
use deepseek_ocr_pipeline::OcrFsOptions;

fn config_path(fs: &LocalFileSystem) -> PathBuf {
    let vp = deepseek_ocr_config::fs::VirtualPath::config_file();
    fs.with_physical_path(&vp, |p| Ok(p.to_path_buf())).unwrap()
}

fn cache_model_path(fs: &LocalFileSystem, model: &str) -> PathBuf {
    let vp = deepseek_ocr_config::fs::VirtualPath::model_config(model);
    fs.with_physical_path(&vp, |p| Ok(p.to_path_buf())).unwrap()
}

#[test]
fn default_roots() {
    let fs = OcrFsOptions::default().build_local_fs();
    let cfg = config_path(&fs);
    let cache = cache_model_path(&fs, "deepseek-ocr");

    assert!(cfg.ends_with("deepseek-ocr/config.toml"));
    assert!(cache.ends_with("deepseek-ocr/models/deepseek-ocr/config.json"));
}

#[test]
fn custom_roots() {
    let fs = OcrFsOptions {
        app_name: "custom-app".into(),
        config_dir: Some(PathBuf::from("/tmp/custom-config")),
        cache_dir: Some(PathBuf::from("/tmp/custom-cache")),
    }
    .build_local_fs();

    let cfg = config_path(&fs);
    let cache = cache_model_path(&fs, "deepseek-ocr");

    assert_eq!(cfg, PathBuf::from("/tmp/custom-config/config.toml"));
    assert_eq!(
        cache,
        PathBuf::from("/tmp/custom-cache/models/deepseek-ocr/config.json")
    );
    assert_eq!(fs.app_name(), "custom-app");
}

#[test]
fn partial_override_defaults_remaining() {
    let fs = OcrFsOptions {
        app_name: "my-app".into(),
        config_dir: Some(PathBuf::from("/tmp/only-config")),
        cache_dir: None,
    }
    .build_local_fs();

    let cfg = config_path(&fs);
    let cache = cache_model_path(&fs, "deepseek-ocr");

    assert_eq!(cfg, PathBuf::from("/tmp/only-config/config.toml"));
    assert!(cache.ends_with("my-app/models/deepseek-ocr/config.json"));
}
