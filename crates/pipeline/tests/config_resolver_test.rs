use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use deepseek_ocr_config::{fs::VirtualPath, LocalFileSystem, VirtualFileSystem};

use deepseek_ocr_pipeline::{
    DeviceKind, OcrConfigPatch, OcrConfigResolver, OcrConfigSource, OcrFsOptions,
    OcrInferencePatch, OcrModelId, OcrModelPatch, OcrPatchLayer, OcrServerPatch,
};

fn patch_layer(source: OcrConfigSource, patch: OcrConfigPatch) -> OcrPatchLayer {
    OcrPatchLayer::new(source, patch)
}

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TestTempDir {
    path: PathBuf,
}

impl TestTempDir {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn make_temp_dir() -> Result<TestTempDir> {
    let mut path = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    path.push(format!("deepseek-ocr-pipeline-test-{pid}-{nanos}-{seq}"));
    fs::create_dir_all(&path)?;
    Ok(TestTempDir { path })
}

fn fs_options(tmpdir: &TestTempDir) -> OcrFsOptions {
    OcrFsOptions {
        app_name: "test-deepseek-ocr".to_string(),
        config_dir: Some(tmpdir.path().join("config")),
        cache_dir: Some(tmpdir.path().join("cache")),
    }
}

fn mk_fs(tmpdir: &TestTempDir) -> LocalFileSystem {
    let _ = fs::create_dir_all(tmpdir.path().join("config"));
    let _ = fs::create_dir_all(tmpdir.path().join("cache"));
    fs_options(tmpdir).build_local_fs()
}

fn assert_model_patch_empty(patch: &OcrModelPatch) {
    assert!(patch.id.is_none());
    assert!(patch.config.is_none());
    assert!(patch.tokenizer.is_none());
    assert!(patch.weights.is_none());
    assert!(patch.snapshot.is_none());
}

fn assert_inference_patch_empty(patch: &OcrInferencePatch) {
    assert!(patch.device.is_none());
    assert!(patch.precision.is_none());
    assert!(patch.template.is_none());
    assert!(patch.vision.base_size.is_none());
    assert!(patch.vision.image_size.is_none());
    assert!(patch.vision.crop_mode.is_none());
    assert!(patch.decode.max_new_tokens.is_none());
    assert!(patch.decode.do_sample.is_none());
    assert!(patch.decode.temperature.is_none());
    assert!(patch.decode.top_p.is_none());
    assert!(patch.decode.top_k.is_none());
    assert!(patch.decode.repetition_penalty.is_none());
    assert!(patch.decode.no_repeat_ngram_size.is_none());
    assert!(patch.decode.seed.is_none());
    assert!(patch.decode.use_cache.is_none());
}

fn assert_server_patch_empty(patch: &OcrServerPatch) {
    assert!(patch.host.is_none());
    assert!(patch.port.is_none());
}

fn write_default_config(fs: &LocalFileSystem) -> Result<std::path::PathBuf> {
    let path = fs.with_physical_path(&VirtualPath::config_file(), |p| Ok(p.to_path_buf()))?;
    fs.write(
        &VirtualPath::config_file(),
        r#"
[models]
active = "deepseek-ocr"

[models.entries.deepseek-ocr]

[inference]
device = "cpu"
template = "plain"
base_size = 1024
image_size = 640
crop_mode = true

[server]
host = "0.0.0.0"
port = 8000
"#
        .as_bytes(),
    )?;
    Ok(path)
}

#[test]
fn merge_empty_patch() -> Result<()> {
    let tmp = make_temp_dir()?;
    let fs = mk_fs(&tmp);

    write_default_config(&fs)?;

    let mut resolver = OcrConfigResolver::new();
    resolver.push_layer(patch_layer(
        OcrConfigSource::Defaults,
        OcrConfigPatch::default(),
    ));

    let merged = resolver.merged_patch()?;
    assert!(merged.config_path.is_none());
    assert!(merged.fs.is_none());
    assert_model_patch_empty(&merged.model);
    assert_inference_patch_empty(&merged.inference);
    assert_server_patch_empty(&merged.server);

    let cfg = resolver.resolve()?;
    assert_eq!(cfg.models.active, "deepseek-ocr");
    Ok(())
}

#[test]
fn merge_partial_patch() -> Result<()> {
    let tmp = make_temp_dir()?;
    let fs = mk_fs(&tmp);

    let cfg_path = write_default_config(&fs)?;

    let mut resolver = OcrConfigResolver::new();
    resolver.push_layer(patch_layer(
        OcrConfigSource::Defaults,
        OcrConfigPatch::default(),
    ));

    let patch = OcrConfigPatch {
        config_path: Some(cfg_path.clone()),
        inference: OcrInferencePatch {
            device: Some(DeviceKind::Metal),
            ..Default::default()
        },
        model: OcrModelPatch {
            id: Some(OcrModelId::try_from("paddleocr-vl")?),
            ..Default::default()
        },
        server: OcrServerPatch {
            host: Some("127.0.0.1".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    resolver.push_layer(patch_layer(OcrConfigSource::CliArgs, patch));

    let merged = resolver.merged_patch()?;
    assert_eq!(merged.config_path.as_ref(), Some(&cfg_path));
    assert!(matches!(merged.inference.device, Some(DeviceKind::Metal)));
    assert_eq!(
        merged.model.id.as_ref().map(|m| m.as_str()),
        Some("paddleocr-vl")
    );
    assert_eq!(merged.server.host.as_deref(), Some("127.0.0.1"));

    let cfg = resolver.resolve()?;
    assert_eq!(cfg.models.active, "paddleocr-vl");
    assert!(matches!(cfg.inference.device, DeviceKind::Metal));
    assert_eq!(cfg.server.host, "127.0.0.1");
    Ok(())
}

#[test]
fn merge_conflicting_patch_last_wins() -> Result<()> {
    let tmp = make_temp_dir()?;
    let fs = mk_fs(&tmp);

    write_default_config(&fs)?;

    let mut resolver = OcrConfigResolver::new();

    let mut low = OcrConfigPatch::default();
    low.inference.device = Some(DeviceKind::Cpu);
    low.model.id = Some(OcrModelId::try_from("deepseek-ocr-q4k")?);
    low.server.port = Some(8001);
    resolver.push_layer(patch_layer(OcrConfigSource::ConfigFile, low));

    let mut high = OcrConfigPatch::default();
    high.inference.device = Some(DeviceKind::Cuda);
    high.model.id = Some(OcrModelId::try_from("dots-ocr")?);
    high.server.port = Some(9000);
    resolver.push_layer(patch_layer(OcrConfigSource::CliArgs, high));

    let merged = resolver.merged_patch()?;
    assert!(matches!(merged.inference.device, Some(DeviceKind::Cuda)));
    assert_eq!(
        merged.model.id.as_ref().map(|m| m.as_str()),
        Some("dots-ocr")
    );
    assert_eq!(merged.server.port, Some(9000));

    let cfg = resolver.resolve()?;
    assert_eq!(cfg.models.active, "dots-ocr");
    assert!(matches!(cfg.inference.device, DeviceKind::Cuda));
    assert_eq!(cfg.server.port, 9000);
    Ok(())
}
