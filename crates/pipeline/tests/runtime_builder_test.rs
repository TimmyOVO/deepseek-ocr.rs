use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use deepseek_ocr_config::{fs::VirtualPath, LocalFileSystem, VirtualFileSystem};
use deepseek_ocr_pipeline::{
    DeviceKind, OcrConfigPatch, OcrFsOptions, OcrModelManager, OcrPipelineObserver, OcrRuntime,
    Precision,
};

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
    path.push(format!(
        "deepseek-ocr-pipeline-runtime-test-{pid}-{nanos}-{seq}"
    ));
    fs::create_dir_all(&path)?;
    Ok(TestTempDir { path })
}

fn test_fs_options(tmpdir: &TestTempDir) -> OcrFsOptions {
    OcrFsOptions {
        app_name: "test-deepseek-ocr".to_string(),
        config_dir: Some(tmpdir.path().join("config")),
        cache_dir: Some(tmpdir.path().join("cache")),
    }
}

fn write_config_file(fs: &LocalFileSystem, precision: Option<&str>) -> Result<PathBuf> {
    let path = fs.with_physical_path(&VirtualPath::config_file(), |p| Ok(p.to_path_buf()))?;
    let precision_line = precision
        .map(|value| format!("precision = \"{value}\"\n"))
        .unwrap_or_default();

    fs.write(
        &VirtualPath::config_file(),
        format!(
            "[models]\nactive = \"deepseek-ocr\"\n\n[models.entries.deepseek-ocr]\n\n[inference]\ndevice = \"cpu\"\n{precision_line}template = \"plain\"\nbase_size = 1024\nimage_size = 640\ncrop_mode = true\n\n[server]\nhost = \"0.0.0.0\"\nport = 8000\n"
        )
        .as_bytes(),
    )?;
    Ok(path)
}

fn defaults_layer_patch(tmp: &TestTempDir, config_path: PathBuf) -> OcrConfigPatch {
    OcrConfigPatch {
        config_path: Some(config_path),
        fs: Some(test_fs_options(tmp)),
        ..Default::default()
    }
}

fn manager(runtime: &OcrRuntime) -> &OcrModelManager {
    runtime.manager()
}

#[derive(Default)]
struct TestObserver;

impl OcrPipelineObserver for TestObserver {}

#[test]
fn build_uses_builder_precision_over_resolved_config() -> Result<()> {
    let tmp = make_temp_dir()?;
    let fs = test_fs_options(&tmp).build_local_fs();
    let config_path = write_config_file(&fs, Some("f16"))?;

    let mut cli_patch = OcrConfigPatch::default();
    cli_patch.inference.precision = Some(Precision::F32);

    let runtime = OcrRuntime::builder()
        .with_defaults_layer(defaults_layer_patch(&tmp, config_path))
        .with_cli_args_layer(cli_patch)
        .with_precision(Precision::Bf16)
        .build()?;

    assert!(matches!(manager(&runtime).device_kind(), DeviceKind::Cpu));
    assert!(matches!(
        manager(&runtime).precision(),
        Some(Precision::Bf16)
    ));
    Ok(())
}

#[test]
fn build_uses_resolved_layer_precedence_without_builder_override() -> Result<()> {
    let tmp = make_temp_dir()?;
    let fs = test_fs_options(&tmp).build_local_fs();
    let config_path = write_config_file(&fs, Some("f16"))?;

    let mut config_patch = OcrConfigPatch::default();
    config_patch.inference.precision = Some(Precision::F32);

    let mut cli_patch = OcrConfigPatch::default();
    cli_patch.inference.precision = Some(Precision::Bf16);

    let runtime = OcrRuntime::builder()
        .with_defaults_layer(defaults_layer_patch(&tmp, config_path))
        .with_config_file_layer(config_patch)
        .with_cli_args_layer(cli_patch)
        .build()?;

    assert!(matches!(manager(&runtime).device_kind(), DeviceKind::Cpu));
    assert!(matches!(
        manager(&runtime).precision(),
        Some(Precision::Bf16)
    ));
    Ok(())
}

#[test]
fn build_wires_custom_observer_into_manager() -> Result<()> {
    let tmp = make_temp_dir()?;
    let fs = test_fs_options(&tmp).build_local_fs();
    let config_path = write_config_file(&fs, None)?;

    let observer: Arc<dyn OcrPipelineObserver> = Arc::new(TestObserver);

    let runtime = OcrRuntime::builder()
        .with_defaults_layer(defaults_layer_patch(&tmp, config_path))
        .with_observer(Arc::clone(&observer))
        .build()?;

    let actual = manager(&runtime).observer() as *const dyn OcrPipelineObserver as *const ();
    let expected = Arc::as_ptr(&observer) as *const ();
    assert_eq!(actual, expected);
    Ok(())
}
