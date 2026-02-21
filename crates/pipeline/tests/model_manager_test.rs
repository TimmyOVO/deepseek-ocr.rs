use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use candle_core::DType;
use deepseek_ocr_config::AppConfig;
use deepseek_ocr_config::LocalFileSystem;
use deepseek_ocr_pipeline::{
    DeviceKind, OcrModelId, OcrModelManager, OcrPipelineEvent, OcrPipelineObserver,
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
        "deepseek-ocr-pipeline-manager-test-{pid}-{nanos}-{seq}"
    ));
    fs::create_dir_all(&path)?;
    Ok(TestTempDir { path })
}

fn write_invalid_model_files(root: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let config_path = root.join("model-config.json");
    let tokenizer_path = root.join("tokenizer.json");
    let weights_path = root.join("model.safetensors");
    fs::write(&config_path, b"{}")?;
    fs::write(&tokenizer_path, b"{}")?;
    fs::write(&weights_path, b"")?;
    Ok((config_path, tokenizer_path, weights_path))
}

fn write_config_file(
    root: &Path,
    config_path: &Path,
    tokenizer_path: &Path,
    weights_path: &Path,
) -> Result<PathBuf> {
    let app_config_path = root.join("config.toml");
    let content = format!(
        "[models]\nactive = \"deepseek-ocr\"\n\n[models.entries.deepseek-ocr]\nkind = \"deepseek\"\nconfig = \"{}\"\ntokenizer = \"{}\"\nweights = \"{}\"\n\n[inference]\ndevice = \"cpu\"\ntemplate = \"plain\"\nbase_size = 1024\nimage_size = 640\ncrop_mode = true\nmax_new_tokens = 8\nuse_cache = true\n\n[server]\nhost = \"127.0.0.1\"\nport = 8000\n",
        config_path.display(),
        tokenizer_path.display(),
        weights_path.display()
    );
    fs::write(&app_config_path, content)?;
    Ok(app_config_path)
}

#[derive(Default)]
struct CountingObserver {
    started: AtomicUsize,
}

impl CountingObserver {
    fn started_count(&self) -> usize {
        self.started.load(Ordering::SeqCst)
    }
}

impl OcrPipelineObserver for CountingObserver {
    fn on_event(&self, event: &OcrPipelineEvent) {
        if matches!(event, OcrPipelineEvent::ModelLoadStarted { .. }) {
            self.started.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[test]
fn load_uses_singleflight_for_same_model_id_even_on_failure() -> Result<()> {
    let tmp = make_temp_dir()?;
    let config_root = tmp.path().join("config-root");
    let cache_root = tmp.path().join("cache-root");
    fs::create_dir_all(&config_root)?;
    fs::create_dir_all(&cache_root)?;

    let (model_config, model_tokenizer, model_weights) = write_invalid_model_files(tmp.path())?;
    let app_config_path =
        write_config_file(tmp.path(), &model_config, &model_tokenizer, &model_weights)?;

    let fs = LocalFileSystem::with_directories("pipeline-test", config_root, cache_root);
    let (config, _descriptor) = AppConfig::load_or_init(&fs, Some(&app_config_path))?;

    let observer = Arc::new(CountingObserver::default());
    let observer_trait: Arc<dyn OcrPipelineObserver> = observer.clone();
    let manager = Arc::new(
        OcrModelManager::new(fs, Arc::new(config), DeviceKind::Cpu, None, DType::F32)
            .with_observer(observer_trait),
    );
    let model_id = OcrModelId::try_from("deepseek-ocr")?;

    let workers = 8usize;
    let barrier = Arc::new(Barrier::new(workers));
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let barrier = Arc::clone(&barrier);
        let manager = Arc::clone(&manager);
        let model_id = model_id.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            manager.load(&model_id)
        }));
    }

    for handle in handles {
        let result = handle.join().expect("worker thread panicked");
        assert!(result.is_err(), "expected invalid model resources to fail");
    }

    assert_eq!(
        observer.started_count(),
        1,
        "real model load should happen exactly once"
    );

    Ok(())
}
