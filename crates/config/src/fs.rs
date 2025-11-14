use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Namespace {
    Config,
    Cache,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct VirtualPath {
    namespace: Namespace,
    segments: Vec<String>,
}

impl VirtualPath {
    pub fn new(namespace: Namespace, segments: Vec<String>) -> Self {
        Self {
            namespace,
            segments,
        }
    }

    pub fn namespace(&self) -> Namespace {
        self.namespace
    }

    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    pub fn join(&self, segment: impl Into<String>) -> Self {
        let mut segments = self.segments.clone();
        segments.push(segment.into());
        Self {
            namespace: self.namespace,
            segments,
        }
    }

    pub fn config_file() -> Self {
        Self::new(Namespace::Config, vec!["config.toml".into()])
    }

    pub fn config_dir() -> Self {
        Self::new(Namespace::Config, Vec::new())
    }

    pub fn model_dir(model_id: impl Into<String>) -> Self {
        Self::new(Namespace::Cache, vec!["models".into(), model_id.into()])
    }

    pub fn model_config(model_id: impl Into<String>) -> Self {
        Self::model_dir(model_id).join("config.json")
    }

    pub fn model_tokenizer(model_id: impl Into<String>) -> Self {
        Self::model_dir(model_id).join("tokenizer.json")
    }

    pub fn model_weights(model_id: impl Into<String>) -> Self {
        Self::model_dir(model_id).join("model.safetensors")
    }

    pub fn model_snapshot(model_id: impl Into<String>) -> Self {
        Self::model_dir(model_id).join("snapshot.dsq")
    }
}

/// Abstraction over storage backends used by the application.
pub trait VirtualFileSystem {
    fn read(&self, path: &VirtualPath) -> Result<Vec<u8>>;
    fn write(&self, path: &VirtualPath, contents: &[u8]) -> Result<()>;
    fn exists(&self, path: &VirtualPath) -> Result<bool>;
    fn ensure_dir(&self, path: &VirtualPath) -> Result<()>;
    fn ensure_parent(&self, path: &VirtualPath) -> Result<()>;
    fn remove_file(&self, path: &VirtualPath) -> Result<()>;

    fn with_physical_path<F, T>(&self, path: &VirtualPath, func: F) -> Result<T>
    where
        F: FnOnce(&Path) -> Result<T>;
}

#[derive(Debug, Clone)]
pub struct LocalFileSystem {
    app_name: String,
    config_root: PathBuf,
    cache_root: PathBuf,
}

impl LocalFileSystem {
    pub fn new(app_name: impl Into<String>) -> Self {
        let name = app_name.into();
        let config_root = env::var("DEEPSEEK_OCR_CONFIG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_config_dir(&name));
        let cache_root = env::var("DEEPSEEK_OCR_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_cache_dir(&name));
        Self {
            app_name: name,
            config_root,
            cache_root,
        }
    }

    pub fn with_directories(
        app_name: impl Into<String>,
        config_root: PathBuf,
        cache_root: PathBuf,
    ) -> Self {
        Self {
            app_name: app_name.into(),
            config_root,
            cache_root,
        }
    }

    pub fn app_name(&self) -> &str {
        &self.app_name
    }

    fn resolve<'a>(&'a self, path: &VirtualPath) -> Result<PathBuf> {
        let root = match path.namespace() {
            Namespace::Config => &self.config_root,
            Namespace::Cache => &self.cache_root,
        };
        let mut buf = root.clone();
        for segment in path.segments() {
            buf.push(segment);
        }
        Ok(buf)
    }
}

impl VirtualFileSystem for LocalFileSystem {
    fn read(&self, path: &VirtualPath) -> Result<Vec<u8>> {
        let physical = self.resolve(path)?;
        fs::read(&physical).with_context(|| format!("failed to read {}", physical.display()))
    }

    fn write(&self, path: &VirtualPath, contents: &[u8]) -> Result<()> {
        let physical = self.resolve(path)?;
        self.ensure_parent(path)?;
        fs::write(&physical, contents)
            .with_context(|| format!("failed to write {}", physical.display()))
    }

    fn exists(&self, path: &VirtualPath) -> Result<bool> {
        Ok(self.resolve(path)?.exists())
    }

    fn ensure_dir(&self, path: &VirtualPath) -> Result<()> {
        let physical = self.resolve(path)?;
        fs::create_dir_all(&physical)
            .with_context(|| format!("failed to create directory {}", physical.display()))
    }

    fn ensure_parent(&self, path: &VirtualPath) -> Result<()> {
        let physical = self.resolve(path)?;
        if let Some(parent) = physical.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        Ok(())
    }

    fn remove_file(&self, path: &VirtualPath) -> Result<()> {
        let physical = self.resolve(path)?;
        if physical.exists() {
            fs::remove_file(&physical)
                .with_context(|| format!("failed to remove {}", physical.display()))?;
        }
        Ok(())
    }

    fn with_physical_path<F, T>(&self, path: &VirtualPath, func: F) -> Result<T>
    where
        F: FnOnce(&Path) -> Result<T>,
    {
        let physical = self.resolve(path)?;
        func(&physical)
    }
}

fn default_config_dir(app_name: &str) -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| fallback_home(".config"))
        .join(app_name)
}

fn default_cache_dir(app_name: &str) -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| fallback_home(".cache"))
        .join(app_name)
}

fn fallback_home(component: &str) -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(component)
}
