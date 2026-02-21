use std::path::PathBuf;

use deepseek_ocr_config::fs::{Namespace, VirtualPath};
use deepseek_ocr_config::VirtualFileSystem;

/// File-system related options.
///
/// 设计目标：
/// - 允许调用方显式指定 config/cache roots（而不是只能依赖环境变量）；
/// - 支持未来接入自定义 `VirtualFileSystem`（例如嵌入式/远程存储）。
#[derive(Debug, Clone)]
pub struct OcrFsOptions {
    /// 应用名称（用于默认目录名）。
    pub app_name: String,

    /// 覆盖默认 config root。
    pub config_dir: Option<PathBuf>,

    /// 覆盖默认 cache root。
    pub cache_dir: Option<PathBuf>,
}

impl Default for OcrFsOptions {
    fn default() -> Self {
        Self {
            app_name: Self::default_app_name().to_string(),
            config_dir: None,
            cache_dir: None,
        }
    }
}

impl OcrFsOptions {
    /// Returns the default application name used for filesystem directories.
    pub const fn default_app_name() -> &'static str {
        "deepseek-ocr"
    }

    /// Build a LocalFileSystem using provided overrides or fallbacks.
    pub fn build_local_fs(self) -> deepseek_ocr_config::LocalFileSystem {
        let app_name = if self.app_name.is_empty() {
            Self::default_app_name().to_string()
        } else {
            self.app_name
        };

        let (default_config_root, default_cache_root) = default_roots(&app_name);

        match (self.config_dir, self.cache_dir) {
            (Some(config_dir), Some(cache_dir)) => {
                deepseek_ocr_config::LocalFileSystem::with_directories(
                    app_name, config_dir, cache_dir,
                )
            }
            (Some(config_dir), None) => deepseek_ocr_config::LocalFileSystem::with_directories(
                app_name,
                config_dir,
                default_cache_root,
            ),
            (None, Some(cache_dir)) => deepseek_ocr_config::LocalFileSystem::with_directories(
                app_name,
                default_config_root,
                cache_dir,
            ),
            (None, None) => deepseek_ocr_config::LocalFileSystem::new(app_name),
        }
    }
}

fn default_roots(app_name: &str) -> (PathBuf, PathBuf) {
    let fs = deepseek_ocr_config::LocalFileSystem::new(app_name.to_string());

    let config_root = fs
        .with_physical_path(&VirtualPath::new(Namespace::Config, Vec::new()), |p| {
            Ok(p.to_path_buf())
        })
        .expect("config root resolution must succeed");

    let cache_root = fs
        .with_physical_path(&VirtualPath::model_dir("_root_probe_"), |p| {
            let parent = p
                .parent()
                .and_then(|p| p.parent())
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))?;
            Ok(parent.to_path_buf())
        })
        .expect("cache root resolution must succeed");

    (config_root, cache_root)
}
