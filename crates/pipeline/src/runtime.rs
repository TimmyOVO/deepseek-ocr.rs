use std::sync::Arc;

use anyhow::Result;
use deepseek_ocr_core::runtime::{default_dtype_for_device, prepare_device_and_dtype};

use crate::{
    config::{build_local_fs, OcrConfig, OcrConfigPatch, OcrConfigSource},
    ext::{patch_layer, IntoConfigOverrides},
    DeviceKind, OcrConfigResolver, OcrModelManager, OcrPipelineObserver, Precision,
};

/// Runtime options used to bootstrap the OCR pipeline.
///
/// 当前处于 API 龙骨阶段：`OcrRuntimeBuilder` 只负责表达“覆盖流程的组装点”。
#[derive(Debug, Clone, Default)]
pub struct OcrRuntimeOptions {
    pub device: Option<DeviceKind>,
    pub precision: Option<Precision>,
}

/// Builder for `OcrRuntime`.
///
/// 设计目标：
/// - 让 CLI/Server 不需要理解 config/assets/core/infer-* 的内部细节；
/// - 把“覆盖流程（layers + precedence）”集中表达在一个地方。
#[derive(Default)]
pub struct OcrRuntimeBuilder {
    observer: Option<Arc<dyn OcrPipelineObserver>>,

    defaults: Option<OcrConfigPatch>,
    config_file: Option<OcrConfigPatch>,
    cli_args: Option<OcrConfigPatch>,

    device: Option<DeviceKind>,
    precision: Option<Precision>,
}

impl OcrRuntimeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_observer(mut self, observer: Arc<dyn OcrPipelineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Lowest priority layer.
    pub fn with_defaults_layer(mut self, patch: OcrConfigPatch) -> Self {
        self.defaults = Some(patch);
        self
    }

    /// Middle priority layer.
    pub fn with_config_file_layer(mut self, patch: OcrConfigPatch) -> Self {
        self.config_file = Some(patch);
        self
    }

    /// Highest priority layer for app-level overrides.
    pub fn with_cli_args_layer(mut self, patch: OcrConfigPatch) -> Self {
        self.cli_args = Some(patch);
        self
    }

    pub fn with_device(mut self, device: DeviceKind) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = Some(precision);
        self
    }

    pub fn build(self) -> Result<OcrRuntime> {
        let OcrRuntimeBuilder {
            observer,
            defaults,
            config_file,
            cli_args,
            device,
            precision,
        } = self;

        let mut resolver = OcrConfigResolver::new();

        if let Some(defaults) = defaults {
            resolver.push_layer(patch_layer(OcrConfigSource::Defaults, defaults));
        }
        if let Some(cfg) = config_file {
            resolver.push_layer(patch_layer(OcrConfigSource::ConfigFile, cfg));
        }
        if let Some(cli) = cli_args {
            resolver.push_layer(patch_layer(OcrConfigSource::CliArgs, cli));
        }

        let merged = resolver.merged_patch()?;
        let fs = build_local_fs(merged.fs.clone());
        let base_overrides = merged.into_config_overrides()?;

        let (mut resolved, _descriptor): (OcrConfig, _) =
            deepseek_ocr_config::AppConfig::load_or_init(
                &fs,
                base_overrides.config_path.as_deref(),
            )?;
        resolved.apply_overrides(&base_overrides);
        resolved.normalise(&fs)?;

        let final_device = device.unwrap_or(resolved.inference.device);
        let final_precision = precision.or(resolved.inference.precision);
        let (prepared_device, prepared_dtype) =
            prepare_device_and_dtype(final_device, final_precision)?;
        let dtype = prepared_dtype.unwrap_or_else(|| default_dtype_for_device(&prepared_device));

        let mut manager =
            OcrModelManager::new(fs, Arc::new(resolved), final_device, final_precision, dtype);
        if let Some(observer) = observer {
            manager = manager.with_observer(observer);
        }

        Ok(OcrRuntime { manager })
    }
}

/// High-level runtime facade.
///
/// `OcrRuntime` 代表一套可复用的推理环境：
/// - 统一加载/初始化 config
/// - 统一 assets 下载与资源准备
/// - 统一设备与 dtype 选择
/// - 统一 model manager（缓存/复用）
///
/// 应用层（CLI/Server）只需要通过它来获取 `OcrPipelineHandle`。
pub struct OcrRuntime {
    manager: OcrModelManager,
}

impl OcrRuntime {
    pub fn builder() -> OcrRuntimeBuilder {
        OcrRuntimeBuilder::new()
    }

    pub fn manager(&self) -> &OcrModelManager {
        &self.manager
    }

    pub fn manager_mut(&mut self) -> &mut OcrModelManager {
        &mut self.manager
    }
}
