//! OCR Pipeline
//!
//! `deepseek-ocr-pipeline` 是一个“高层封装（facade）”crate：
//!
//! - 目标是让应用层（CLI/Server/第三方调用者）只依赖这一层。
//! - 内部再去组合 `deepseek-ocr-config`/`deepseek-ocr-assets`/`deepseek-ocr-core`
//!   以及各个 `deepseek-ocr-infer-*` 后端实现。
//! - 对外尽量暴露**少量且稳定**的入口对象与语义接口（类似 Transformers 的
//!   `from_pretrained` / `generate` 风格）。
//!
//! 当前处于“API 龙骨阶段” ：
//! - 重点是把能力边界与扩展点定下来，供 code review；
//! - 具体实现会在评审通过后逐步迁移落地；
//! - 因此部分方法会以 `todo!()`/`bail!()` 占位。

mod api;
mod config;
mod ext;
mod fs;
mod manager;
mod model;
mod observer;
mod pipeline;
mod runtime;

pub mod prelude;

pub use api::{OcrMessage, OcrPrompt, OcrRequest, OcrResponse, OcrRole, OcrStreamCallback};

pub use config::{
    OcrConfig, OcrConfigLayer, OcrConfigPatch, OcrConfigResolver, OcrConfigSource,
    OcrInferencePatch, OcrModelPatch, OcrPatchLayer, OcrServerPatch, OcrVisionPatch,
};

pub use fs::OcrFsOptions;

pub use manager::{OcrModelListing, OcrModelManager};

pub use model::OcrModelId;

pub use observer::{OcrPipelineEvent, OcrPipelineObserver};

pub use pipeline::{OcrPipeline, OcrPipelineHandle};

pub use runtime::{OcrRuntime, OcrRuntimeBuilder, OcrRuntimeOptions};

// Re-export small, stable enums/structs needed by callers.
//
// NOTE: 这些类型当前复用 `deepseek-ocr-core` 的定义，以避免重复维护；
// 未来如需稳定 ABI/semver 边界，可以在 pipeline 内引入自有类型并做适配。
pub use deepseek_ocr_core::{DecodeParameters, DecodeParametersPatch, ModelKind, VisionSettings};

pub use deepseek_ocr_config;
pub use deepseek_ocr_core;
pub use deepseek_ocr_core::runtime::{DeviceKind, Precision};
