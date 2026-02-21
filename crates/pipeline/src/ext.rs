//! Internal glue (extension traits / adapters).
//!
//! 该模块用于承载 pipeline 与底层 crates 之间的适配层：
//! - pipeline 对外暴露稳定语义；
//! - 内部把这些语义映射到 `deepseek-ocr-config` / `deepseek-ocr-core`。

use anyhow::Result;

use crate::config::{OcrConfigPatch, OcrConfigSource, OcrPatchLayer};

/// Converts pipeline-level patch into the concrete config override struct.
///
/// 注意：真正的优先级由 `OcrConfigResolver` 通过 layer 顺序控制。
pub trait IntoConfigOverrides {
    fn into_config_overrides(self) -> Result<deepseek_ocr_config::ConfigOverrides>;
}

impl IntoConfigOverrides for OcrConfigPatch {
    fn into_config_overrides(self) -> Result<deepseek_ocr_config::ConfigOverrides> {
        if self.model.snapshot.is_some() {
            anyhow::bail!(
                "snapshot path override is not supported yet; use DEEPSEEK_OCR_SNAPSHOT_OVERRIDE"
            );
        }

        Ok(deepseek_ocr_config::ConfigOverrides {
            config_path: self.config_path,
            model_id: self.model.id.map(|id| id.to_string()),
            model_config: self.model.config,
            tokenizer: self.model.tokenizer,
            weights: self.model.weights,
            inference: deepseek_ocr_config::InferenceOverride {
                device: self.inference.device,
                precision: self.inference.precision,
                template: self.inference.template,
                base_size: self.inference.vision.base_size,
                image_size: self.inference.vision.image_size,
                crop_mode: self.inference.vision.crop_mode,
                decode: self.inference.decode,
            },
            server: deepseek_ocr_config::config::ServerOverride {
                host: self.server.host,
                port: self.server.port,
            },
        })
    }
}

/// Helper to wrap a patch as a resolver layer.
pub fn patch_layer(source: OcrConfigSource, patch: OcrConfigPatch) -> OcrPatchLayer {
    OcrPatchLayer::new(source, patch)
}
