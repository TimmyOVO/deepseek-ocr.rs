//! PaddleOCR-specific wrapper around the shared DSQ snapshot runtime.

use anyhow::{Context, Result};
use deepseek_ocr_dsq_models::{AdapterRegistry, LinearSpec as AdapterLinearSpec};
use serde_json::to_value;

use crate::config::PaddleOcrVlConfig;

pub use deepseek_ocr_dsq_models::AdapterScope;
pub use deepseek_ocr_dsq_runtime::*;

/// Construct runtime `LinearSpec`s for the PaddleOCR-VL adapter scope.
pub fn paddle_snapshot_specs(
    cfg: &PaddleOcrVlConfig,
    scope: AdapterScope,
) -> Result<Vec<deepseek_ocr_dsq_runtime::LinearSpec>> {
    let adapter = AdapterRegistry::global()
        .get("paddleocr-vl")
        .context("paddleocr-vl adapter not registered")?;
    let cfg_value =
        to_value(cfg).context("failed to serialize PaddleOCR-VL config for snapshot specs")?;
    let specs = adapter
        .discover(&cfg_value, scope)
        .context("failed to discover Paddle snapshot specs")?;
    Ok(specs
        .into_iter()
        .map(|spec: AdapterLinearSpec| {
            deepseek_ocr_dsq_runtime::LinearSpec::new(spec.name, spec.out_dim, spec.in_dim)
        })
        .collect())
}
