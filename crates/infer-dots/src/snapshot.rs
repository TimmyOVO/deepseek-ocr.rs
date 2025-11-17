//! DotsOCR-specific wrapper around the shared DSQ snapshot runtime.
//!
//! This module mirrors the PaddleOCR-VL integration:
//! - DSQ snapshots are *sidecar* containers that only store a subset of
//!   quantized linear layers.
//! - Baseline safetensors weights are still required at runtime. The
//!   snapshot simply replaces selected dense matmuls with QMatMul kernels.

use anyhow::{Context, Result};
use deepseek_ocr_dsq_models::{AdapterRegistry, LinearSpec as AdapterLinearSpec};
use serde_json::to_value;

use crate::config::DotsOcrConfig;

pub use deepseek_ocr_dsq_models::AdapterScope;
pub use deepseek_ocr_dsq_runtime::*;

/// Construct runtime `LinearSpec`s for the DotsOCR adapter scope.
///
/// The underlying adapter operates on the same JSON schema as the upstream
/// `dots.ocr/config.json`. Here we serialise the already-parsed
/// [`DotsOcrConfig`] back into a `serde_json::Value` so the adapter logic
/// stays shared between the offline exporter (`deepseek-ocr-dsq-cli`) and
/// the runtime loader.
pub fn dots_snapshot_specs(
    cfg: &DotsOcrConfig,
    scope: AdapterScope,
) -> Result<Vec<deepseek_ocr_dsq_runtime::LinearSpec>> {
    let adapter = AdapterRegistry::global()
        .get("dots-ocr")
        .context("dots-ocr adapter not registered")?;
    let cfg_value =
        to_value(cfg).context("failed to serialise DotsOCR config for snapshot specs")?;
    let specs = adapter
        .discover(&cfg_value, scope)
        .context("failed to discover DotsOCR snapshot specs")?;
    Ok(specs
        .into_iter()
        .map(|spec: AdapterLinearSpec| {
            deepseek_ocr_dsq_runtime::LinearSpec::new(spec.name, spec.out_dim, spec.in_dim)
        })
        .collect())
}
