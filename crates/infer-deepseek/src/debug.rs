//! Debug-only helpers for inference.
//!
//! These are designed to be zero-cost in normal runs, and only activated when
//! an env var is set. The main gate relies on these to capture diagnostics for
//! the first mismatch step.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

/// If set, the model will compute and emit top-2 logits for a specific generated-token step.
///
/// Step definition:
/// - step=0 corresponds to the *first generated token*, selected from the prompt prefill logits.
/// - step=N corresponds to selecting token N (0-based) from the logits given prompt + first N tokens.
pub const ENV_DEBUG_LOGITS_STEP: &str = "DEEPSEEK_OCR_DEBUG_LOGITS_STEP";

/// If set, the model will write a small JSON blob containing logits diagnostics
/// to this path.
pub const ENV_DEBUG_LOGITS_JSON: &str = "DEEPSEEK_OCR_DEBUG_LOGITS_JSON";

#[derive(Debug, Clone)]
pub struct DebugLogitsConfig {
    pub step: usize,
}

#[derive(Debug, Clone)]
pub struct DebugLogitsTop2 {
    pub step: usize,
    pub top1_id: usize,
    pub top1: f32,
    pub top2_id: usize,
    pub top2: f32,
}

pub fn debug_logits_config_from_env() -> Option<DebugLogitsConfig> {
    let raw = std::env::var(ENV_DEBUG_LOGITS_STEP).ok()?;
    let step: usize = raw.parse().ok()?;
    Some(DebugLogitsConfig { step })
}

pub fn debug_logits_json_path_from_env() -> Option<PathBuf> {
    let raw = std::env::var(ENV_DEBUG_LOGITS_JSON).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(raw))
}

fn top2(values: &[f32]) -> Option<(usize, f32, usize, f32)> {
    if values.is_empty() {
        return None;
    }
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut second_i = 0usize;
    let mut second_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if v > best_v {
            second_i = best_i;
            second_v = best_v;
            best_i = i;
            best_v = v;
        } else if v > second_v {
            second_i = i;
            second_v = v;
        }
    }
    Some((best_i, best_v, second_i, second_v))
}

pub fn logits_top2_at_step(step: usize, logits_1d: &Tensor) -> Result<DebugLogitsTop2> {
    let vec = logits_1d
        .to_dtype(DType::F32)?
        .contiguous()?
        .to_vec1::<f32>()
        .context("failed to materialize logits for debug")?;
    let (t1, v1, t2, v2) = top2(&vec).context("empty logits")?;
    Ok(DebugLogitsTop2 {
        step,
        top1_id: t1,
        top1: v1,
        top2_id: t2,
        top2: v2,
    })
}

#[derive(Debug, serde::Serialize)]
struct DebugLogitsJson {
    schema_version: u32,
    generated_step: usize,
    selected_token_id: i64,
    top1_id: usize,
    top1: f32,
    top2_id: usize,
    top2: f32,
    margin: f32,
}

pub fn write_debug_logits_json(path: &Path, info: &DebugLogitsTop2, selected: i64) -> Result<()> {
    let payload = DebugLogitsJson {
        schema_version: 1,
        generated_step: info.step,
        selected_token_id: selected,
        top1_id: info.top1_id,
        top1: info.top1,
        top2_id: info.top2_id,
        top2: info.top2,
        margin: info.top1 - info.top2,
    };
    let bytes = serde_json::to_vec_pretty(&payload).context("serialize debug logits json")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    std::fs::write(path, bytes)
        .with_context(|| format!("failed to write debug logits json {}", path.display()))?;
    Ok(())
}
