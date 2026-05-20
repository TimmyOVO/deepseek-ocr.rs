use std::sync::Arc;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tracing::info;

use crate::config::DeepseekOcrConfig;
use crate::vision::{ClipVisionModel, SamBackbone};

/// Detects total VRAM in bytes for the given CUDA device.
/// Tries nvidia-smi first, then falls back to `DEEPSEEK_OCR_VRAM_MB` env var.
pub(crate) fn get_vram_bytes(device: &Device) -> Option<u64> {
    if !device.is_cuda() {
        return None;
    }
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if let Ok(s) = String::from_utf8(output.stdout) {
            if let Ok(mb) = s.trim().parse::<u64>() {
                return Some(mb * 1024 * 1024);
            }
        }
    }
    if let Ok(vram_mb) = std::env::var("DEEPSEEK_OCR_VRAM_MB") {
        if let Ok(mb) = vram_mb.parse::<u64>() {
            return Some(mb * 1024 * 1024);
        }
    }
    None
}

/// Whether to use sequential VRAM swap (low-VRAM devices < 6GB).
pub(crate) fn should_use_vram_swap(device: &Device) -> bool {
    match get_vram_bytes(device) {
        Some(bytes) => bytes < 6 * 1024 * 1024 * 1024,
        None => false,
    }
}

/// Manages sequential VRAM swap for low-VRAM CUDA devices.
///
/// Never holds more than one heavy model in VRAM at a time:
///   Phase 1: SAM on CUDA (~1.26 GB + activations ≈ 1.6 GB peak)
///   Phase 2: CLIP on CUDA (~0.86 GB + activations ≈ 1.1 GB peak)
///
/// The LM (Q4K, ~950 MB) stays on CUDA permanently because QMatMul
/// (quantized matrix multiply) does not support device transfer.
///
/// High-VRAM devices (>6 GB) skip this entirely and run normally.
pub(crate) struct SequentialVramSwap {
    weights_path: std::path::PathBuf,
    cfg: Arc<DeepseekOcrConfig>,
    dtype: DType,
    cuda_device: Device,
}

impl SequentialVramSwap {
    pub(crate) fn new(
        weights_path: &Path,
        cfg: Arc<DeepseekOcrConfig>,
        dtype: DType,
        cuda_device: &Device,
    ) -> Self {
        info!("SequentialVramSwap initialized — one vision model at a time on CUDA");
        Self {
            weights_path: weights_path.to_path_buf(),
            cfg,
            dtype,
            cuda_device: cuda_device.clone(),
        }
    }

    /// Load SAM backbone on CUDA from the safetensor file.
    /// Caller MUST `drop()` the result before calling `load_clip_on_cuda()`
    /// or running LM generation.
    pub(crate) fn load_sam_on_cuda(&self) -> Result<SamBackbone> {
        let t0 = Instant::now();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.weights_path.as_path()],
                self.dtype,
                &self.cuda_device,
            )
        }
        .with_context(|| {
            format!(
                "failed to mmap SAM weights on CUDA from {}",
                self.weights_path.display()
            )
        })?;
        let sam = SamBackbone::new(self.cfg.as_ref(), &vb.pp("model").pp("sam_model"))
            .context("failed to load SAM backbone on CUDA")?;
        info!(
            elapsed = %format!("{:.2}s", t0.elapsed().as_secs_f32()),
            "SAM loaded on CUDA"
        );
        Ok(sam)
    }

    /// Load CLIP vision model on CUDA from the safetensor file.
    /// Caller MUST `drop()` the result before calling `load_sam_on_cuda()`
    /// or running LM generation.
    pub(crate) fn load_clip_on_cuda(&self) -> Result<ClipVisionModel> {
        let t0 = Instant::now();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.weights_path.as_path()],
                self.dtype,
                &self.cuda_device,
            )
        }
        .with_context(|| {
            format!(
                "failed to mmap CLIP weights on CUDA from {}",
                self.weights_path.display()
            )
        })?;
        let clip = ClipVisionModel::load(self.cfg.as_ref(), &vb.pp("model").pp("vision_model"))
            .context("failed to load CLIP vision model on CUDA")?;
        info!(
            elapsed = %format!("{:.2}s", t0.elapsed().as_secs_f32()),
            "CLIP loaded on CUDA"
        );
        Ok(clip)
    }

}
