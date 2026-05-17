use std::sync::Arc;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tracing::info;

use crate::config::DeepseekOcrConfig;
use crate::vision::SamBackbone;

/// Manages temporary CUDA copies of SAM model for fast vision forward pass.
/// CLIP is kept on CPU permanently. The LM transformer stays on CUDA.
/// SAM is loaded on-demand from the safetensor mmap and dropped after use.
pub(crate) struct VramSwapManager {
    weights_path: PathBuf,
    cfg: Arc<DeepseekOcrConfig>,
    dtype: DType,
    cuda_device: Device,
}

impl VramSwapManager {
    pub(crate) fn new(
        weights_path: PathBuf,
        cfg: Arc<DeepseekOcrConfig>,
        dtype: DType,
        cuda_device: Device,
    ) -> Self {
        info!(
            "VramSwapManager initialized for device ({:?}) (SAM-on-demand mode)",
            cuda_device
        );
        Self { weights_path, cfg, dtype, cuda_device }
    }

    /// Load SAM on the CUDA device for fast vision encoding.
    /// The caller **must** drop the returned SAM after the vision forward
    /// pass to free VRAM for LM generation. CLIP stays on CPU.
    pub(crate) fn load_sam_cuda(&self) -> Result<SamBackbone> {
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
}
