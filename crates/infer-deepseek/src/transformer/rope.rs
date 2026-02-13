use crate::config::DeepseekV2Config;
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use deepseek_ocr_core::tensor::to_dtype_if_needed;

#[cfg(feature = "memlog")]
use deepseek_ocr_core::memlog;

/// Growable cache for RoPE cosine/sine tables keyed by `(dtype, rope_dim, device)`.
///
/// Capacity grows geometrically so we only reallocate `O(log T)` times as the decode sequence
/// length increases.
pub struct RopeCache {
    cos: Tensor, // shape (1, 1, cap, rope_dim)
    sin: Tensor, // shape (1, 1, cap, rope_dim)
    len: usize,  // logical length currently computed (<= cap)
    cap: usize,  // allocated capacity in positions
    rope_dim: usize,
    dtype: DType,
    device: Device,
    force_f32: bool,
}

impl RopeCache {
    pub fn new(device: &Device, dtype: DType, rope_dim: usize) -> Result<Self> {
        ensure!(
            rope_dim.is_multiple_of(2),
            "rope dimension must be even (got {rope_dim})"
        );
        let force_f32 = matches!(dtype, DType::F16 | DType::BF16);
        let cache_dtype = if force_f32 { DType::F32 } else { dtype };
        let cos = Tensor::zeros((1, 1, 0, rope_dim), cache_dtype, device)?;
        let sin = Tensor::zeros((1, 1, 0, rope_dim), cache_dtype, device)?;
        Ok(Self {
            cos,
            sin,
            len: 0,
            cap: 0,
            rope_dim,
            dtype,
            device: device.clone(),
            force_f32,
        })
    }

    pub fn matches(&self, dtype: DType, rope_dim: usize, device: &Device) -> bool {
        self.dtype == dtype
            && self.rope_dim == rope_dim
            && self.device.location() == device.location()
    }

    pub fn ensure_len(&mut self, cfg: &DeepseekV2Config, want: usize) -> Result<()> {
        if want <= self.len {
            return Ok(());
        }
        if want > self.cap {
            let mut new_cap = self.cap.max(1);
            while new_cap < want {
                new_cap *= 2;
            }
            self.rebuild(cfg, new_cap)?;
        }
        self.len = want;
        Ok(())
    }

    fn rebuild(&mut self, cfg: &DeepseekV2Config, new_cap: usize) -> Result<()> {
        let cache_dtype = if self.force_f32 {
            DType::F32
        } else {
            self.dtype
        };
        if new_cap == 0 {
            self.cos = Tensor::zeros((1, 1, 0, self.rope_dim), cache_dtype, &self.device)?;
            self.sin = Tensor::zeros((1, 1, 0, self.rope_dim), cache_dtype, &self.device)?;
            self.cap = 0;
            self.len = 0;
            #[cfg(feature = "memlog")]
            memlog::set_rope(0);
            return Ok(());
        }

        let (cos, sin) = build_rope_tables(cfg, new_cap, self.rope_dim, &self.device, cache_dtype)?;
        self.cos = cos;
        self.sin = sin;
        self.cap = new_cap;
        if self.len > self.cap {
            self.len = self.cap;
        }
        #[cfg(feature = "memlog")]
        {
            let total = memlog::tensor_bytes(&self.cos) + memlog::tensor_bytes(&self.sin);
            memlog::set_rope(total);
        }
        Ok(())
    }

    pub fn select(
        &self,
        batch: usize,
        seq_len: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if seq_len == 0 {
            let zeros = Tensor::zeros((batch, 1, 0, self.rope_dim), self.dtype, &self.device)?;
            return Ok((zeros.clone(), zeros));
        }
        if position_ids.is_none() {
            ensure!(
                self.len >= seq_len,
                "rope cache length {} insufficient for seq_len {}",
                self.len,
                seq_len
            );
        }
        let cos_active = self
            .cos
            .narrow(2, 0, self.len)?
            .expand((batch, 1, self.len, self.rope_dim))?
            .contiguous()?;
        let sin_active = self
            .sin
            .narrow(2, 0, self.len)?
            .expand((batch, 1, self.len, self.rope_dim))?
            .contiguous()?;

        if let Some(ids) = position_ids {
            ensure!(
                ids.shape().dims() == [batch, seq_len],
                "position_ids shape {:?} must match (batch={}, seq={})",
                ids.shape().dims(),
                batch,
                seq_len
            );
            let ids_i64 = to_dtype_if_needed(ids, DType::I64)?;
            let max_pos = ids_i64
                .to_device(&Device::Cpu)?
                .max_all()?
                .to_scalar::<i64>()? as usize;
            ensure!(
                self.len > max_pos,
                "rope cache length {} insufficient for max position {}",
                self.len,
                max_pos
            );
            let ids = ids_i64
                .reshape((batch, 1, seq_len, 1))?
                .expand((batch, 1, seq_len, self.rope_dim))?
                .contiguous()?;
            let cos = cos_active.gather(&ids, 2)?;
            let sin = sin_active.gather(&ids, 2)?;
            Ok((cos, sin))
        } else {
            let cos = cos_active.narrow(2, 0, seq_len)?;
            let sin = sin_active.narrow(2, 0, seq_len)?;
            Ok((cos, sin))
        }
    }

    pub fn rope_dim(&self) -> usize {
        self.rope_dim
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

fn build_rope_tables(
    cfg: &DeepseekV2Config,
    cache_len: usize,
    rope_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    ensure!(
        rope_dim.is_multiple_of(2),
        "rope dimension must be even, got {rope_dim}"
    );
    let base = cfg.rope_theta;
    let half = rope_dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = (i as f32 * 2.0) / (rope_dim as f32);
        inv_freq.push(1.0f32 / base.powf(exponent));
    }

    let pos = Tensor::arange(0i64, cache_len as i64, device)?
        .to_dtype(DType::F32)?
        .reshape((cache_len, 1))?;
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
    let angles = pos.matmul(&inv_freq)?;
    let cos_half = angles.cos()?;
    let sin_half = angles.sin()?;
    let cos_full = Tensor::cat(&[cos_half.clone(), cos_half], 1)?;
    let sin_full = Tensor::cat(&[sin_half.clone(), sin_half], 1)?;
    let cos = cos_full
        .to_dtype(dtype)?
        .reshape((1, 1, cache_len, rope_dim))?;
    let sin = sin_full
        .to_dtype(dtype)?
        .reshape((1, 1, cache_len, rope_dim))?;
    Ok((cos, sin))
}
