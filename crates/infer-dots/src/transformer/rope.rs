use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor, shape::D};

/// Cache for RoPE cosine/sine tables keyed by `(device, dtype, rope_dim)`.
#[derive(Debug)]
pub struct RopeCache {
    cos: Tensor,
    sin: Tensor,
    len: usize,
    cap: usize,
    rope_dim: usize,
    dtype: DType,
    device: Device,
    theta: f64,
}

impl RopeCache {
    pub fn new(device: &Device, dtype: DType, rope_dim: usize, theta: f64) -> Result<Self> {
        ensure!(
            rope_dim % 2 == 0,
            "rope dimension must be even (got {rope_dim})"
        );
        Ok(Self {
            cos: Tensor::zeros((1, 1, 0, rope_dim), dtype, device)?,
            sin: Tensor::zeros((1, 1, 0, rope_dim), dtype, device)?,
            len: 0,
            cap: 0,
            rope_dim,
            dtype,
            device: device.clone(),
            theta,
        })
    }

    pub fn ensure_len(&mut self, want: usize) -> Result<()> {
        if want <= self.len {
            return Ok(());
        }
        if want > self.cap {
            let mut new_cap = self.cap.max(1);
            while new_cap < want {
                new_cap *= 2;
            }
            self.rebuild(new_cap)?;
        }
        self.len = want;
        Ok(())
    }

    fn rebuild(&mut self, new_cap: usize) -> Result<()> {
        if new_cap == 0 {
            self.cos = Tensor::zeros((1, 1, 0, self.rope_dim), self.dtype, &self.device)?;
            self.sin = Tensor::zeros((1, 1, 0, self.rope_dim), self.dtype, &self.device)?;
            self.cap = 0;
            self.len = 0;
            return Ok(());
        }
        let (cos, sin) =
            build_rope_tables(new_cap, self.rope_dim, self.theta, &self.device, self.dtype)?;
        self.cos = cos;
        self.sin = sin;
        self.cap = new_cap;
        if self.len > self.cap {
            self.len = self.cap;
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
        ensure!(
            self.len >= seq_len || position_ids.is_some(),
            "rope cache length {} insufficient for seq_len {}",
            self.len,
            seq_len
        );
        let cos = self
            .cos
            .narrow(2, 0, self.len)?
            .expand((batch, 1, self.len, self.rope_dim))?
            .contiguous()?;
        let sin = self
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
            let ids = if ids.dtype() == DType::I64 {
                ids.clone()
            } else {
                ids.to_dtype(DType::I64)?
            };
            let ids = ids
                .reshape((batch, 1, seq_len, 1))?
                .expand((batch, 1, seq_len, self.rope_dim))?
                .contiguous()?;
            Ok((cos.gather(&ids, 2)?, sin.gather(&ids, 2)?))
        } else {
            Ok((
                cos.narrow(2, self.len - seq_len, seq_len)?,
                sin.narrow(2, self.len - seq_len, seq_len)?,
            ))
        }
    }
}

fn build_rope_tables(
    cache_len: usize,
    rope_dim: usize,
    theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    ensure!(
        rope_dim % 2 == 0,
        "rope dimension must be even (got {rope_dim})"
    );
    let half = rope_dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    let theta = theta as f32;
    for i in 0..half {
        let exponent = (i * 2) as f32 / rope_dim as f32;
        inv_freq.push(theta.powf(-exponent));
    }
    let positions = Tensor::arange(0i64, cache_len as i64, device)?
        .to_dtype(DType::F32)?
        .reshape((cache_len, 1))?;
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
    let angles = positions.matmul(&inv_freq)?;
    let cos_half = angles.cos()?;
    let sin_half = angles.sin()?;
    let cos = Tensor::cat(&[cos_half.clone(), cos_half], D::Minus1)?;
    let sin = Tensor::cat(&[sin_half.clone(), sin_half], D::Minus1)?;
    let cos = cos.to_dtype(dtype)?.reshape((1, 1, cache_len, rope_dim))?;
    let sin = sin.to_dtype(dtype)?.reshape((1, 1, cache_len, rope_dim))?;
    Ok((cos, sin))
}
