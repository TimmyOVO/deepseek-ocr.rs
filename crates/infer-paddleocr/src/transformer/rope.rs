use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};
use candle_core::{DType, Tensor, shape::D};

use crate::config::PaddleOcrVlConfig;

/// Computes cos/sin tables for Ernie's 3-axis rotary embeddings.
pub struct ErnieRotaryEmbedding {
    head_dim: usize,
    inv_freq: Vec<f32>,
    mrope_sections: Vec<usize>,
    doubled_sections: Vec<usize>,
    cfg: Arc<PaddleOcrVlConfig>,
}

impl ErnieRotaryEmbedding {
    pub fn new(cfg: Arc<PaddleOcrVlConfig>) -> Result<Self> {
        let head_dim = cfg.head_dim;
        ensure!(
            head_dim % 2 == 0,
            "rope head dim must be even (got {head_dim})"
        );
        let theta = cfg.rope_theta as f32;
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = i as f32 / half as f32;
            inv_freq.push(theta.powf(-exponent));
        }
        let sections = cfg
            .rope_scaling
            .as_ref()
            .and_then(|s| {
                if s.mrope_section.is_empty() {
                    None
                } else {
                    Some(s.mrope_section.clone())
                }
            })
            .ok_or_else(|| anyhow!("mrope_section missing from rope_scaling configuration"))?;
        ensure!(
            sections.len() == 3,
            "expected 3 sections for multimodal rope, got {}",
            sections.len()
        );
        let doubled_sections: Vec<usize> = sections.iter().map(|v| v * 2).collect();
        let total: usize = doubled_sections.iter().sum();
        ensure!(
            total == head_dim,
            "sum of doubled mrope sections {} must equal head_dim {}",
            total,
            head_dim
        );
        Ok(Self {
            head_dim,
            inv_freq,
            mrope_sections: sections,
            doubled_sections,
            cfg,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn sections(&self) -> &[usize] {
        &self.mrope_sections
    }

    pub fn doubled_sections(&self) -> &[usize] {
        &self.doubled_sections
    }

    pub fn config(&self) -> &PaddleOcrVlConfig {
        self.cfg.as_ref()
    }

    pub fn cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let (axes, batch, seq_len) = position_ids.shape().dims3()?;
        ensure!(axes == 3, "position ids must have shape [3, batch, seq]");
        let device = position_ids.device();
        let pos = if position_ids.dtype() == DType::F32 {
            position_ids.clone()
        } else {
            position_ids.to_dtype(DType::F32)?
        };
        let pos = pos.reshape((axes, batch, seq_len, 1))?;
        let inv = Tensor::from_vec(
            self.inv_freq.clone(),
            (1, 1, 1, self.inv_freq.len()),
            device,
        )?
        .expand((axes, batch, seq_len, self.inv_freq.len()))?
        .contiguous()?;
        let pos = pos
            .expand((axes, batch, seq_len, self.inv_freq.len()))?
            .contiguous()?;
        let angles = pos.mul(&inv)?;
        let cos_half = angles.cos()?;
        let sin_half = angles.sin()?;
        let cos = Tensor::cat(&[cos_half.clone(), cos_half], D::Minus1)?;
        let sin = Tensor::cat(&[sin_half.clone(), sin_half], D::Minus1)?;
        Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
    }
}
