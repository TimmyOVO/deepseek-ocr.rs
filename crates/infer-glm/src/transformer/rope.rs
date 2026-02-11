use std::sync::Arc;

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use crate::config::GlmOcrTextConfig;

pub struct GlmTextRotaryEmbedding {
    inv_freq: Vec<f32>,
    mrope_sections: Vec<usize>,
}

impl GlmTextRotaryEmbedding {
    pub fn new(cfg: Arc<GlmOcrTextConfig>) -> Result<Self> {
        let rope_theta = cfg.rope_parameters.rope_theta.unwrap_or(10_000.0) as f32;
        let partial = cfg.rope_parameters.partial_rotary_factor.unwrap_or(1.0);
        let rope_dim = ((cfg.head_dim as f64) * partial).round() as usize;
        ensure!(rope_dim.is_multiple_of(2), "rope dim must be even, got {rope_dim}");
        let half = rope_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = (i * 2) as f32 / rope_dim as f32;
            inv_freq.push(rope_theta.powf(-exponent));
        }
        let mrope_sections = if cfg.rope_parameters.mrope_section.is_empty() {
            vec![8, 12, 12]
        } else {
            cfg.rope_parameters.mrope_section.clone()
        };
        ensure!(
            mrope_sections.iter().sum::<usize>() == half,
            "mrope sections sum {} must equal rope half dim {}",
            mrope_sections.iter().sum::<usize>(),
            half
        );
        Ok(Self {
            inv_freq,
            mrope_sections,
        })
    }

    pub fn cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let (axes, batch, seq_len) = position_ids.shape().dims3()?;
        ensure!(axes == 3, "position ids must be [3, batch, seq]");
        let ids = if position_ids.dtype() == DType::I64 {
            position_ids.clone()
        } else {
            position_ids.to_dtype(DType::I64)?
        };
        let ids_host = ids.to_vec3::<i64>()?;
        let half = self.inv_freq.len();
        let head_dim = half * 2;

        let mut freqs = Vec::with_capacity(batch * seq_len * half);
        for (b, batch_ids) in ids_host[0].iter().enumerate().take(batch) {
            for (s, _) in batch_ids.iter().enumerate().take(seq_len) {
                let mut offset = 0usize;
                for (chunk_idx, width) in self.mrope_sections.iter().copied().enumerate() {
                    let axis = chunk_idx % 3;
                    let pos = ids_host[axis][b][s] as f32;
                    for j in 0..width {
                        let idx = offset + j;
                        freqs.push(pos * self.inv_freq[idx]);
                    }
                    offset += width;
                }
            }
        }

        let mut emb = Vec::with_capacity(batch * seq_len * head_dim);
        for chunk in freqs.chunks_exact(half) {
            emb.extend_from_slice(chunk);
            emb.extend_from_slice(chunk);
        }

        let emb = Tensor::from_vec(emb, (batch, seq_len, head_dim), position_ids.device())?
            .to_dtype(DType::F32)?;
        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;
        Ok((cos, sin))
    }
}
