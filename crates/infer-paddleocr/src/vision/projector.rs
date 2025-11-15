use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor, shape::D};
use candle_nn::VarBuilder;

use crate::{
    config::PaddleOcrVisionConfig, snapshot::SnapshotLinearMap, transformer::LinearWeights,
};

pub struct SiglipProjector {
    pre_norm: ProjectorLayerNorm,
    linear1: ProjectorLinear,
    linear2: ProjectorLinear,
    merge_size: usize,
    vision_hidden: usize,
}

#[derive(Debug)]
pub struct ProjectorOutput {
    pub embeddings: Tensor,
    pub grid: (usize, usize, usize),
}

impl SiglipProjector {
    pub fn load(
        vb: &candle_nn::VarBuilder,
        vision_cfg: &PaddleOcrVisionConfig,
        output_hidden: usize,
        model_dtype: DType,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let merge_size = vision_cfg.spatial_merge_size;
        let vision_hidden = vision_cfg.hidden_size;
        let projector_vb = vb.pp("mlp_AR");
        let compute_dtype = resolve_projector_compute_dtype(model_dtype);
        let pre_norm = ProjectorLayerNorm::load(
            projector_vb.pp("pre_norm"),
            vision_cfg.hidden_size,
            1e-5,
            compute_dtype,
        )?;
        let merged_hidden = vision_hidden * merge_size * merge_size;
        let mut snapshot_hits = snapshot_hits;
        let linear1 = ProjectorLinear::load(
            projector_vb.pp("linear_1"),
            merged_hidden,
            merged_hidden,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let linear2 = ProjectorLinear::load(
            projector_vb.pp("linear_2"),
            output_hidden,
            merged_hidden,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        Ok(Self {
            pre_norm,
            linear1,
            linear2,
            merge_size,
            vision_hidden,
        })
    }

    pub fn project_single(
        &self,
        features: &Tensor,
        grid: (usize, usize, usize),
    ) -> Result<ProjectorOutput> {
        let (t, h, w) = grid;
        ensure!(
            h % self.merge_size == 0 && w % self.merge_size == 0,
            "grid dimensions must be divisible by merge size {} (grid: {:?})",
            self.merge_size,
            grid
        );
        let expected_tokens = t * h * w;
        let (tokens, hidden) = features.dims2()?;
        ensure!(
            tokens == expected_tokens && hidden == self.vision_hidden,
            "projector expected features shaped ({expected_tokens}, {}), got ({tokens}, {hidden})",
            self.vision_hidden
        );
        let normed = self.pre_norm.forward(features)?;
        let reshaped = self.reshape_for_merge(&normed, grid)?;
        let reshaped = if reshaped.dtype() == self.pre_norm.compute_dtype() {
            reshaped
        } else {
            reshaped.to_dtype(self.pre_norm.compute_dtype())?
        };
        let merged = self.linear1.forward(&reshaped)?;
        let activated = merged.gelu()?;
        let projected = self
            .linear2
            .forward(&activated)?
            .to_dtype(features.dtype())?;
        Ok(ProjectorOutput {
            embeddings: projected,
            grid: (t, h / self.merge_size, w / self.merge_size),
        })
    }

    pub fn project_batch(
        &self,
        features: &[Tensor],
        grids: &[(usize, usize, usize)],
    ) -> Result<Vec<ProjectorOutput>> {
        ensure!(
            features.len() == grids.len(),
            "project_batch requires features and grid counts to match"
        );
        features
            .iter()
            .zip(grids.iter())
            .map(|(tensor, &grid)| self.project_single(tensor, grid))
            .collect()
    }

    fn reshape_for_merge(&self, features: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, h, w) = grid;
        let merge = self.merge_size;
        let merge_area = merge * merge;
        let merged_hidden = self.vision_hidden * merge_area;

        let reshaped = features
            .reshape((t, h, w, self.vision_hidden))?
            .reshape((t, h / merge, merge, w / merge, merge, self.vision_hidden))?
            .permute((0, 1, 3, 2, 4, 5))?
            .reshape((t * (h / merge) * (w / merge), merged_hidden))?;
        Ok(reshaped)
    }
}

impl ProjectorOutput {
    pub fn tokens(&self) -> &Tensor {
        &self.embeddings
    }

    pub fn grid(&self) -> (usize, usize, usize) {
        self.grid
    }

    pub fn into_embeddings(self) -> Tensor {
        self.embeddings
    }
}

fn resolve_projector_compute_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    }
}

struct ProjectorLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    compute_dtype: DType,
}

impl ProjectorLayerNorm {
    fn load(vb: VarBuilder, size: usize, eps: f64, compute_dtype: DType) -> Result<Self> {
        let weight = vb
            .get(size, "weight")
            .context("missing projector layernorm weight")?
            .to_dtype(compute_dtype)?;
        let bias = vb
            .get(size, "bias")
            .context("missing projector layernorm bias")?
            .to_dtype(compute_dtype)?;
        Ok(Self {
            weight,
            bias,
            eps,
            compute_dtype,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dtype = input.dtype();
        let x = if dtype == self.compute_dtype {
            input.clone()
        } else {
            input.to_dtype(self.compute_dtype)?
        };
        let hidden = x.dim(D::Minus1)?;
        let mean = (x.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let denom = (var + self.eps)?.sqrt()?;
        let normed = centered.broadcast_div(&denom)?;
        let scaled = normed.broadcast_mul(&self.weight)?;
        let shifted = scaled.broadcast_add(&self.bias)?;
        Ok(shifted.to_dtype(dtype)?)
    }

    fn compute_dtype(&self) -> DType {
        self.compute_dtype
    }
}

struct ProjectorLinear {
    weights: LinearWeights,
    compute_dtype: DType,
}

impl ProjectorLinear {
    fn load(
        vb: VarBuilder,
        out_dim: usize,
        in_dim: usize,
        compute_dtype: DType,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let weights =
            LinearWeights::load(vb, out_dim, in_dim, true, snapshot_hits, snapshot_label)?;
        Ok(Self {
            weights,
            compute_dtype,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = if input.dtype() == self.compute_dtype {
            input.clone()
        } else {
            input.to_dtype(self.compute_dtype)?
        };
        let mut out = self.weights.matmul_2d(&x)?;
        if let Some(bias) = &self.weights.bias {
            let bias = if bias.dtype() == self.compute_dtype {
                bias.clone()
            } else {
                bias.to_dtype(self.compute_dtype)?
            };
            out = out.broadcast_add(&bias.reshape((1, self.weights.out_dim))?)?;
        }
        Ok(out)
    }
}
