use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Module, Tensor, shape::D};
use candle_nn::{Linear, VarBuilder, ops::softmax};
use deepseek_ocr_core::tensor::gather_token_embeddings;

use super::SiglipImagePatches;
use crate::{
    config::PaddleOcrVisionConfig, snapshot::SnapshotLinearMap, transformer::LinearWeights,
};

pub struct SiglipVisionModel {
    embeddings: SiglipEmbeddings,
    encoder: SiglipEncoder,
    post_layernorm: PreciseLayerNorm,
    compute_dtype: DType,
}

impl SiglipVisionModel {
    pub fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVisionConfig,
        model_dtype: DType,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let compute_dtype = resolve_compute_dtype(model_dtype);
        let vision_vb = vb.pp("visual").pp("vision_model");
        let embeddings = SiglipEmbeddings::load(&vision_vb.pp("embeddings"), cfg)?;
        let encoder = SiglipEncoder::load(
            &vision_vb.pp("encoder"),
            cfg,
            compute_dtype,
            snapshot_hits,
            snapshot_label,
        )?;
        let post_layernorm = PreciseLayerNorm::load(
            vision_vb.pp("post_layernorm"),
            cfg.hidden_size,
            cfg.layer_norm_eps as f64,
            compute_dtype,
        )?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
            compute_dtype,
        })
    }

    pub fn forward(
        &self,
        patches: &SiglipImagePatches,
        use_rope: bool,
        interpolate_pos_encoding: bool,
        device: &Device,
    ) -> Result<Tensor> {
        let embeddings = self.embeddings.forward(
            &patches.patches,
            patches.grid_thw,
            &patches.position_ids,
            interpolate_pos_encoding,
            device,
        )?;
        let compute = embeddings.to_dtype(self.compute_dtype)?;
        let hidden = self.encoder.forward(
            &compute,
            patches.grid_thw,
            &patches.height_ids,
            &patches.width_ids,
            use_rope,
            device,
        )?;
        let normalized = self.post_layernorm.forward(&hidden)?;
        Ok(normalized.to_dtype(patches.patches.dtype())?)
    }

    pub fn forward_with_states(
        &self,
        patches: &SiglipImagePatches,
        use_rope: bool,
        interpolate_pos_encoding: bool,
        device: &Device,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let embeddings = self.embeddings.forward(
            &patches.patches,
            patches.grid_thw,
            &patches.position_ids,
            interpolate_pos_encoding,
            device,
        )?;
        let compute = embeddings.to_dtype(self.compute_dtype)?;
        let (hidden, states) = self.encoder.forward_with_states(
            &compute,
            patches.grid_thw,
            &patches.height_ids,
            &patches.width_ids,
            use_rope,
            device,
        )?;
        let normalized = self
            .post_layernorm
            .forward(&hidden)?
            .to_dtype(patches.patches.dtype())?;
        let squeezed = Self::squeeze_states(states, patches.patches.dtype())?;
        Ok((normalized, squeezed))
    }

    #[cfg(test)]
    pub(crate) fn debug_layer_outputs(
        &self,
        input: &Tensor,
        patches: &SiglipImagePatches,
        layer_index: usize,
        use_rope: bool,
        device: &Device,
    ) -> Result<LayerDebug> {
        ensure!(
            layer_index < self.encoder.layers.len(),
            "layer index {} out of bounds",
            layer_index
        );
        let rope = if use_rope {
            Some(
                self.encoder
                    .build_rotary(&patches.height_ids, &patches.width_ids, device)?,
            )
        } else {
            None
        };
        let compute_input = input.to_dtype(self.compute_dtype)?;
        let layer = &self.encoder.layers[layer_index];
        let norm1 = layer.layer_norm1.forward(&compute_input)?;
        let attn_out = layer.attention.forward(&norm1, rope.as_ref())?;
        let after_attn = compute_input.add(&attn_out)?;
        let norm2 = layer.layer_norm2.forward(&after_attn)?;
        let mlp_out = layer.mlp.forward(&norm2)?;
        let output = after_attn.add(&mlp_out)?;
        let target_dtype = input.dtype();
        Ok(LayerDebug {
            norm1: norm1.to_dtype(target_dtype)?,
            attn_out: attn_out.to_dtype(target_dtype)?,
            after_attn: after_attn.to_dtype(target_dtype)?,
            norm2: norm2.to_dtype(target_dtype)?,
            mlp_out: mlp_out.to_dtype(target_dtype)?,
            output: output.to_dtype(target_dtype)?,
        })
    }

    pub fn encode_hidden_with_states(
        &self,
        hidden: &Tensor,
        patches: &SiglipImagePatches,
        use_rope: bool,
        device: &Device,
    ) -> Result<Vec<Tensor>> {
        let compute = hidden.to_dtype(self.compute_dtype)?;
        let (_, states) = self.encoder.forward_with_states(
            &compute,
            patches.grid_thw,
            &patches.height_ids,
            &patches.width_ids,
            use_rope,
            device,
        )?;
        Self::squeeze_states(states, hidden.dtype())
    }

    pub fn debug_positional_encoding(
        &self,
        grid: (usize, usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        self.embeddings.interpolate_pos_encoding(grid, device)
    }

    fn squeeze_states(mut states: Vec<Tensor>, target_dtype: DType) -> Result<Vec<Tensor>> {
        for state in &mut states {
            let (batch, tokens, hidden) = state
                .shape()
                .dims3()
                .context("vision encoder states must be [batch, tokens, hidden]")?;
            ensure!(
                batch == 1,
                "vision encoder debug states expect batch size 1 (got {batch})"
            );
            *state = state.reshape((tokens, hidden))?.to_dtype(target_dtype)?;
        }
        Ok(states)
    }
}

fn resolve_compute_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    }
}

struct SiglipEmbeddings {
    patch_linear: Linear,
    packing_position_embedding: Tensor,
    position_embedding: Tensor,
    embed_dim: usize,
    patch_size: usize,
    base_grid: usize,
    channels: usize,
    compute_dtype: DType,
}

impl SiglipEmbeddings {
    fn load(vb: &VarBuilder, cfg: &PaddleOcrVisionConfig) -> Result<Self> {
        let patch_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size;
        let patch_vb = vb.pp("patch_embedding");
        let weight = patch_vb.get(
            (
                cfg.hidden_size,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size,
            ),
            "weight",
        )?;
        let bias = patch_vb.get(cfg.hidden_size, "bias")?;
        let compute_dtype = match weight.dtype() {
            DType::F16 | DType::BF16 => DType::F32,
            DType::F32 | DType::F64 => weight.dtype(),
            other => anyhow::bail!("unsupported patch weight dtype {:?}", other),
        };
        if std::env::var("SIGLIP_DEBUG_PATCH_DTYPE").is_ok() {
            eprintln!(
                "siglip patch weight dtype {:?} -> compute {:?}",
                weight.dtype(),
                compute_dtype
            );
        }
        let weight = if weight.dtype() != compute_dtype {
            weight.to_dtype(compute_dtype)?
        } else {
            weight
        };
        let bias = if bias.dtype() != compute_dtype {
            bias.to_dtype(compute_dtype)?
        } else {
            bias
        };
        let reshaped = weight.reshape((cfg.hidden_size, patch_dim))?;
        let patch_linear = Linear::new(reshaped, Some(bias));
        let packing_position_embedding = vb
            .pp("packing_position_embedding")
            .get((32768, cfg.hidden_size), "weight")
            .context("missing packing_position_embedding.weight")?;
        let base_grid = cfg.image_size / cfg.patch_size;
        let num_positions = base_grid * base_grid;
        let position_embedding = vb
            .pp("position_embedding")
            .get((num_positions, cfg.hidden_size), "weight")
            .context("missing position_embedding.weight")?;
        Ok(Self {
            patch_linear,
            packing_position_embedding,
            position_embedding,
            embed_dim: cfg.hidden_size,
            patch_size: cfg.patch_size,
            base_grid,
            channels: cfg.num_channels,
            compute_dtype,
        })
    }

    fn forward(
        &self,
        patches: &Tensor,
        grid: (usize, usize, usize),
        position_ids: &[i64],
        interpolate_pos_encoding: bool,
        device: &Device,
    ) -> Result<Tensor> {
        let (num_patches, channels, height, width) = patches
            .shape()
            .dims4()
            .context("siglip embeddings expect patches with shape [num_patches, C, H, W]")?;
        ensure!(channels == self.channels, "unexpected channel count");
        ensure!(
            height == self.patch_size && width == self.patch_size,
            "patch dimensions ({height}, {width}) mismatch expected patch size {}",
            self.patch_size
        );
        let patch_dim = self.channels * self.patch_size * self.patch_size;
        let flattened = patches
            .reshape((num_patches, patch_dim))?
            .to_dtype(self.compute_dtype)?;
        let embeddings = self.patch_linear.forward(&flattened)?;

        let embeddings = if interpolate_pos_encoding {
            let pos = self
                .interpolate_pos_encoding(grid, device)?
                .to_dtype(embeddings.dtype())?;
            embeddings.add(&pos)?
        } else {
            let pos = self
                .gather_packing_embeddings(position_ids, num_patches, device)?
                .to_dtype(embeddings.dtype())?;
            embeddings.add(&pos)?
        };

        let (t, h, w) = grid;
        ensure!(
            t * h * w == num_patches,
            "grid {:?} incompatible with {} patches",
            grid,
            num_patches
        );
        let output = embeddings
            .reshape((1, num_patches, self.embed_dim))?
            .contiguous()?;
        Ok(output)
    }

    fn gather_packing_embeddings(
        &self,
        ids: &[i64],
        expected: usize,
        device: &Device,
    ) -> Result<Tensor> {
        ensure!(
            ids.len() == expected,
            "packing position ids length {} mismatch expected {}",
            ids.len(),
            expected
        );
        let ids_tensor =
            Tensor::from_vec(ids.to_vec(), (expected, 1), device)?.to_dtype(DType::I64)?;
        let gathered = gather_token_embeddings(&self.packing_position_embedding, &ids_tensor)?;
        Ok(gathered.reshape((expected, self.embed_dim))?)
    }

    fn interpolate_pos_encoding(
        &self,
        grid: (usize, usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        let (_, h, w) = grid;
        let base = self.position_embedding.to_dtype(DType::F32)?.reshape((
            self.base_grid,
            self.base_grid,
            self.embed_dim,
        ))?;
        let nested = base.to_vec3::<f32>()?;
        let resized = resize_positional_grid(&nested, self.embed_dim, h, w);
        let mut spatial = Tensor::from_vec(resized, (h * w, self.embed_dim), device)?
            .to_dtype(self.position_embedding.dtype())?;
        if grid.0 > 1 {
            let mut copies = Vec::with_capacity(grid.0);
            for _ in 0..grid.0 {
                copies.push(spatial.clone());
            }
            let refs: Vec<&Tensor> = copies.iter().collect();
            spatial = Tensor::cat(&refs, 0)?;
        }
        Ok(spatial)
    }
}

struct SiglipEncoder {
    layers: Vec<SiglipEncoderLayer>,
    rotary: SiglipRotaryEmbedding,
}

impl SiglipEncoder {
    fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVisionConfig,
        compute_dtype: DType,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut snapshot_hits = snapshot_hits;
        for idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            layers.push(SiglipEncoderLayer::load(
                &layer_vb,
                cfg,
                compute_dtype,
                snapshot_hits.as_deref_mut(),
                snapshot_label,
            )?);
        }
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary = SiglipRotaryEmbedding::new(head_dim, compute_dtype);
        Ok(Self { layers, rotary })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        _grid: (usize, usize, usize),
        height_ids: &[i64],
        width_ids: &[i64],
        use_rope: bool,
        device: &Device,
    ) -> Result<Tensor> {
        self.forward_internal(hidden, height_ids, width_ids, use_rope, device, None)
    }

    fn forward_with_states(
        &self,
        hidden: &Tensor,
        _grid: (usize, usize, usize),
        height_ids: &[i64],
        width_ids: &[i64],
        use_rope: bool,
        device: &Device,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut captured = Vec::with_capacity(self.layers.len() + 1);
        let hidden = self.forward_internal(
            hidden,
            height_ids,
            width_ids,
            use_rope,
            device,
            Some(&mut captured),
        )?;
        Ok((hidden, captured))
    }

    fn forward_internal(
        &self,
        hidden: &Tensor,
        height_ids: &[i64],
        width_ids: &[i64],
        use_rope: bool,
        device: &Device,
        capture: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        ensure!(
            height_ids.len() == width_ids.len(),
            "height/width position id length mismatch"
        );
        let rope = if use_rope {
            Some(self.build_rotary(height_ids, width_ids, device)?)
        } else {
            None
        };
        let mut state = hidden.clone();
        if let Some(store) = capture {
            store.push(state.clone());
            for layer in &self.layers {
                state = layer.forward(&state, rope.as_ref())?;
                store.push(state.clone());
            }
        } else {
            for layer in &self.layers {
                state = layer.forward(&state, rope.as_ref())?;
            }
        }
        Ok(state)
    }

    fn build_rotary(
        &self,
        height_ids: &[i64],
        width_ids: &[i64],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let max_height = height_ids.iter().copied().max().unwrap_or(0);
        let max_width = width_ids.iter().copied().max().unwrap_or(0);
        let table = self
            .rotary
            .table((max_height.max(max_width) + 1) as usize, device)?;
        let height = gather_rows(&table, height_ids, device)?;
        let width = gather_rows(&table, width_ids, device)?;
        let stacked = Tensor::stack(&[&height, &width], 1)?;
        let inv_dim = self.rotary.freq_len();
        let seq_len = height_ids.len();
        let flattened = stacked.reshape((seq_len, 2 * inv_dim))?;
        let repeated = Tensor::cat(&[&flattened, &flattened], D::Minus1)?;
        let cos = repeated.cos()?;
        let sin = repeated.sin()?;
        Ok((cos, sin))
    }
}

struct SiglipRotaryEmbedding {
    inv_freq: Vec<f32>,
    dtype: DType,
}

impl SiglipRotaryEmbedding {
    fn new(head_dim: usize, dtype: DType) -> Self {
        let base_dim = (head_dim / 2).max(1);
        let mut inv_freq = Vec::new();
        let mut idx = 0usize;
        while idx < base_dim {
            let exponent = idx as f32 / base_dim as f32;
            let value = 1f32 / 10000f32.powf(exponent);
            inv_freq.push(value);
            idx += 2;
        }
        if inv_freq.is_empty() {
            inv_freq.push(1.0);
        }
        Self { inv_freq, dtype }
    }

    fn table(&self, size: usize, device: &Device) -> Result<Tensor> {
        let cols = self.inv_freq.len();
        let mut data: Vec<f32> = Vec::with_capacity(size * cols);
        for pos in 0..size {
            let pos_val = pos as f32;
            for &freq in &self.inv_freq {
                data.push(pos_val * freq);
            }
        }
        let table = Tensor::from_vec(data, (size, cols), device)?;
        if table.dtype() == self.dtype {
            Ok(table)
        } else {
            Ok(table.to_dtype(self.dtype)?)
        }
    }

    fn freq_len(&self) -> usize {
        self.inv_freq.len()
    }
}

struct SiglipEncoderLayer {
    layer_norm1: PreciseLayerNorm,
    layer_norm2: PreciseLayerNorm,
    attention: SiglipAttention,
    mlp: SiglipMlp,
}

impl SiglipEncoderLayer {
    fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVisionConfig,
        compute_dtype: DType,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let layer_norm1 = PreciseLayerNorm::load(
            vb.pp("layer_norm1"),
            cfg.hidden_size,
            cfg.layer_norm_eps as f64,
            compute_dtype,
        )?;
        let mut snapshot_hits = snapshot_hits;
        let attention = SiglipAttention::load(
            &vb.pp("self_attn"),
            cfg,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let layer_norm2 = PreciseLayerNorm::load(
            vb.pp("layer_norm2"),
            cfg.hidden_size,
            cfg.layer_norm_eps as f64,
            compute_dtype,
        )?;
        let mlp = SiglipMlp::load(
            &vb.pp("mlp"),
            cfg,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        Ok(Self {
            layer_norm1,
            layer_norm2,
            attention,
            mlp,
        })
    }

    fn forward(&self, hidden: &Tensor, rope: Option<&(Tensor, Tensor)>) -> Result<Tensor> {
        let normed = self.layer_norm1.forward(hidden)?;
        let attn_output = self.attention.forward(&normed, rope)?;
        let residual = hidden.add(&attn_output)?;
        let normed = self.layer_norm2.forward(&residual)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(residual.add(&mlp_out)?)
    }
}

struct SiglipAttention {
    q_proj: VisionLinear,
    k_proj: VisionLinear,
    v_proj: VisionLinear,
    out_proj: VisionLinear,
    num_heads: usize,
    head_dim: usize,
    compute_dtype: DType,
}

impl SiglipAttention {
    fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVisionConfig,
        compute_dtype: DType,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let mut snapshot_hits = snapshot_hits;
        let q_proj = VisionLinear::load(
            vb.pp("q_proj"),
            cfg.hidden_size,
            cfg.hidden_size,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let k_proj = VisionLinear::load(
            vb.pp("k_proj"),
            cfg.hidden_size,
            cfg.hidden_size,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let v_proj = VisionLinear::load(
            vb.pp("v_proj"),
            cfg.hidden_size,
            cfg.hidden_size,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let out_proj = VisionLinear::load(
            vb.pp("out_proj"),
            cfg.hidden_size,
            cfg.hidden_size,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim,
            compute_dtype,
        })
    }

    fn forward(&self, hidden: &Tensor, rope: Option<&(Tensor, Tensor)>) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden
            .shape()
            .dims3()
            .context("attention expects [batch, seq, hidden]")?;
        ensure!(
            hidden_size == self.head_dim * self.num_heads,
            "hidden size {hidden_size} incompatible with heads {} and head_dim {}",
            self.num_heads,
            self.head_dim
        );
        let q = self
            .q_proj
            .forward(hidden)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(hidden)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(hidden)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = if let Some((cos, sin)) = rope {
            let cos = cos
                .reshape((1, 1, seq_len, self.head_dim))?
                .to_dtype(q.dtype())?
                .expand((batch, self.num_heads, seq_len, self.head_dim))?;
            let sin = sin
                .reshape((1, 1, seq_len, self.head_dim))?
                .to_dtype(q.dtype())?
                .expand((batch, self.num_heads, seq_len, self.head_dim))?;
            let q_rot = q.mul(&cos)?.add(&rotate_half_local(&q)?.mul(&sin)?)?;
            let k_rot = k.mul(&cos)?.add(&rotate_half_local(&k)?.mul(&sin)?)?;
            (q_rot, k_rot)
        } else {
            (q, k)
        };

        let q = q
            .contiguous()?
            .reshape((batch * self.num_heads, seq_len, self.head_dim))?;
        let k = k
            .contiguous()?
            .reshape((batch * self.num_heads, seq_len, self.head_dim))?;
        let v = v
            .contiguous()?
            .reshape((batch * self.num_heads, seq_len, self.head_dim))?;

        let scale = (self.head_dim as f64).sqrt() as f32;
        let q = q.to_dtype(self.compute_dtype)?;
        let k = k.to_dtype(self.compute_dtype)?;
        let v = v.to_dtype(self.compute_dtype)?;

        let attn_scores = q.matmul(&k.transpose(1, 2)?)?;
        let scaling =
            Tensor::full(scale, (), attn_scores.device())?.to_dtype(attn_scores.dtype())?;
        let attn_scores = attn_scores.broadcast_div(&scaling)?;
        let attn_probs = softmax(&attn_scores, D::Minus1).context("attention softmax failed")?;
        let context =
            attn_probs
                .matmul(&v)?
                .reshape((batch, self.num_heads, seq_len, self.head_dim))?;
        let context = context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, hidden_size))?
            .to_dtype(hidden.dtype())?;
        Ok(self.out_proj.forward(&context)?)
    }
}

struct SiglipMlp {
    fc1: VisionLinear,
    fc2: VisionLinear,
}

impl SiglipMlp {
    fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVisionConfig,
        compute_dtype: DType,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let mut snapshot_hits = snapshot_hits;
        let fc1 = VisionLinear::load(
            vb.pp("fc1"),
            cfg.intermediate_size,
            cfg.hidden_size,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let fc2 = VisionLinear::load(
            vb.pp("fc2"),
            cfg.hidden_size,
            cfg.intermediate_size,
            compute_dtype,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = gelu_pytorch_tanh(&self.fc1.forward(input)?)?;
        Ok(self.fc2.forward(&hidden)?)
    }
}

fn gelu_pytorch_tanh(input: &Tensor) -> Result<Tensor> {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape().clone();
    let coeff = Tensor::full(SQRT_2_OVER_PI, shape.clone(), device)?.to_dtype(dtype)?;
    let kappa = Tensor::full(0.044715f32, shape.clone(), device)?.to_dtype(dtype)?;
    let x_cubed = input.mul(input)?.mul(input)?;
    let inner = input.add(&x_cubed.mul(&kappa)?)?;
    let tanh_arg = inner.mul(&coeff)?;
    let tanh = tanh_arg.tanh()?;
    let ones = Tensor::full(1f32, tanh.shape().clone(), device)?.to_dtype(dtype)?;
    let halves = Tensor::full(0.5f32, tanh.shape().clone(), device)?.to_dtype(dtype)?;
    let term = tanh.add(&ones)?.mul(&halves)?;
    Ok(term.mul(input)?)
}

fn resize_positional_grid(
    base: &[Vec<Vec<f32>>],
    embed_dim: usize,
    target_h: usize,
    target_w: usize,
) -> Vec<f32> {
    let base_h = base.len().max(1);
    let base_w = base.first().map(|row| row.len()).unwrap_or(1).max(1);
    if target_h == base_h && target_w == base_w {
        return base
            .iter()
            .flat_map(|row| row.iter().flat_map(|col| col.iter().copied()))
            .collect();
    }
    let mut temp = vec![0f32; target_h * base_w * embed_dim];
    for ty in 0..target_h {
        let (y0, y1, y_lerp) = lerp_coord(ty, target_h, base_h);
        for x in 0..base_w {
            for d in 0..embed_dim {
                let v0 = base[y0][x][d];
                let v1 = base[y1][x][d];
                temp[(ty * base_w + x) * embed_dim + d] = v0 + (v1 - v0) * y_lerp;
            }
        }
    }
    if target_w == base_w {
        return temp;
    }
    let mut output = vec![0f32; target_h * target_w * embed_dim];
    for ty in 0..target_h {
        for tx in 0..target_w {
            let (x0, x1, x_lerp) = lerp_coord(tx, target_w, base_w);
            for d in 0..embed_dim {
                let v0 = temp[(ty * base_w + x0) * embed_dim + d];
                let v1 = temp[(ty * base_w + x1) * embed_dim + d];
                output[(ty * target_w + tx) * embed_dim + d] = v0 + (v1 - v0) * x_lerp;
            }
        }
    }
    output
}

fn lerp_coord(idx: usize, target: usize, base: usize) -> (usize, usize, f32) {
    if base <= 1 {
        return (0, 0, 0.0);
    }
    if target <= 1 {
        return (0, 0, 0.0);
    }
    let scale = base as f32 / target as f32;
    let real = (idx as f32 + 0.5) * scale - 0.5;
    let clamped = real.clamp(0.0, (base - 1) as f32);
    let low = clamped.floor() as usize;
    let high = (low + 1).min(base - 1);
    let frac = clamped - low as f32;
    (low, high, frac)
}

fn gather_rows(table: &Tensor, ids: &[i64], device: &Device) -> Result<Tensor> {
    if ids.is_empty() {
        return Ok(Tensor::zeros(
            (0, table.shape().dims2()?.1),
            table.dtype(),
            device,
        )?);
    }
    let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), device)?.to_dtype(DType::I64)?;
    Ok(table.index_select(&idx, 0)?)
}

#[cfg(test)]
pub(crate) struct LayerDebug {
    pub norm1: Tensor,
    pub attn_out: Tensor,
    pub after_attn: Tensor,
    pub norm2: Tensor,
    pub mlp_out: Tensor,
    pub output: Tensor,
}

struct VisionLinear {
    weights: LinearWeights,
    compute_dtype: DType,
}

impl VisionLinear {
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
        let dims = input.shape().dims().to_vec();
        ensure!(!dims.is_empty(), "vision linear expects rank >= 1");
        let last = *dims.last().expect("non-empty dims");
        ensure!(
            last == self.weights.in_dim,
            "vision linear expected last dim {} got {}",
            self.weights.in_dim,
            last
        );
        let outer: usize = if dims.len() == 1 {
            1
        } else {
            dims[..dims.len() - 1].iter().product()
        };
        let cast = if input.dtype() == self.compute_dtype {
            input.clone()
        } else {
            input.to_dtype(self.compute_dtype)?
        };
        let reshaped = cast.reshape((outer, self.weights.in_dim))?;
        let mut out = self.weights.matmul_2d(&reshaped)?;
        if let Some(bias) = &self.weights.bias {
            let bias = if bias.dtype() == self.compute_dtype {
                bias.clone()
            } else {
                bias.to_dtype(self.compute_dtype)?
            };
            out = out.broadcast_add(&bias.reshape((1, self.weights.out_dim))?)?;
        }
        let mut new_dims = dims;
        let last_idx = new_dims.len() - 1;
        new_dims[last_idx] = self.weights.out_dim;
        let mut restored = out.reshape(new_dims)?;
        if restored.dtype() != input.dtype() {
            restored = restored.to_dtype(input.dtype())?;
        }
        Ok(restored)
    }
}

struct PreciseLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    compute_dtype: DType,
}

impl PreciseLayerNorm {
    fn load(vb: VarBuilder, size: usize, eps: f64, compute_dtype: DType) -> Result<Self> {
        let weight = vb
            .get(size, "weight")
            .context("missing layernorm weight")?
            .to_dtype(compute_dtype)?;
        let bias = vb
            .get(size, "bias")
            .context("missing layernorm bias")?
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
}

fn rotate_half_local(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    ensure!(!dims.is_empty(), "rotate_half requires tensor rank >= 1");
    let last = *dims.last().expect("checked non-empty");
    ensure!(last % 2 == 0, "rotate_half expects even last dimension");
    let half = last / 2;
    let first = tensor.narrow(D::Minus1, 0, half)?;
    let second = tensor.narrow(D::Minus1, half, half)?;
    let neg_second = second.neg()?;
    Ok(Tensor::cat(&[&neg_second, &first], D::Minus1)?)
}
