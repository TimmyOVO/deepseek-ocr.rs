use crate::{
    config::DeepseekV2Config,
    quantization::run_quantized_matmul,
    transformer::{
        cache::{KvCacheChunk, KvCacheEntry},
        weights::{
            AttentionWeights, DenseMlpWeights, LinearWeights, MlpWeights, MoeWeights,
            TransformerBlockWeights,
        },
    },
};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor, shape::D};
#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;
use candle_nn::ops::{rms_norm_slow, sigmoid, softmax};

fn is_low_precision(t: &Tensor) -> bool {
    matches!(t.dtype(), DType::F16 | DType::BF16)
}
/// Low-precision (f16/bf16) can accumulate enough numeric error in sensitive
/// reductions to flip greedy argmax in near-tie steps. Keep these ops in f32,
/// then cast back to the working dtype.
fn rms_norm_stable(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // Use f32 accumulation for RMSNorm to reduce drift in low precision.
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?.contiguous()?;
    rms_norm_slow(&x_f32, &w_f32, eps).context("rms_norm f32 failed")
}

fn add_stable(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if matches!(a.dtype(), DType::F16 | DType::BF16) {
        let a_f32 = a.to_dtype(DType::F32)?;
        let b_f32 = b.to_dtype(DType::F32)?;
        Ok(a_f32
            .add(&b_f32)?
            .to_dtype(a.dtype())?
            .to_device(a.device())?)
    } else {
        Ok(a.add(b)?)
    }
}

/// Candle implementation of a single DeepSeek transformer decoder block (non-flash path).
///
/// This version supports dense MLP layers. Routed MoE layers return a `bail!` placeholder for now.
pub struct TransformerBlock<'a> {
    pub cfg: &'a DeepseekV2Config,
    pub weights: &'a TransformerBlockWeights,
    use_flash_attention: bool,
}

pub struct BlockOutput {
    pub hidden_states: Tensor,
    pub present_key_value: Option<KvCacheChunk>,
    pub aux_loss: Option<Tensor>,
}

struct MlpForwardOutput {
    hidden_states: Tensor,
    aux_loss: Option<Tensor>,
}

impl<'a> TransformerBlock<'a> {
    pub fn new(
        cfg: &'a DeepseekV2Config,
        weights: &'a TransformerBlockWeights,
        use_flash_attention: bool,
    ) -> Self {
        Self {
            cfg,
            weights,
            use_flash_attention,
        }
    }

    pub fn forward(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        additive_attn_bias: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
        past_key_value: Option<&KvCacheEntry>,
        use_cache: bool,
    ) -> Result<BlockOutput> {
        if is_low_precision(hidden_states) {
            let (_, seq_len, _) = hidden_states.shape().dims3()?;
            if seq_len == 1 {
                // Decode steps are the most sensitive to low-precision drift.
                // Keep the block in f32 and carry f32 residuals through decode.
                return self
                    .forward_internal_f32(
                        layer_idx,
                        hidden_states,
                        additive_attn_bias,
                        rope,
                        past_key_value,
                        use_cache,
                    )
                    .context("block forward (low precision decode f32) failed");
            }
            return self
                .forward_internal(
                    layer_idx,
                    hidden_states,
                    additive_attn_bias,
                    rope,
                    past_key_value,
                    use_cache,
                )
                .context("block forward (low precision) failed");
        }

        self.forward_internal(
            layer_idx,
            hidden_states,
            additive_attn_bias,
            rope,
            past_key_value,
            use_cache,
        )
    }

    fn forward_internal(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        additive_attn_bias: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
        past_key_value: Option<&KvCacheEntry>,
        use_cache: bool,
    ) -> Result<BlockOutput> {
        let residual = hidden_states;
        let normed = rms_norm_stable(
            residual,
            &self.weights.input_layernorm.weight,
            self.cfg.rms_norm_eps,
        )?;

        let (attn_out, present_cache) = attention_forward(
            &normed,
            &self.weights.attention,
            self.cfg,
            AttentionForwardOptions {
                additive_attn_bias,
                rope,
                past_key_value,
                use_cache,
                use_flash_attention: self.use_flash_attention,
            },
        )
        .context("attention forward failed")?;
        let hidden_states = add_stable(residual, &attn_out).context("residual add (attention)")?;

        let residual = &hidden_states;
        let normed = rms_norm_stable(
            residual,
            &self.weights.post_attention_layernorm.weight,
            self.cfg.rms_norm_eps,
        )?;
        let (mlp_hidden, aux_loss) = if is_low_precision(residual) {
            match &self.weights.mlp {
                MlpWeights::Dense(dense) => (
                    run_dense_mlp_f32_keep(&normed.to_dtype(DType::F32)?, dense, self.cfg)?
                        .to_dtype(residual.dtype())?,
                    None,
                ),
                MlpWeights::Moe(moe) => (
                    run_moe(layer_idx, &normed.to_dtype(DType::F32)?, moe, self.cfg)?
                        .hidden_states
                        .to_dtype(DType::F32)?,
                    None,
                ),
            }
        } else {
            let MlpForwardOutput {
                hidden_states,
                aux_loss,
            } = mlp_forward(layer_idx, &normed, &self.weights.mlp, self.cfg)
                .context("mlp forward failed")?;
            (hidden_states, aux_loss)
        };

        let output = add_stable(residual, &mlp_hidden).context("residual add (mlp)")?;
        let present = if use_cache { present_cache } else { None };
        Ok(BlockOutput {
            hidden_states: output,
            present_key_value: present,
            aux_loss,
        })
    }

    fn forward_internal_f32(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        additive_attn_bias: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
        past_key_value: Option<&KvCacheEntry>,
        use_cache: bool,
    ) -> Result<BlockOutput> {
        let residual_f32 = hidden_states.to_dtype(DType::F32)?;
        let w_in = self
            .weights
            .input_layernorm
            .weight
            .to_dtype(DType::F32)?
            .contiguous()?;
        let normed_f32 = rms_norm_slow(&residual_f32, &w_in, self.cfg.rms_norm_eps)
            .context("input rms norm failed")?;

        let (attn_out, mut present_cache) = attention_forward_f32_keep(
            &normed_f32,
            &self.weights.attention,
            self.cfg,
            additive_attn_bias,
            rope,
            past_key_value,
            use_cache,
            hidden_states.dtype(),
        )
        .context("attention forward f32 keep failed")?;
        if let Some(chunk) = present_cache.as_mut() {
            chunk.key_t = chunk.key_t.contiguous()?;
            chunk.value = chunk.value.contiguous()?;
        }

        let post_attn = residual_f32
            .add(&attn_out)
            .context("residual add (attention)")?;

        let w_post = self
            .weights
            .post_attention_layernorm
            .weight
            .to_dtype(DType::F32)?
            .contiguous()?;
        let normed2 = rms_norm_slow(&post_attn, &w_post, self.cfg.rms_norm_eps)
            .context("post-attention rms norm failed")?;

        let (mlp_out_f32, aux_loss) = match &self.weights.mlp {
            MlpWeights::Dense(dense) => (run_dense_mlp_f32_keep(&normed2, dense, self.cfg)?, None),
            MlpWeights::Moe(moe) => {
                let out = run_moe(layer_idx, &normed2, moe, self.cfg)?
                    .hidden_states
                    .to_dtype(DType::F32)?;
                (out, None)
            }
        };

        let out_f32 = post_attn.add(&mlp_out_f32).context("residual add (mlp)")?;

        let (_, seq_len, _) = hidden_states.shape().dims3()?;
        let keep_f32_out = seq_len == 1;
        let out = if keep_f32_out {
            out_f32
        } else {
            out_f32.to_dtype(hidden_states.dtype())?
        };
        Ok(BlockOutput {
            hidden_states: out,
            present_key_value: if use_cache { present_cache } else { None },
            aux_loss,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_forward_f32_keep(
    hidden_states: &Tensor,
    weights: &AttentionWeights,
    cfg: &DeepseekV2Config,
    additive_attn_bias: Option<&Tensor>,
    rope: Option<(&Tensor, &Tensor)>,
    past_key_value: Option<&KvCacheEntry>,
    use_cache: bool,
    _cache_dtype: DType,
) -> Result<(Tensor, Option<KvCacheChunk>)> {
    let device = hidden_states.device();
    let (batch, seq_len, hidden_size) = hidden_states.shape().dims3()?;
    let head_dim = hidden_size / cfg.num_attention_heads;
    let v_head_dim_cfg = cfg.v_head_dim.unwrap_or(head_dim);
    let v_head_dim = if v_head_dim_cfg == 0 {
        head_dim
    } else {
        v_head_dim_cfg
    };
    let num_kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
    let kv_head_dim_cfg = cfg.qk_nope_head_dim.unwrap_or(head_dim);
    let kv_head_dim = if kv_head_dim_cfg == 0 {
        head_dim
    } else {
        kv_head_dim_cfg
    };

    let q = apply_linear_f32_keep(hidden_states, &weights.q_proj)?
        .reshape((batch, seq_len, cfg.num_attention_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let k = apply_linear_f32_keep(hidden_states, &weights.k_proj)?
        .reshape((batch, seq_len, num_kv_heads, kv_head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = apply_linear_f32_keep(hidden_states, &weights.v_proj)?
        .reshape((batch, seq_len, num_kv_heads, v_head_dim))?
        .transpose(1, 2)?
        .contiguous()?;

    let mut q = q;
    let mut k = k;
    if let Some((cos, sin)) = rope {
        let rope_dim_cfg = cfg.qk_rope_head_dim.unwrap_or(head_dim);
        let rope_dim = if rope_dim_cfg == 0 {
            head_dim
        } else {
            rope_dim_cfg
        };
        if rope_dim > 0 {
            let q_rot = q.narrow(D::Minus1, 0, rope_dim)?;
            let q_tail = if rope_dim < head_dim {
                Some(q.narrow(D::Minus1, rope_dim, head_dim - rope_dim)?)
            } else {
                None
            };
            let k_rot = k.narrow(D::Minus1, 0, rope_dim)?;
            let k_tail = if rope_dim < kv_head_dim {
                Some(k.narrow(D::Minus1, rope_dim, kv_head_dim - rope_dim)?)
            } else {
                None
            };
            let q_rot = apply_rope(
                &q_rot,
                &cos.to_dtype(DType::F32)?,
                &sin.to_dtype(DType::F32)?,
                cfg.use_mla,
            )?;
            let k_rot = apply_rope(
                &k_rot,
                &cos.to_dtype(DType::F32)?,
                &sin.to_dtype(DType::F32)?,
                cfg.use_mla,
            )?;
            q = if let Some(tail) = q_tail {
                Tensor::cat(&[q_rot, tail], D::Minus1)?
            } else {
                q_rot
            };
            k = if let Some(tail) = k_tail {
                Tensor::cat(&[k_rot, tail], D::Minus1)?
            } else {
                k_rot
            };
            q = q.contiguous()?;
            k = k.contiguous()?;
        }
    }

    ensure!(cfg.num_attention_heads.is_multiple_of(num_kv_heads));
    let repeats = cfg.num_attention_heads / num_kv_heads;
    let k_new = repeat_kv(&k, repeats)?.contiguous()?;
    let v_new = repeat_kv(&v, repeats)?.contiguous()?;

    let mut cache_key_t_view: Option<Tensor> = None;
    let mut cache_value_view: Option<Tensor> = None;
    let past_len = if let Some(cache) = past_key_value {
        let past_len = cache.seq_len();
        if past_len > 0 {
            let key_view = cache.key_view()?.contiguous()?;
            let value_view = cache.value_view()?.contiguous()?;
            cache_key_t_view = Some(key_view.to_dtype(DType::F32)?);
            cache_value_view = Some(value_view.to_dtype(DType::F32)?);
        }
        past_len
    } else {
        0
    };

    let k_new_t = transpose(&k_new, 2, 3)?;

    let scores_new = q.matmul(&k_new_t)?;
    let attn_scores_mat = if let Some(cache_key_t) = cache_key_t_view.as_ref() {
        if past_len > 0 {
            let scores_past = q.matmul(cache_key_t)?;
            Tensor::cat(&[scores_past, scores_new], D::Minus1)?
        } else {
            scores_new
        }
    } else {
        scores_new
    };

    let scale = (head_dim as f64).sqrt();
    let mut attn_scores = (attn_scores_mat / scale)?;
    if let Some(bias) = additive_attn_bias {
        let bias = bias.to_dtype(DType::F32)?;
        let bias = bias.broadcast_as(attn_scores.shape().dims())?;
        attn_scores = attn_scores.broadcast_add(&bias)?;
    }
    let attn_weights = softmax(&attn_scores, D::Minus1).context("attention softmax f32 failed")?;

    let out_f32 = if let Some(cache_value) = cache_value_view.as_ref() {
        if past_len > 0 {
            let w_past = attn_weights.narrow(D::Minus1, 0, past_len)?;
            let w_new = attn_weights.narrow(D::Minus1, past_len, seq_len)?;
            let past_out = w_past.matmul(cache_value)?;
            let new_out = w_new.matmul(&v_new)?;
            past_out.add(&new_out)?
        } else {
            attn_weights.matmul(&v_new)?
        }
    } else {
        attn_weights.matmul(&v_new)?
    };

    let present = if use_cache {
        let store_dtype = DType::F32;
        let (k_store, v_store) = if store_dtype == DType::F32 {
            (k_new_t.to_dtype(DType::F32)?, v_new.to_dtype(DType::F32)?)
        } else {
            (k_new_t.to_dtype(store_dtype)?, v_new.to_dtype(store_dtype)?)
        };
        Some(KvCacheChunk::new(k_store, v_store)?)
    } else {
        None
    };

    let attn_output = out_f32.permute((0, 2, 1, 3))?.reshape((
        batch,
        seq_len,
        cfg.num_attention_heads * v_head_dim,
    ))?;

    let out = apply_linear_f32_keep(&attn_output, &weights.o_proj)?;
    let out = out.to_device(device)?;
    Ok((out, present))
}

struct AttentionForwardOptions<'a> {
    additive_attn_bias: Option<&'a Tensor>,
    rope: Option<(&'a Tensor, &'a Tensor)>,
    past_key_value: Option<&'a KvCacheEntry>,
    use_cache: bool,
    use_flash_attention: bool,
}

fn attention_forward(
    hidden_states: &Tensor,
    weights: &AttentionWeights,
    cfg: &DeepseekV2Config,
    options: AttentionForwardOptions<'_>,
) -> Result<(Tensor, Option<KvCacheChunk>)> {
    if cfg.q_lora_rank.is_some() || cfg.kv_lora_rank.is_some() {
        bail!("LoRA attention path not yet implemented");
    }

    if options.use_flash_attention
        && let Some(result) = flash_attention_forward(
            hidden_states,
            weights,
            cfg,
            options.rope,
            options.additive_attn_bias,
            options.past_key_value,
            options.use_cache,
        )?
    {
        return Ok(result);
    }

    let (batch, seq_len, hidden_size) = hidden_states
        .shape()
        .dims3()
        .context("attention expects hidden_states with shape [batch, seq, hidden]")?;
    if hidden_size != cfg.hidden_size {
        bail!(
            "config hidden_size {} does not match tensor hidden dim {}",
            cfg.hidden_size,
            hidden_size
        );
    }

    let head_dim = hidden_size / cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
    let kv_head_dim = head_dim;
    let v_head_dim = if cfg.v_head_dim.unwrap_or(0) == 0 {
        head_dim
    } else {
        cfg.v_head_dim.unwrap()
    };

    let use_attn_f32 = is_low_precision(hidden_states);
    let use_f32_projection = use_attn_f32 || hidden_states.dtype() == DType::F32;
    let use_cache_f32 = matches!(hidden_states.dtype(), DType::F16 | DType::BF16);

    // Query / key / value projections.
    let mut q = if use_f32_projection {
        apply_linear_f32_keep(hidden_states, &weights.q_proj)?
    } else {
        apply_linear(hidden_states, &weights.q_proj)?
    }
    .reshape((batch, seq_len, cfg.num_attention_heads, head_dim))?;
    let mut k = if use_f32_projection {
        apply_linear_f32_keep(hidden_states, &weights.k_proj)?
    } else {
        apply_linear(hidden_states, &weights.k_proj)?
    }
    .reshape((batch, seq_len, num_kv_heads, kv_head_dim))?;
    let v = if use_f32_projection {
        apply_linear_f32_keep(hidden_states, &weights.v_proj)?
    } else {
        apply_linear(hidden_states, &weights.v_proj)?
    }
    .reshape((batch, seq_len, num_kv_heads, v_head_dim))?;

    q = q.permute((0, 2, 1, 3))?;
    k = k.permute((0, 2, 1, 3))?;
    let v = v.permute((0, 2, 1, 3))?;

    let rope_dim = cfg.qk_rope_head_dim.unwrap_or(head_dim);
    let rope_dim = if rope_dim == 0 { head_dim } else { rope_dim };
    ensure!(
        rope_dim <= head_dim,
        "rope dimension {} exceeds q head dimension {}",
        rope_dim,
        head_dim
    );
    ensure!(
        rope_dim <= kv_head_dim,
        "rope dimension {} exceeds k head dimension {}",
        rope_dim,
        kv_head_dim
    );
    if rope_dim > 0 {
        let (cos, sin) = options.rope.context("missing rope tensors for attention")?;
        ensure!(
            cos.shape().dims() == [batch, 1, seq_len, rope_dim],
            "cos shape {:?} incompatible with (batch={}, seq={}, rope_dim={})",
            cos.shape().dims(),
            batch,
            seq_len,
            rope_dim
        );
        ensure!(
            sin.shape().dims() == [batch, 1, seq_len, rope_dim],
            "sin shape {:?} incompatible with (batch={}, seq={}, rope_dim={})",
            sin.shape().dims(),
            batch,
            seq_len,
            rope_dim
        );
        let cos = if use_f32_projection {
            cos.to_dtype(DType::F32)?
        } else {
            cos.clone()
        };
        let sin = if use_f32_projection {
            sin.to_dtype(DType::F32)?
        } else {
            sin.clone()
        };
        if use_f32_projection {
            q = q.to_dtype(DType::F32)?;
            k = k.to_dtype(DType::F32)?;
        }
        let q_rot = q.narrow(D::Minus1, 0, rope_dim)?;
        let k_rot = k.narrow(D::Minus1, 0, rope_dim)?;
        let q_tail = if rope_dim < head_dim {
            Some(q.narrow(D::Minus1, rope_dim, head_dim - rope_dim)?)
        } else {
            None
        };
        let k_tail = if rope_dim < kv_head_dim {
            Some(k.narrow(D::Minus1, rope_dim, kv_head_dim - rope_dim)?)
        } else {
            None
        };
        let q_rot = apply_rope(&q_rot, &cos, &sin, cfg.use_mla)?;
        let k_rot = apply_rope(&k_rot, &cos, &sin, cfg.use_mla)?;
        q = if let Some(tail) = q_tail {
            Tensor::cat(&[q_rot, tail], D::Minus1)?
        } else {
            q_rot
        };
        k = if let Some(tail) = k_tail {
            Tensor::cat(&[k_rot, tail], D::Minus1)?
        } else {
            k_rot
        };
    }

    ensure!(
        cfg.num_attention_heads.is_multiple_of(num_kv_heads),
        "num_attention_heads {} must be divisible by num_key_value_heads {}",
        cfg.num_attention_heads,
        num_kv_heads
    );
    let repeats = cfg.num_attention_heads / num_kv_heads;
    let mut k_new = repeat_kv(&k, repeats)?;
    let mut v_new = repeat_kv(&v, repeats)?;
    if use_attn_f32 {
        v_new = v_new.to_dtype(DType::F32)?;
    }

    q = q.contiguous()?;
    k_new = k_new.contiguous()?;
    v_new = v_new.contiguous()?;

    let mut cache_key_t_view: Option<Tensor> = None;
    let mut cache_value_view: Option<Tensor> = None;
    let past_len = if let Some(cache) = options.past_key_value {
        let key_view = cache.key_view()?;
        let value_view = cache.value_view()?;
        let (cache_batch, cache_heads, cache_dim, _) = key_view
            .shape()
            .dims4()
            .context("cache key tensor must be 4D")?;
        ensure!(
            cache_batch == batch,
            "cache batch {} does not match current batch {}",
            cache_batch,
            batch
        );
        ensure!(
            cache_heads == cfg.num_attention_heads,
            "cache heads {} does not match attention heads {}",
            cache_heads,
            cfg.num_attention_heads
        );
        ensure!(
            cache_dim == kv_head_dim,
            "cache key head dim {} does not match kv_head_dim {}",
            cache_dim,
            kv_head_dim
        );
        let value_dims = value_view.shape().dims();
        ensure!(
            value_dims[0] == batch,
            "cache value batch {} does not match current batch {}",
            value_dims[0],
            batch
        );
        ensure!(
            value_dims[1] == cfg.num_attention_heads,
            "cache value heads {} does not match attention heads {}",
            value_dims[1],
            cfg.num_attention_heads
        );
        ensure!(
            value_dims[3] == v_head_dim,
            "cache value head dim {} does not match v_head_dim {}",
            value_dims[3],
            v_head_dim
        );
        cache_key_t_view = Some(key_view);
        cache_value_view = Some(value_view);
        cache.seq_len()
    } else {
        0
    };

    let k_new_t = transpose(&k_new, 2, 3)?.contiguous()?;
    let attn_scores_mat = if use_attn_f32 {
        // f16->f32: attention score matmul can be extremely sensitive.
        let q_f32 = q.to_dtype(DType::F32)?;
        let k_new_t_f32 = k_new_t.to_dtype(DType::F32)?;
        if let Some(cache_key_t) = cache_key_t_view.as_ref() {
            let scores_new = q_f32.matmul(&k_new_t_f32)?;
            if past_len > 0 {
                let cache_key_t_f32 = cache_key_t.contiguous()?.to_dtype(DType::F32)?;
                let scores_past = q_f32.matmul(&cache_key_t_f32)?;
                Tensor::cat(&[scores_past, scores_new], D::Minus1)?
            } else {
                scores_new
            }
        } else {
            q_f32.matmul(&k_new_t_f32)?
        }
    } else if let Some(cache_key_t) = cache_key_t_view.as_ref() {
        let k_new_t_f16 = k_new_t.to_dtype(q.dtype())?;
        let scores_new = q.matmul(&k_new_t_f16)?;
        if past_len > 0 {
            let cache_key_t = if q.dtype() == DType::F32 {
                cache_key_t.contiguous()?.to_dtype(DType::F32)?
            } else {
                cache_key_t.contiguous()?.to_dtype(q.dtype())?
            };
            let scores_past = q.matmul(&cache_key_t)?;
            Tensor::cat(&[scores_past, scores_new], D::Minus1)?
        } else {
            scores_new
        }
    } else {
        let k_new_t_f16 = k_new_t.to_dtype(q.dtype())?;
        q.matmul(&k_new_t_f16)?
    };

    let scale = (head_dim as f64).sqrt();
    let mut attn_scores = (attn_scores_mat / scale)?;
    if let Some(bias) = options.additive_attn_bias {
        let bias = if bias.dtype() != attn_scores.dtype() {
            bias.to_dtype(attn_scores.dtype())?
        } else {
            bias.clone()
        };
        let bias = bias.broadcast_as(attn_scores.shape().dims())?;
        attn_scores = attn_scores.broadcast_add(&bias)?;
    }
    if use_attn_f32 {
        // Keep attention scores in f32 before softmax to reduce near-tie drift.
        attn_scores = attn_scores.to_dtype(DType::F32)?;
    }
    let attn_weights = if use_attn_f32 {
        // Keep softmax in f32 for stability; downstream matmul decides whether to cast.
        softmax(&attn_scores.to_dtype(DType::F32)?, D::Minus1)
            .context("attention softmax failed")?
    } else {
        softmax(&attn_scores, D::Minus1).context("attention softmax failed")?
    };

    // Keep (attn_weights @ values) matmul in f32 when we upcasted softmax.
    let attn_output = if attn_weights.dtype() == DType::F32 && use_attn_f32 {
        let v_new_f32 = v_new.to_dtype(DType::F32)?;
        if let Some(cache_value_view) = cache_value_view.as_ref() {
            let accum = if past_len > 0 {
                let cache_value_f32 = cache_value_view.contiguous()?.to_dtype(DType::F32)?;
                Some(
                    attn_weights
                        .narrow(D::Minus1, 0, past_len)?
                        .matmul(&cache_value_f32)?,
                )
            } else {
                None
            };
            let contrib_new = attn_weights
                .narrow(D::Minus1, past_len, seq_len)?
                .matmul(&v_new_f32)?;
            if let Some(existing) = accum {
                existing.add(&contrib_new)?
            } else {
                contrib_new
            }
        } else {
            attn_weights.matmul(&v_new_f32)?
        }
    } else {
        let v_new = if v_new.dtype() == attn_weights.dtype() {
            v_new.clone()
        } else {
            v_new.to_dtype(attn_weights.dtype())?
        };
        if let Some(cache_value_view) = cache_value_view.as_ref() {
            let accum = if past_len > 0 {
                let cache_value = cache_value_view
                    .contiguous()?
                    .to_dtype(attn_weights.dtype())?;
                Some(
                    attn_weights
                        .narrow(D::Minus1, 0, past_len)?
                        .matmul(&cache_value)?,
                )
            } else {
                None
            };
            let contrib_new = attn_weights
                .narrow(D::Minus1, past_len, seq_len)?
                .matmul(&v_new)?;
            if let Some(existing) = accum {
                existing.add(&contrib_new)?
            } else {
                contrib_new
            }
        } else {
            attn_weights.matmul(&v_new)?
        }
    };
    let present = if options.use_cache {
        let store_dtype = if use_cache_f32 || use_attn_f32 {
            DType::F32
        } else {
            k_new_t.dtype()
        };
        let (k_store, v_store) = if store_dtype == k_new_t.dtype() {
            (k_new_t.clone(), v_new.clone())
        } else {
            (k_new_t.to_dtype(store_dtype)?, v_new.to_dtype(store_dtype)?)
        };
        Some(KvCacheChunk::new(k_store, v_store)?)
    } else {
        None
    };
    let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((
        batch,
        seq_len,
        cfg.num_attention_heads * v_head_dim,
    ))?;

    let out = if use_attn_f32 {
        // Output projection is another sensitive reduction path; keep it in f32.
        apply_linear_f32_then_cast(&attn_output, &weights.o_proj, hidden_states.dtype())?
    } else {
        apply_linear(&attn_output, &weights.o_proj)?
    };
    Ok((out, present))
}

fn flash_attention_forward(
    hidden_states: &Tensor,
    weights: &AttentionWeights,
    cfg: &DeepseekV2Config,
    rope: Option<(&Tensor, &Tensor)>,
    additive_attn_bias: Option<&Tensor>,
    past_key_value: Option<&KvCacheEntry>,
    use_cache: bool,
) -> Result<Option<(Tensor, Option<KvCacheChunk>)>> {
    #[cfg(not(feature = "flash-attn"))]
    {
        let _ = (
            hidden_states,
            weights,
            cfg,
            rope,
            additive_attn_bias,
            past_key_value,
            use_cache,
        );
        Ok(None)
    }
    #[cfg(feature = "flash-attn")]
    {
        if additive_attn_bias.is_some() || past_key_value.is_some() || use_cache {
            return Ok(None);
        }
        let device = hidden_states.device();
        if !device.is_cuda() {
            return Ok(None);
        }
        let (batch, seq_len, hidden_size) = hidden_states.shape().dims3()?;
        let dtype = hidden_states.dtype();
        match dtype {
            DType::F16 | DType::BF16 => {}
            _ => return Ok(None),
        }
        let head_dim = hidden_size / cfg.num_attention_heads;
        if !head_dim.is_multiple_of(8) || head_dim > 256 {
            return Ok(None);
        }
        let num_kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
        if !cfg.num_attention_heads.is_multiple_of(num_kv_heads) {
            return Ok(None);
        }

        let mut q = apply_linear(hidden_states, &weights.q_proj)?
            .reshape((batch, seq_len, cfg.num_attention_heads, head_dim))?
            .to_dtype(dtype)?
            .to_device(device)?;
        let kv_head_dim = head_dim;
        let mut k = apply_linear(hidden_states, &weights.k_proj)?
            .reshape((batch, seq_len, num_kv_heads, kv_head_dim))?
            .to_dtype(dtype)?
            .to_device(device)?;
        let v_head_dim = if cfg.v_head_dim.unwrap_or(0) == 0 {
            head_dim
        } else {
            cfg.v_head_dim.unwrap()
        };
        let v = apply_linear(hidden_states, &weights.v_proj)?
            .reshape((batch, seq_len, num_kv_heads, v_head_dim))?
            .to_dtype(dtype)?
            .to_device(device)?;

        if let Some((cos, sin)) = rope {
            let (cos, sin) = (cos.to_device(device)?, sin.to_device(device)?);
            let rope_dim_cfg = cfg.qk_rope_head_dim.unwrap_or(head_dim);
            let rope_dim = if rope_dim_cfg == 0 {
                head_dim
            } else {
                rope_dim_cfg
            };
            ensure!(
                rope_dim <= head_dim,
                "rope dimension {} exceeds q head dimension {}",
                rope_dim,
                head_dim
            );
            ensure!(
                rope_dim <= kv_head_dim,
                "rope dimension {} exceeds k head dimension {}",
                rope_dim,
                kv_head_dim
            );
            ensure!(
                cos.shape().dims() == [batch, 1, seq_len, rope_dim],
                "cos shape {:?} incompatible with (batch={}, seq={}, rope_dim={})",
                cos.shape().dims(),
                batch,
                seq_len,
                rope_dim
            );
            ensure!(
                sin.shape().dims() == [batch, 1, seq_len, rope_dim],
                "sin shape {:?} incompatible with (batch={}, seq={}, rope_dim={})",
                sin.shape().dims(),
                batch,
                seq_len,
                rope_dim
            );
            let q_rot = q.narrow(D::Minus1, 0, rope_dim)?;
            let q_tail = if rope_dim < head_dim {
                Some(q.narrow(D::Minus1, rope_dim, head_dim - rope_dim)?)
            } else {
                None
            };
            let k_rot = k.narrow(D::Minus1, 0, rope_dim)?;
            let k_tail = if rope_dim < kv_head_dim {
                Some(k.narrow(D::Minus1, rope_dim, kv_head_dim - rope_dim)?)
            } else {
                None
            };
            let q_rot = apply_rope(&q_rot, &cos, &sin, cfg.use_mla)?;
            let k_rot = apply_rope(&k_rot, &cos, &sin, cfg.use_mla)?;
            q = if let Some(tail) = q_tail {
                Tensor::cat(&[q_rot, tail], D::Minus1)?
            } else {
                q_rot
            }
            .contiguous()?;
            k = if let Some(tail) = k_tail {
                Tensor::cat(&[k_rot, tail], D::Minus1)?
            } else {
                k_rot
            }
            .contiguous()?;
        }

        q = q.contiguous()?;
        k = k.contiguous()?;
        let v = v.contiguous()?;
        let causal = true;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v_t = v.transpose(1, 2)?;
        let attn = flash_attn(&q, &k, &v_t, scale, causal)?;
        let attn = attn.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            cfg.num_attention_heads * v_head_dim,
        ))?;
        let out = apply_linear(&attn, &weights.o_proj)?;
        Ok(Some((out, None)))
    }
}

fn mlp_forward(
    layer_idx: usize,
    hidden_states: &Tensor,
    weights: &MlpWeights,
    cfg: &DeepseekV2Config,
) -> Result<MlpForwardOutput> {
    match weights {
        MlpWeights::Dense(dense) => run_dense_mlp(layer_idx, hidden_states, dense, cfg),
        MlpWeights::Moe(moe) => run_moe(layer_idx, hidden_states, moe, cfg),
    }
}

fn apply_linear(input: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() < 2 {
        bail!("linear expects rank >= 2, received {:?}", dims);
    }
    let last_dim = *dims.last().expect("at least one dim");
    let (out_dim, in_dim) = (weights.out_dim, weights.in_dim);
    if in_dim != last_dim {
        bail!(
            "linear weight expects input dim {}, got {}",
            in_dim,
            last_dim
        );
    }

    let leading = dims[..dims.len() - 1].iter().product::<usize>();
    let input2d = input.reshape((leading, in_dim))?.contiguous()?;
    let proj = if let Some(qm) = &weights.qmatmul {
        run_quantized_matmul(&weights.label, qm, &input2d)?
    } else {
        let weight = weights
            .weight
            .as_ref()
            .context("float linear weight missing for non-quantized layer")?;
        let weight = if weight.dtype() != input2d.dtype() {
            weight.to_dtype(input2d.dtype())?
        } else {
            weight.clone()
        };
        let weight = weight.contiguous()?;
        input2d.matmul(&transpose(&weight, 0, 1)?)?
    };
    let proj = if let Some(bias) = &weights.bias {
        let bias = if bias.dtype() != proj.dtype() {
            bias.to_dtype(proj.dtype())?
        } else {
            bias.clone()
        };
        proj.broadcast_add(&bias.reshape((1, out_dim))?)?
    } else {
        proj
    };
    proj.reshape(
        dims[..dims.len() - 1]
            .iter()
            .copied()
            .chain(std::iter::once(out_dim))
            .collect::<Vec<_>>(),
    )
    .context("failed to reshape linear output")
}

fn apply_linear_f32_then_cast(
    input: &Tensor,
    weights: &LinearWeights,
    out_dtype: DType,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() < 2 {
        bail!("linear expects rank >= 2, received {:?}", dims);
    }
    let last_dim = *dims.last().expect("at least one dim");
    let (out_dim, in_dim) = (weights.out_dim, weights.in_dim);
    if in_dim != last_dim {
        bail!(
            "linear weight expects input dim {}, got {}",
            in_dim,
            last_dim
        );
    }

    // Quantized path: keep existing behaviour.
    if let Some(qm) = &weights.qmatmul {
        let leading = dims[..dims.len() - 1].iter().product::<usize>();
        let input2d = input.reshape((leading, in_dim))?.contiguous()?;
        let proj = run_quantized_matmul(&weights.label, qm, &input2d)?;
        let proj = if let Some(bias) = &weights.bias {
            proj.broadcast_add(&bias.reshape((1, out_dim))?)?
        } else {
            proj
        };
        return Ok(proj
            .reshape(
                dims[..dims.len() - 1]
                    .iter()
                    .copied()
                    .chain(std::iter::once(out_dim))
                    .collect::<Vec<_>>(),
            )?
            .to_dtype(out_dtype)?);
    }

    let leading = dims[..dims.len() - 1].iter().product::<usize>();
    let x2d = input
        .to_dtype(DType::F32)?
        .reshape((leading, in_dim))?
        .contiguous()?;
    let weight = weights
        .weight
        .as_ref()
        .context("float linear weight missing for non-quantized layer")?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let mut proj = x2d.matmul(&transpose(&weight, 0, 1)?)?;
    if let Some(bias) = &weights.bias {
        let bias = bias.to_dtype(DType::F32)?;
        proj = proj.broadcast_add(&bias.reshape((1, out_dim))?)?;
    }
    Ok(proj
        .reshape(
            dims[..dims.len() - 1]
                .iter()
                .copied()
                .chain(std::iter::once(out_dim))
                .collect::<Vec<_>>(),
        )?
        .to_dtype(out_dtype)?)
}

fn apply_linear_f32_keep(input: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() < 2 {
        bail!("linear expects rank >= 2, received {:?}", dims);
    }
    let last_dim = *dims.last().expect("at least one dim");
    let (out_dim, in_dim) = (weights.out_dim, weights.in_dim);
    if in_dim != last_dim {
        bail!(
            "linear weight expects input dim {}, got {}",
            in_dim,
            last_dim
        );
    }
    let leading = dims[..dims.len() - 1].iter().product::<usize>();
    let input2d = input.reshape((leading, in_dim))?.contiguous()?;

    let proj = if let Some(qm) = &weights.qmatmul {
        run_quantized_matmul(&weights.label, qm, &input2d)?.to_dtype(DType::F32)?
    } else {
        let weight = weights
            .weight
            .as_ref()
            .context("float linear weight missing for non-quantized layer")?;
        let x = input2d.to_dtype(DType::F32)?;
        let w = if let Some(w_f32) = &weights.weight_f32 {
            w_f32.clone()
        } else {
            weight.to_dtype(DType::F32)?
        };
        let w = w.contiguous()?;
        x.matmul(&transpose(&w, 0, 1)?)?
    };

    let proj = if let Some(bias) = &weights.bias {
        let bias = bias.to_dtype(DType::F32)?;
        proj.broadcast_add(&bias.reshape((1, out_dim))?)?
    } else {
        proj
    };

    proj.reshape(
        dims[..dims.len() - 1]
            .iter()
            .copied()
            .chain(std::iter::once(out_dim))
            .collect::<Vec<_>>(),
    )
    .context("failed to reshape linear output")
}

fn repeat_kv(t: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 0 {
        bail!("repeat_kv expects repeats >= 1");
    }
    if repeats == 1 {
        return Ok(t.clone());
    }
    let (batch, heads, seq_len, dim) = t
        .shape()
        .dims4()
        .context("expected [batch, heads, seq, dim] tensor")?;
    let expanded = t
        .unsqueeze(2)?
        .expand((batch, heads, repeats, seq_len, dim))?
        .reshape((batch, heads * repeats, seq_len, dim))?;
    Ok(expanded.contiguous()?)
}

fn apply_activation(input: &Tensor, name: &str) -> Result<Tensor> {
    let normalized = name.to_ascii_lowercase();
    match normalized.as_str() {
        "silu" | "swish" => Ok(input.silu()?),
        "relu" => Ok(input.relu()?),
        // Match PyTorch `nn.GELU()` default (approximate="none").
        "gelu" => Ok(input.gelu_erf()?),
        "gelu_erf" => Ok(input.gelu_erf()?),
        _ => bail!("activation `{name}` not implemented"),
    }
}

fn run_dense_mlp_f32_keep(
    hidden_states: &Tensor,
    weights: &DenseMlpWeights,
    cfg: &DeepseekV2Config,
) -> Result<Tensor> {
    let gate = apply_linear_f32_keep(hidden_states, &weights.gate_proj)?;
    let up = apply_linear_f32_keep(hidden_states, &weights.up_proj)?;
    let activated = apply_activation(&gate, &cfg.hidden_act)
        .with_context(|| format!("unsupported activation {}", cfg.hidden_act))?;
    let fused = activated.broadcast_mul(&up)?;
    apply_linear_f32_keep(&fused, &weights.down_proj)
}

fn run_dense_mlp(
    _layer_idx: usize,
    hidden_states: &Tensor,
    weights: &DenseMlpWeights,
    cfg: &DeepseekV2Config,
) -> Result<MlpForwardOutput> {
    let use_f32 = is_low_precision(hidden_states);

    let (gate, up) = if use_f32 {
        (
            apply_linear_f32_keep(hidden_states, &weights.gate_proj)?,
            apply_linear_f32_keep(hidden_states, &weights.up_proj)?,
        )
    } else {
        (
            apply_linear(hidden_states, &weights.gate_proj)?,
            apply_linear(hidden_states, &weights.up_proj)?,
        )
    };

    let activated = apply_activation(&gate, &cfg.hidden_act)
        .with_context(|| format!("unsupported activation {}", cfg.hidden_act))?;
    let fused = activated.broadcast_mul(&up)?;

    let down = if use_f32 {
        let out_dtype = hidden_states.dtype();
        apply_linear_f32_then_cast(&fused, &weights.down_proj, out_dtype)?
    } else {
        apply_linear(&fused, &weights.down_proj)?
    };
    Ok(MlpForwardOutput {
        hidden_states: down,
        aux_loss: None,
    })
}

fn run_moe(
    layer_idx: usize,
    hidden_states: &Tensor,
    weights: &MoeWeights,
    cfg: &DeepseekV2Config,
) -> Result<MlpForwardOutput> {
    let n_routed = cfg
        .n_routed_experts
        .with_context(|| "MoE config missing n_routed_experts")?;
    ensure!(n_routed > 0, "n_routed_experts must be > 0 for MoE");
    let num_experts_per_tok = cfg
        .num_experts_per_tok
        .with_context(|| "MoE config missing num_experts_per_tok")?;
    ensure!(
        num_experts_per_tok > 0 && num_experts_per_tok <= n_routed,
        "num_experts_per_tok ({num_experts_per_tok}) must be within 1..=n_routed_experts ({n_routed})"
    );
    ensure!(
        weights.experts.len() == n_routed,
        "MoE expert count {} does not match config n_routed_experts {}",
        weights.experts.len(),
        n_routed
    );
    let topk_method = cfg.topk_method.as_deref().unwrap_or("greedy");
    ensure!(
        topk_method == "greedy",
        "MoE topk_method `{topk_method}` not yet supported (greedy only)"
    );
    let scoring = cfg.scoring_func.as_deref().unwrap_or("softmax");
    ensure!(
        scoring == "softmax" || scoring == "sigmoid",
        "MoE scoring `{scoring}` not yet supported"
    );
    ensure!(
        cfg.ep_size <= 1,
        "MoE ep_size > 1 not supported in Candle port (got {})",
        cfg.ep_size
    );

    let (batch, seq_len, hidden) = hidden_states.shape().dims3()?;
    let token_count = batch * seq_len;
    let topk = num_experts_per_tok;
    let assignment_count = token_count * topk;

    let device = hidden_states.device();
    let low_precision = matches!(hidden_states.dtype(), DType::F16 | DType::BF16);
    let tokens = hidden_states.reshape((token_count, hidden))?.contiguous()?;

    let tokens_f32 = tokens.to_dtype(DType::F32)?;
    let gate_weight = weights.gate_weight.to_dtype(DType::F32)?.contiguous()?;
    let mut logits = tokens_f32.matmul(&transpose(&gate_weight, 0, 1)?)?;
    if let Some(bias) = &weights.aux_bias {
        let bias = bias.to_dtype(DType::F32)?.reshape((1, n_routed))?;
        logits = logits.broadcast_add(&bias)?;
    }
    // Keep gating in f32 for determinism across low-precision backends.
    let scores = match scoring {
        "softmax" => softmax(&logits, D::Minus1)?,
        "sigmoid" => sigmoid(&logits)?,
        _ => unreachable!("validated scoring method earlier"),
    };
    let scores = scores.to_dtype(DType::F32)?;
    let scores = scores.contiguous()?;
    let (sorted_scores, sorted_indices) = scores.sort_last_dim(false)?;
    let sorted_scores = sorted_scores.contiguous()?;
    let sorted_indices = sorted_indices.contiguous()?;
    let mut topk_weights = sorted_scores.narrow(D::Minus1, 0, topk)?;
    let topk_indices = sorted_indices
        .narrow(D::Minus1, 0, topk)?
        .to_dtype(DType::I64)?
        .contiguous()?;

    if topk > 1 && cfg.norm_topk_prob {
        let denom = topk_weights.sum_keepdim(D::Minus1)?;
        let eps = Tensor::full(1e-20f32, denom.shape(), denom.device())?;
        topk_weights = topk_weights.broadcast_div(&denom.add(&eps)?)?;
    }
    if cfg.routed_scaling_factor != 1.0 {
        let scale = Tensor::full(
            cfg.routed_scaling_factor,
            topk_weights.shape(),
            topk_weights.device(),
        )?;
        topk_weights = topk_weights.mul(&scale)?;
    }
    let topk_weights = topk_weights.contiguous()?;
    let topk_indices = topk_indices.contiguous()?;

    let flat_topk_ids = topk_indices.reshape((assignment_count,))?.contiguous()?;
    let flat_ids = flat_topk_ids.to_vec1::<i64>()?;
    let mut idxs_vec: Vec<u32> = (0..assignment_count as u32).collect();
    idxs_vec.sort_by_key(|&pos| flat_ids[pos as usize]);

    let idxs = Tensor::from_vec(idxs_vec.clone(), (assignment_count,), device)?.contiguous()?;
    let mut token_pos_vec: Vec<u32> = Vec::with_capacity(assignment_count);
    for &pos in idxs_vec.iter() {
        token_pos_vec.push((pos as usize / topk) as u32);
    }
    let token_pos = Tensor::from_vec(token_pos_vec, (assignment_count,), device)?.contiguous()?;
    let sorted_tokens = tokens.index_select(&token_pos, 0)?.contiguous()?;

    let mut tokens_per_expert = vec![0usize; n_routed];
    for &expert_id in flat_ids.iter() {
        let expert_id = expert_id as usize;
        ensure!(
            expert_id < n_routed,
            "expert id {expert_id} out of range 0..{n_routed}"
        );
        tokens_per_expert[expert_id] += 1;
    }

    let mut outputs: Vec<Tensor> = Vec::new();
    let mut start_idx = 0usize;
    for (expert_idx, &num_tokens) in tokens_per_expert.iter().enumerate() {
        let end_idx = start_idx + num_tokens;
        ensure!(end_idx <= assignment_count, "moe routing slice overflow");
        if num_tokens != 0 {
            let tokens_for_expert = sorted_tokens.narrow(0, start_idx, num_tokens)?;
            let expert_out = if low_precision {
                run_dense_mlp_f32_keep(
                    &tokens_for_expert.to_dtype(DType::F32)?,
                    &weights.experts[expert_idx],
                    cfg,
                )?
            } else {
                run_dense_mlp(
                    layer_idx,
                    &tokens_for_expert,
                    &weights.experts[expert_idx],
                    cfg,
                )?
                .hidden_states
            };
            outputs.push(expert_out.to_dtype(DType::F32)?);
        }
        start_idx = end_idx;
    }
    ensure!(
        start_idx == assignment_count,
        "moe routing consumed {start_idx} assignments but expected {assignment_count}"
    );

    let outs = if outputs.is_empty() {
        Tensor::zeros((assignment_count, hidden), DType::F32, device)?
    } else {
        Tensor::cat(&outputs, 0)?
    }
    .contiguous()?;

    let new_x = {
        let new_x = Tensor::zeros((assignment_count, hidden), DType::F32, device)?;
        let idx_matrix = idxs
            .reshape((assignment_count, 1))?
            .expand((assignment_count, hidden))?
            .contiguous()?;
        new_x.scatter_set(&idx_matrix, &outs, 0)?;
        new_x
    };

    let combined = {
        let x = new_x.reshape((token_count, topk, hidden))?;
        let w = topk_weights.to_dtype(DType::F32)?;
        x.broadcast_mul(&w.unsqueeze(D::Minus1)?)?
            .sum(1)?
            .to_dtype(hidden_states.dtype())?
            .reshape((batch, seq_len, hidden))?
    };

    let mut combined = combined;
    if let Some(shared) = &weights.shared_experts {
        let shared_out = run_dense_mlp_f32_keep(&hidden_states.to_dtype(DType::F32)?, shared, cfg)?
            .to_dtype(hidden_states.dtype())?;
        let shared_out = shared_out.to_device(hidden_states.device())?;
        combined = add_stable(&combined, &shared_out)?;
    }

    Ok(MlpForwardOutput {
        hidden_states: combined,
        aux_loss: None,
    })
}

fn transpose(t: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor> {
    let mut dims: Vec<usize> = (0..t.rank()).collect();
    dims.swap(dim0, dim1);
    Ok(t.permute(dims)?)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, reorder: bool) -> Result<Tensor> {
    // DeepSeek MLA path uses an extra even/odd regrouping before rotate_half.
    let x = if reorder {
        let last = x.dim(D::Minus1)?;
        if last == 0 {
            x.clone()
        } else {
            ensure!(
                last % 2 == 0,
                "apply_rope expects an even rope dimension, got {last}"
            );
            let (b, h, s, d) = x
                .shape()
                .dims4()
                .context("apply_rope expects x with shape [batch, heads, seq, rope_dim]")?;
            ensure!(d == last, "internal rope dim mismatch (d={d}, last={last})");
            x.reshape((b, h, s, d / 2, 2))?
                .transpose(3, 4)?
                .contiguous()?
                .reshape((b, h, s, d))?
        }
    } else {
        x.clone()
    };

    let out_dtype = x.dtype();
    let low_precision = is_low_precision(&x);
    let (x, cos, sin) = if low_precision {
        (
            x.to_dtype(DType::F32)?,
            cos.to_dtype(DType::F32)?,
            sin.to_dtype(DType::F32)?,
        )
    } else {
        let cos = if cos.dtype() == out_dtype {
            cos.clone()
        } else {
            cos.to_dtype(out_dtype)?
        };
        let sin = if sin.dtype() == out_dtype {
            sin.clone()
        } else {
            sin.to_dtype(out_dtype)?
        };
        (x, cos, sin)
    };

    let rotated = rotate_half(&x)?;
    let x_cos = x.broadcast_mul(&cos)?;
    let rot_sin = rotated.broadcast_mul(&sin)?;
    let out = x_cos.add(&rot_sin)?;
    if low_precision {
        Ok(out.to_dtype(out_dtype)?)
    } else {
        Ok(out)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    ensure!(
        last % 2 == 0,
        "rotate_half expects even dimension, got {last}"
    );
    let left = x.narrow(D::Minus1, 0, last / 2)?;
    let right = x.narrow(D::Minus1, last / 2, last / 2)?;
    let neg_right = right.neg()?;
    Ok(Tensor::cat(&[neg_right, left], D::Minus1)?)
}

/// Construct a padding mask from per-batch sequence lengths.
///
/// Returns a tensor of shape `(batch, seq_len)` with `1.0` for real tokens and `0.0` for padding.
pub fn lengths_to_padding_mask(
    lengths: &[usize],
    seq_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let batch = lengths.len();
    let mut data = vec![0f32; batch * seq_len];
    for (batch_idx, &len) in lengths.iter().enumerate() {
        ensure!(
            len <= seq_len,
            "length {} exceeds sequence dimension {}",
            len,
            seq_len
        );
        for pos in 0..len {
            data[batch_idx * seq_len + pos] = 1.0;
        }
    }
    Ok(Tensor::from_vec(data, (batch, seq_len), device)?)
}

fn mask_fill_value(dtype: DType) -> f32 {
    match dtype {
        DType::F16 | DType::BF16 => -1e4f32,
        _ => -1e9f32,
    }
}

pub fn build_attention_bias(
    pad_mask: Option<&Tensor>,
    batch: usize,
    q_len: usize,
    k_len: usize,
    past_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    let mut bias: Option<Tensor> = None;

    if past_len == 0 && q_len == k_len && q_len > 1 {
        let rows = Tensor::arange(0i64, q_len as i64, device)?.reshape((q_len, 1))?;
        let cols = Tensor::arange(0i64, k_len as i64, device)?.reshape((1, k_len))?;
        let mask = cols.broadcast_gt(&rows)?;
        let mask = mask.to_dtype(dtype)?;
        let fill =
            Tensor::full(mask_fill_value(dtype), mask.shape().clone(), device)?.to_dtype(dtype)?;
        let causal = mask.mul(&fill)?;
        let causal = causal.reshape((1, 1, q_len, k_len))?;
        let causal = causal.expand((batch, 1, q_len, k_len))?;
        bias = Some(causal);
    }

    if let Some(mask) = pad_mask {
        let (b, s) = mask.shape().dims2()?;
        ensure!(
            b == batch,
            "padding mask batch {} does not match input batch {}",
            b,
            batch
        );
        ensure!(
            s == k_len,
            "padding mask seq {} does not match key length {}",
            s,
            k_len
        );
        let mask = if mask.dtype() == dtype {
            mask.clone()
        } else {
            mask.to_dtype(dtype)?
        };
        let ones = Tensor::full(1f32, (batch, k_len), device)?.to_dtype(dtype)?;
        let inv = ones.sub(&mask)?;
        let inv = inv.reshape((batch, 1, 1, k_len))?;
        let fill =
            Tensor::full(mask_fill_value(dtype), inv.shape().clone(), device)?.to_dtype(dtype)?;
        let pad_bias = inv.mul(&fill)?;
        bias = Some(if let Some(existing) = bias {
            existing.broadcast_add(&pad_bias)?
        } else {
            pad_bias
        });
    }

    Ok(bias)
}
