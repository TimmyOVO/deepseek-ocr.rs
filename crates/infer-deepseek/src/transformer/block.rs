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
use candle_nn::ops::{rms_norm, sigmoid, softmax};

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

    /// Forward pass for a single transformer block.
    ///
    /// * `hidden_states` – shape `[batch, seq, hidden]`
    /// * `attention_mask` – optional mask with shape `[batch, 1, seq, seq]`
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        additive_attn_bias: Option<&Tensor>,
        rope: Option<(&Tensor, &Tensor)>,
        past_key_value: Option<&KvCacheEntry>,
        use_cache: bool,
    ) -> Result<BlockOutput> {
        let residual = hidden_states;
        let normed = rms_norm(
            hidden_states,
            &self.weights.input_layernorm.weight,
            self.cfg.rms_norm_eps as f32,
        )
        .context("input rms norm failed")?;

        let (attn_out, present_cache) = attention_forward(
            &normed,
            &self.weights.attention,
            self.cfg,
            additive_attn_bias,
            rope,
            past_key_value,
            use_cache,
            self.use_flash_attention,
        )
        .context("attention forward failed")?;
        let hidden_states = residual
            .add(&attn_out)
            .context("residual add (attention)")?;

        let residual = &hidden_states;
        let normed = rms_norm(
            residual,
            &self.weights.post_attention_layernorm.weight,
            self.cfg.rms_norm_eps as f32,
        )
        .context("post-attention rms norm failed")?;
        let MlpForwardOutput {
            hidden_states: mlp_hidden,
            aux_loss,
        } = mlp_forward(&normed, &self.weights.mlp, self.cfg).context("mlp forward failed")?;

        let output = residual.add(&mlp_hidden).context("residual add (mlp)")?;
        let present = if use_cache { present_cache } else { None };
        Ok(BlockOutput {
            hidden_states: output,
            present_key_value: present,
            aux_loss,
        })
    }
}

fn attention_forward(
    hidden_states: &Tensor,
    weights: &AttentionWeights,
    cfg: &DeepseekV2Config,
    additive_attn_bias: Option<&Tensor>,
    rope: Option<(&Tensor, &Tensor)>,
    past_key_value: Option<&KvCacheEntry>,
    use_cache: bool,
    use_flash_attention: bool,
) -> Result<(Tensor, Option<KvCacheChunk>)> {
    if cfg.q_lora_rank.is_some() || cfg.kv_lora_rank.is_some() {
        bail!("LoRA attention path not yet implemented");
    }

    if use_flash_attention {
        if let Some(result) = flash_attention_forward(
            hidden_states,
            weights,
            cfg,
            rope,
            additive_attn_bias,
            past_key_value,
            use_cache,
        )? {
            return Ok(result);
        }
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

    // Query / key / value projections.
    let mut q = apply_linear(hidden_states, &weights.q_proj)?.reshape((
        batch,
        seq_len,
        cfg.num_attention_heads,
        head_dim,
    ))?;
    let mut k = apply_linear(hidden_states, &weights.k_proj)?.reshape((
        batch,
        seq_len,
        num_kv_heads,
        kv_head_dim,
    ))?;
    let v = apply_linear(hidden_states, &weights.v_proj)?.reshape((
        batch,
        seq_len,
        num_kv_heads,
        v_head_dim,
    ))?;

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
        let (cos, sin) = rope.context("missing rope tensors for attention")?;
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
        let q_rot = apply_rope(&q_rot, &cos, &sin)?;
        let k_rot = apply_rope(&k_rot, &cos, &sin)?;
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
        cfg.num_attention_heads % num_kv_heads == 0,
        "num_attention_heads {} must be divisible by num_key_value_heads {}",
        cfg.num_attention_heads,
        num_kv_heads
    );
    let repeats = cfg.num_attention_heads / num_kv_heads;
    let mut k_new = repeat_kv(&k, repeats)?;
    let mut v_new = repeat_kv(&v, repeats)?;

    q = q.contiguous()?;
    k_new = k_new.contiguous()?;
    v_new = v_new.contiguous()?;

    let mut cache_key_t_view: Option<Tensor> = None;
    let mut cache_value_view: Option<Tensor> = None;
    let past_len = if let Some(cache) = past_key_value {
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
    let attn_scores_mat = if let Some(cache_key_t) = cache_key_t_view.as_ref() {
        let scores_new = q.matmul(&k_new_t)?;
        if past_len > 0 {
            let cache_key_t = cache_key_t.contiguous()?;
            let scores_past = q.matmul(&cache_key_t)?;
            Tensor::cat(&[scores_past, scores_new], D::Minus1)?
        } else {
            scores_new
        }
    } else {
        q.matmul(&k_new_t)?
    };

    let scale = (head_dim as f64).sqrt();
    let mut attn_scores = (attn_scores_mat / scale)?;
    if let Some(bias) = additive_attn_bias {
        attn_scores = attn_scores.broadcast_add(bias)?;
    }
    let attn_weights = softmax(&attn_scores, D::Minus1).context("attention softmax failed")?;
    let attn_output = if let Some(cache_value_view) = cache_value_view.as_ref() {
        let accum = if past_len > 0 {
            let cache_value = cache_value_view.contiguous()?;
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
    };
    let present = if use_cache {
        Some(KvCacheChunk::new(k_new_t.clone(), v_new.clone())?)
    } else {
        None
    };
    let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((
        batch,
        seq_len,
        cfg.num_attention_heads * v_head_dim,
    ))?;

    let out = apply_linear(&attn_output, &weights.o_proj)?;
    Ok((out, present))
}

#[allow(unused_variables)]
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
        return Ok(None);
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
        if head_dim % 8 != 0 || head_dim > 256 {
            return Ok(None);
        }
        let num_kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
        if cfg.num_attention_heads % num_kv_heads != 0 {
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
            let q_rot = apply_rope(&q_rot, &cos, &sin)?;
            let k_rot = apply_rope(&k_rot, &cos, &sin)?;
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
    hidden_states: &Tensor,
    weights: &MlpWeights,
    cfg: &DeepseekV2Config,
) -> Result<MlpForwardOutput> {
    match weights {
        MlpWeights::Dense(dense) => run_dense_mlp(hidden_states, dense, cfg),
        MlpWeights::Moe(moe) => run_moe(hidden_states, moe, cfg),
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
        input2d.matmul(&transpose(weight, 0, 1)?)?
    };
    let proj = if let Some(bias) = &weights.bias {
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
        "gelu" => Ok(input.gelu()?),
        "gelu_erf" => Ok(input.gelu_erf()?),
        _ => bail!("activation `{name}` not implemented"),
    }
}

fn run_dense_mlp(
    hidden_states: &Tensor,
    weights: &DenseMlpWeights,
    cfg: &DeepseekV2Config,
) -> Result<MlpForwardOutput> {
    let gate = apply_linear(hidden_states, &weights.gate_proj)?;
    let up = apply_linear(hidden_states, &weights.up_proj)?;
    let activated = apply_activation(&gate, &cfg.hidden_act)
        .with_context(|| format!("unsupported activation {}", cfg.hidden_act))?;
    let fused = activated.broadcast_mul(&up)?;
    let down = apply_linear(&fused, &weights.down_proj)?;
    Ok(MlpForwardOutput {
        hidden_states: down,
        aux_loss: None,
    })
}

fn run_moe(
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
    let tokens = hidden_states.reshape((token_count, hidden))?.contiguous()?;
    let tokens_f32 = tokens.to_dtype(DType::F32)?;
    let gate_weight = weights.gate_weight.to_dtype(DType::F32)?;
    let gate_weight_t = gate_weight.transpose(0, 1)?;
    let logits = tokens_f32.matmul(&gate_weight_t)?;
    let scores = match scoring {
        "softmax" => softmax(&logits, D::Minus1)?,
        "sigmoid" => sigmoid(&logits)?,
        _ => unreachable!("validated scoring method earlier"),
    };
    let scores = scores.contiguous()?;
    let (sorted_scores, sorted_indices) = scores.sort_last_dim(false)?;
    let mut topk_weights = sorted_scores.narrow(D::Minus1, 0, num_experts_per_tok)?;
    let mut topk_indices = sorted_indices.narrow(D::Minus1, 0, num_experts_per_tok)?;

    if num_experts_per_tok > 1 && cfg.norm_topk_prob {
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

    topk_indices = topk_indices.to_dtype(DType::I64)?;
    let topk_weights = topk_weights;

    let assignments_tensor = topk_indices.contiguous()?;
    let weights_tensor = topk_weights.contiguous()?;
    let assignments = assignments_tensor.to_vec2::<i64>()?;
    let weight_vectors = weights_tensor.to_vec2::<f32>()?;
    let mut expert_routes: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_routed];
    for (token_idx, (experts_row, weights_row)) in
        assignments.iter().zip(weight_vectors.iter()).enumerate()
    {
        for k in 0..num_experts_per_tok {
            let expert_id = experts_row[k] as usize;
            let weight = weights_row[k];
            if weight == 0.0 {
                continue;
            }
            expert_routes[expert_id].push((token_idx, weight));
        }
    }

    let dtype = hidden_states.dtype();
    let device = hidden_states.device();
    let accum = Tensor::zeros((token_count, hidden), dtype, device)?.contiguous()?;

    for (expert_idx, assignments) in expert_routes.into_iter().enumerate() {
        if assignments.is_empty() {
            continue;
        }

        let count = assignments.len();
        let token_indices: Vec<i64> = assignments.iter().map(|(token, _)| *token as i64).collect();
        let idx_tensor = Tensor::from_vec(token_indices, (count,), device)?.to_dtype(DType::I64)?;
        let expert_tokens = tokens.index_select(&idx_tensor, 0)?;
        let expert_output =
            run_dense_mlp(&expert_tokens, &weights.experts[expert_idx], cfg)?.hidden_states;

        let gate_weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();
        let mul_weights = Tensor::from_vec(gate_weights, (count,), device)?
            .to_dtype(dtype)?
            .reshape((count, 1))?;
        let weighted = expert_output.broadcast_mul(&mul_weights)?.contiguous()?;

        let idx_matrix = idx_tensor
            .reshape((count, 1))?
            .expand((count, hidden))?
            .contiguous()?;
        accum.scatter_add_set(&idx_matrix, &weighted, 0)?;
    }

    let mut combined = accum.reshape((batch, seq_len, hidden))?;

    if let Some(shared) = &weights.shared_experts {
        let shared_out = run_dense_mlp(hidden_states, shared, cfg)?.hidden_states;
        combined = combined.add(&shared_out)?;
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

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotated = rotate_half(x)?;
    let x_cos = x.broadcast_mul(cos)?;
    let rot_sin = rotated.broadcast_mul(sin)?;
    Ok(x_cos.add(&rot_sin)?)
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
