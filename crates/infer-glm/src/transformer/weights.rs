use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl LinearWeights {
    pub fn load(vb: VarBuilder, out_dim: usize, in_dim: usize, use_bias: bool) -> Result<Self> {
        let weight = vb
            .get((out_dim, in_dim), "weight")
            .with_context(|| format!("missing linear weight at {}", vb.prefix()))?
            .contiguous()?;
        let bias = if use_bias && vb.contains_tensor("bias") {
            Some(
                vb.get(out_dim, "bias")
                    .with_context(|| format!("missing linear bias at {}", vb.prefix()))?
                    .contiguous()?,
            )
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            out_dim,
            in_dim,
        })
    }

    pub fn matmul_2d(&self, input: &Tensor) -> Result<Tensor> {
        let (rows, in_dim) = input.shape().dims2()?;
        ensure!(
            in_dim == self.in_dim,
            "linear input dim mismatch: {} vs {}",
            in_dim,
            self.in_dim
        );
        // Match HF eager path for numerically-sensitive parity:
        // run linear matmul + bias accumulation in fp32, then cast back.
        let out_dtype = input.dtype();
        let input_f32 = if out_dtype == candle_core::DType::F32 {
            input.clone()
        } else {
            input.to_dtype(candle_core::DType::F32)?
        };
        let mut weight_t = self.weight.transpose(0, 1)?;
        if weight_t.dtype() != candle_core::DType::F32 {
            weight_t = weight_t.to_dtype(candle_core::DType::F32)?;
        }
        let mut out = input_f32.matmul(&weight_t)?;
        if let Some(bias) = &self.bias {
            let bias = if bias.dtype() == candle_core::DType::F32 {
                bias.clone()
            } else {
                bias.to_dtype(candle_core::DType::F32)?
            };
            out = out.broadcast_add(&bias.reshape((1, self.out_dim))?)?;
        }
        if out_dtype != candle_core::DType::F32 {
            out = out.to_dtype(out_dtype)?;
        }
        ensure!(out.shape().dims() == [rows, self.out_dim], "linear output shape mismatch");
        Ok(out)
    }

    pub fn forward_2d(&self, input: &Tensor) -> Result<Tensor> {
        self.matmul_2d(input)
    }

    pub fn forward_3d(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, seq, in_dim) = input.shape().dims3()?;
        ensure!(
            in_dim == self.in_dim,
            "linear input dim mismatch: {} vs {}",
            in_dim,
            self.in_dim
        );
        let flat = input.reshape((batch * seq, in_dim))?;
        let out = self.matmul_2d(&flat)?;
        Ok(out.reshape((batch, seq, self.out_dim))?)
    }
}

#[derive(Debug)]
pub struct GlmTextAttentionWeights {
    pub q_proj: LinearWeights,
    pub k_proj: LinearWeights,
    pub v_proj: LinearWeights,
    pub o_proj: LinearWeights,
}

impl GlmTextAttentionWeights {
    pub fn load(
        vb: &VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attention_bias: bool,
    ) -> Result<Self> {
        let q_proj = LinearWeights::load(
            vb.pp("q_proj"),
            num_heads * head_dim,
            hidden_size,
            attention_bias,
        )?;
        let k_proj = LinearWeights::load(
            vb.pp("k_proj"),
            num_kv_heads * head_dim,
            hidden_size,
            attention_bias,
        )?;
        let v_proj = LinearWeights::load(
            vb.pp("v_proj"),
            num_kv_heads * head_dim,
            hidden_size,
            attention_bias,
        )?;
        let o_proj = LinearWeights::load(
            vb.pp("o_proj"),
            hidden_size,
            num_heads * head_dim,
            attention_bias,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }
}

#[derive(Debug)]
pub struct GlmTextMlpWeights {
    pub gate_up_proj: LinearWeights,
    pub down_proj: LinearWeights,
}

impl GlmTextMlpWeights {
    pub fn load(vb: &VarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        let gate_up_proj =
            LinearWeights::load(vb.pp("gate_up_proj"), intermediate_size * 2, hidden_size, false)?;
        let down_proj =
            LinearWeights::load(vb.pp("down_proj"), hidden_size, intermediate_size, false)?;
        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }
}

#[derive(Debug)]
pub struct GlmTextLayerWeights {
    pub attention: GlmTextAttentionWeights,
    pub mlp: GlmTextMlpWeights,
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub post_self_attn_layernorm: Tensor,
    pub post_mlp_layernorm: Tensor,
}

impl GlmTextLayerWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attention_bias: bool,
    ) -> Result<Self> {
        let attention = GlmTextAttentionWeights::load(
            &vb.pp("self_attn"),
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            attention_bias,
        )?;
        let mlp = GlmTextMlpWeights::load(&vb.pp("mlp"), hidden_size, intermediate_size)?;
        let input_layernorm = vb
            .pp("input_layernorm")
            .get(hidden_size, "weight")
            .context("missing input_layernorm.weight")?
            .contiguous()?;
        let post_attention_layernorm = vb
            .pp("post_attention_layernorm")
            .get(hidden_size, "weight")
            .context("missing post_attention_layernorm.weight")?
            .contiguous()?;
        let post_self_attn_layernorm = vb
            .pp("post_self_attn_layernorm")
            .get(hidden_size, "weight")
            .context("missing post_self_attn_layernorm.weight")?
            .contiguous()?;
        let post_mlp_layernorm = vb
            .pp("post_mlp_layernorm")
            .get(hidden_size, "weight")
            .context("missing post_mlp_layernorm.weight")?
            .contiguous()?;
        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            post_self_attn_layernorm,
            post_mlp_layernorm,
        })
    }
}
