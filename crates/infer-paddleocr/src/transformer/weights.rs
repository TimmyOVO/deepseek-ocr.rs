use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::config::PaddleOcrVlConfig;

#[derive(Debug)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LinearWeights {
    pub fn load(vb: VarBuilder, out_dim: usize, in_dim: usize, use_bias: bool) -> Result<Self> {
        let weight = vb
            .get((out_dim, in_dim), "weight")
            .context("missing linear weight")?
            .contiguous()?;
        let bias = if use_bias && vb.contains_tensor("bias") {
            Some(
                vb.get(out_dim, "bias")
                    .context("missing linear bias")?
                    .contiguous()?,
            )
        } else {
            None
        };
        Ok(Self { weight, bias })
    }
}

#[derive(Debug)]
pub struct ErnieAttentionWeights {
    pub q_proj: LinearWeights,
    pub k_proj: LinearWeights,
    pub v_proj: LinearWeights,
    pub o_proj: LinearWeights,
}

impl ErnieAttentionWeights {
    pub fn load(vb: &VarBuilder, cfg: &PaddleOcrVlConfig) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.resolved_num_key_value_heads();

        let attn_vb = vb.pp("self_attn");
        let q_proj = LinearWeights::load(
            attn_vb.pp("q_proj"),
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.use_bias,
        )?;
        let k_proj = LinearWeights::load(
            attn_vb.pp("k_proj"),
            num_kv_heads * head_dim,
            cfg.hidden_size,
            cfg.use_bias,
        )?;
        let v_proj = LinearWeights::load(
            attn_vb.pp("v_proj"),
            num_kv_heads * head_dim,
            cfg.hidden_size,
            cfg.use_bias,
        )?;
        let o_proj = LinearWeights::load(
            attn_vb.pp("o_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.use_bias,
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
pub struct ErnieMlpWeights {
    pub gate_proj: LinearWeights,
    pub up_proj: LinearWeights,
    pub down_proj: LinearWeights,
}

impl ErnieMlpWeights {
    pub fn load(vb: &VarBuilder, cfg: &PaddleOcrVlConfig) -> Result<Self> {
        let mlp_vb = vb.pp("mlp");
        let gate_proj = LinearWeights::load(
            mlp_vb.pp("gate_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            cfg.use_bias,
        )?;
        let up_proj = LinearWeights::load(
            mlp_vb.pp("up_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            cfg.use_bias,
        )?;
        let down_proj = LinearWeights::load(
            mlp_vb.pp("down_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.use_bias,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

#[derive(Debug)]
pub struct ErnieDecoderLayerWeights {
    pub attention: ErnieAttentionWeights,
    pub mlp: ErnieMlpWeights,
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
}

impl ErnieDecoderLayerWeights {
    pub fn load(vb: &VarBuilder, cfg: &PaddleOcrVlConfig) -> Result<Self> {
        let attention = ErnieAttentionWeights::load(vb, cfg)?;
        let mlp = ErnieMlpWeights::load(vb, cfg)?;
        let input_layernorm = vb
            .pp("input_layernorm")
            .get(cfg.hidden_size, "weight")
            .context("missing input_layernorm.weight")?
            .contiguous()?;
        let post_attention_layernorm = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")
            .context("missing post_attention_layernorm.weight")?
            .contiguous()?;
        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

#[derive(Debug)]
pub struct ErnieModelWeights {
    pub embed_tokens: Tensor,
    pub layers: Vec<ErnieDecoderLayerWeights>,
    pub final_norm: Tensor,
}

impl ErnieModelWeights {
    pub fn load(vb: &VarBuilder, cfg: &PaddleOcrVlConfig) -> Result<Self> {
        let embed_tokens = vb
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .context("missing model.embed_tokens.weight")?
            .contiguous()?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            layers.push(ErnieDecoderLayerWeights::load(&layer_vb, cfg)?);
        }

        let final_norm = vb
            .pp("norm")
            .get(cfg.hidden_size, "weight")
            .context("missing model.norm.weight")?
            .contiguous()?;

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
        })
    }
}
