use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor, shape::D};
use candle_nn::{VarBuilder, ops::rms_norm};
use deepseek_ocr_core::tensor::{into_dtype_if_needed, to_dtype_if_needed};

use crate::config::{DeepseekOcrConfig, DeepseekV2Config};
use crate::transformer::block::TransformerBlock;
use crate::transformer::weights::{RmsNormWeights, TransformerWeights};
use crate::vision::sam::SamBackbone;

const QWEN2_HIDDEN_SIZE: usize = 896;
const QWEN2_NUM_LAYERS: usize = 24;
const QWEN2_NUM_HEADS: usize = 14;
const QWEN2_NUM_KV_HEADS: usize = 2;
const QWEN2_INTERMEDIATE_SIZE: usize = 4864;
const QWEN2_MAX_POSITION: usize = 131_072;
const QWEN2_RMS_NORM_EPS: f64 = 1e-6;
const QWEN2_ROPE_THETA: f64 = 1_000_000.0;
const QWEN2_QUERY_768: usize = 144;
const QWEN2_QUERY_1024: usize = 256;
const QWEN2_PROJECTOR_OUT: usize = 1280;

/// Qwen2 decoder configuration used by OCR2's visual encoder path.
#[derive(Debug, Clone)]
pub struct Qwen2DecoderParams {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
}

impl Qwen2DecoderParams {
    pub fn ocr2_default() -> Self {
        Self {
            num_layers: QWEN2_NUM_LAYERS,
            hidden_size: QWEN2_HIDDEN_SIZE,
            num_heads: QWEN2_NUM_HEADS,
            num_kv_heads: QWEN2_NUM_KV_HEADS,
            intermediate_size: QWEN2_INTERMEDIATE_SIZE,
            max_position_embeddings: QWEN2_MAX_POSITION,
            rms_norm_eps: QWEN2_RMS_NORM_EPS,
            rope_theta: QWEN2_ROPE_THETA,
            hidden_act: "silu".to_string(),
        }
    }

    fn as_deepseek_config(&self) -> DeepseekV2Config {
        DeepseekV2Config {
            vocab_size: 0,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: None,
            num_hidden_layers: self.num_layers,
            num_attention_heads: self.num_heads,
            num_key_value_heads: Some(self.num_kv_heads),
            n_shared_experts: None,
            n_routed_experts: None,
            ep_size: 1,
            routed_scaling_factor: 1.0,
            kv_lora_rank: None,
            q_lora_rank: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
            qk_nope_head_dim: None,
            topk_method: None,
            n_group: None,
            topk_group: None,
            num_experts_per_tok: None,
            moe_layer_freq: 2,
            moe_layer_freq_override: None,
            first_k_dense_replace: None,
            norm_topk_prob: false,
            scoring_func: None,
            aux_loss_alpha: 0.001,
            seq_aux: true,
            hidden_act: self.hidden_act.clone(),
            max_position_embeddings: self.max_position_embeddings,
            initializer_range: 0.02,
            rms_norm_eps: self.rms_norm_eps as f32,
            use_cache: false,
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            pretraining_tp: 1,
            tie_word_embeddings: false,
            attn_implementation: None,
            rope_theta: self.rope_theta as f32,
            rope_scaling: None,
            attention_bias: true,
            attention_dropout: 0.0,
            use_mla: false,
            torch_dtype: None,
            lm_head: None,
            rm_head: None,
            extra: BTreeMap::new(),
        }
    }
}

/// Qwen2 decoder-as-encoder used by OCR2 to refine SAM embeddings.
pub struct Qwen2DecoderAsEncoder {
    params: Qwen2DecoderParams,
    cfg: DeepseekV2Config,
    weights: TransformerWeights,
    norm: RmsNormWeights,
    query_768: Tensor,
    query_1024: Tensor,
}

impl Qwen2DecoderAsEncoder {
    pub fn load(params: Qwen2DecoderParams, vb: &VarBuilder) -> Result<Self> {
        let cfg = params.as_deepseek_config();
        let core_vb = vb.pp("model").pp("model");
        let weights = TransformerWeights::load(&cfg, &core_vb, None)
            .context("failed to load Qwen2 decoder weights")?;
        let norm_weight = core_vb
            .pp("norm")
            .get(cfg.hidden_size, "weight")
            .context("missing Qwen2 final norm weight")?;
        let norm = RmsNormWeights {
            weight: norm_weight,
        };
        let query_768 = vb
            .get((QWEN2_QUERY_768, cfg.hidden_size), "query_768.weight")
            .context("missing Qwen2 query_768 embedding")?;
        let query_1024 = vb
            .get((QWEN2_QUERY_1024, cfg.hidden_size), "query_1024.weight")
            .context("missing Qwen2 query_1024 embedding")?;

        Ok(Self {
            params,
            cfg,
            weights,
            norm,
            query_768,
            query_1024,
        })
    }

    pub fn forward(&self, sam_features: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = sam_features
            .shape()
            .dims4()
            .context("Qwen2 decoder expects SAM features as [batch, channels, height, width]")?;
        ensure!(
            channels == self.cfg.hidden_size,
            "Qwen2 decoder expects hidden size {}, got {}",
            self.cfg.hidden_size,
            channels
        );
        let tokens = sam_features
            .flatten(2, 3)?
            .transpose(1, 2)?
            .contiguous()
            .context("SAM features flatten/transpose not contiguous")?;
        let seq_len = height * width;
        self.forward_tokens(tokens, seq_len, batch)
    }

    fn forward_tokens(&self, tokens: Tensor, seq_len: usize, batch: usize) -> Result<Tensor> {
        let dtype = tokens.dtype();
        let device = tokens.device().clone();

        let query_embed = self
            .select_query_embedding(seq_len, &device, dtype)?
            .unsqueeze(0)?
            .expand((batch, seq_len, self.cfg.hidden_size))?;

        let combined = Tensor::cat(&[tokens, query_embed], 1)?
            .contiguous()
            .context("Qwen2 combined inputs not contiguous")?;

        let token_type_ids = build_token_type_ids(batch, seq_len, &device)?;
        let attn_bias = build_custom_attention_mask(&token_type_ids, dtype)?;

        let rope_dim = self
            .cfg
            .qk_rope_head_dim
            .unwrap_or(self.cfg.hidden_size / self.cfg.num_attention_heads);
        let rope_dim = if rope_dim == 0 {
            self.cfg.hidden_size / self.cfg.num_attention_heads
        } else {
            rope_dim
        };
        let (cos, sin) = build_rope_tables(
            combined.shape().dims3()?.1,
            rope_dim,
            self.params.rope_theta,
            &device,
            dtype,
        )?;
        let cos = cos.expand((batch, 1, combined.shape().dims3()?.1, rope_dim))?;
        let sin = sin.expand((batch, 1, combined.shape().dims3()?.1, rope_dim))?;

        let mut hidden = combined;
        for weights in &self.weights.layers {
            let block = TransformerBlock::new(&self.cfg, weights, false);
            let output = block.forward(
                0,
                &hidden,
                Some(&attn_bias),
                Some((&cos, &sin)),
                None,
                false,
            )?;
            hidden = output.hidden_states;
        }

        let hidden = rms_norm(&hidden, &self.norm.weight, self.cfg.rms_norm_eps)?;
        let query_start = seq_len;
        hidden
            .narrow(D::Minus2, query_start, seq_len)
            .context("Qwen2 decoder query slice failed")
    }

    fn select_query_embedding(
        &self,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let query = match seq_len {
            QWEN2_QUERY_768 => self.query_768.clone(),
            QWEN2_QUERY_1024 => self.query_1024.clone(),
            _ => bail!(
                "unsupported Qwen2 query length {seq_len} (expected {QWEN2_QUERY_768} or {QWEN2_QUERY_1024})"
            ),
        };
        let mut query = query;
        if !query.device().same_device(device) {
            query = query.to_device(device)?;
        }
        query = into_dtype_if_needed(query, dtype)?;
        Ok(query)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2VisionParams {
    pub hidden_size: usize,
    pub projector_out: usize,
    pub query_768: usize,
    pub query_1024: usize,
}

impl Qwen2VisionParams {
    pub fn ocr2_default() -> Self {
        Self {
            hidden_size: QWEN2_HIDDEN_SIZE,
            projector_out: QWEN2_PROJECTOR_OUT,
            query_768: QWEN2_QUERY_768,
            query_1024: QWEN2_QUERY_1024,
        }
    }
}

/// Vision encoder for OCR2: SAM -> Qwen2 decoder-as-encoder -> projector.
pub struct Qwen2VisionEncoder {
    params: Qwen2VisionParams,
    sam: SamBackbone,
    decoder: Qwen2DecoderAsEncoder,
    projector: Qwen2Projector,
    view_separator: Tensor,
}

/// OCR2 vision inputs (global image + optional local tiles).
pub struct Qwen2VisionInput<'a> {
    pub global: &'a Tensor,
    pub patches: Option<&'a Tensor>,
    pub crop_shape: Option<(usize, usize)>,
}

impl Qwen2VisionEncoder {
    pub fn new(cfg: &DeepseekOcrConfig, vb: &VarBuilder) -> Result<Self> {
        let params = Qwen2VisionParams::ocr2_default();
        let sam = SamBackbone::new(cfg, &vb.pp("model").pp("sam_model"))?;

        let qwen2_vb = vb.pp("model").pp("qwen2_model");
        let decoder_params = Qwen2DecoderParams::ocr2_default();
        let decoder = Qwen2DecoderAsEncoder::load(decoder_params, &qwen2_vb)?;

        let projector = Qwen2Projector::load(
            &vb.pp("model").pp("projector").pp("layers"),
            params.hidden_size,
            params.projector_out,
        )?;
        let view_separator = vb
            .pp("model")
            .get(params.projector_out, "view_seperator")
            .context("missing Qwen2 view_seperator")?;

        Ok(Self {
            params,
            sam,
            decoder,
            projector,
            view_separator,
        })
    }

    pub fn with_dummy_weights(cfg: &DeepseekOcrConfig) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        Self::new(cfg, &vb)
    }

    pub fn params(&self) -> &Qwen2VisionParams {
        &self.params
    }

    pub fn sam_backbone(&self) -> &SamBackbone {
        &self.sam
    }

    /// Encode global + local tiles into a single token sequence.
    ///
    /// Token order (matches OCR2 Python):
    /// 1) local patch tokens (patch-major, each patch flattened row-major),
    /// 2) global view tokens (row-major),
    /// 3) view separator token.
    ///
    /// Token counts:
    /// - per view: `n_query = (H/64) * (W/64)` where `H,W` are the input pixel size.
    ///   (SAM uses 16x patching then 4x downsample → grid side = H/64.)
    /// - total tokens = `local_patches * n_query + n_query + 1`.
    ///
    /// Attention mask semantics inside Qwen2 decoder:
    /// - image tokens (token_type=0) attend to all image tokens, not query tokens.
    /// - query tokens (token_type=1) attend to all image tokens + causal over queries.
    pub fn encode(&self, input: Qwen2VisionInput<'_>) -> Result<Tensor> {
        let global = self.prepare_image_tensor(input.global)?;
        let global_proj = self.encode_view(&global)?;
        let (global_batch, global_seq, hidden) = global_proj.shape().dims3()?;
        ensure!(global_batch == 1, "global view expects batch size 1");
        ensure!(
            hidden == self.params.projector_out,
            "projector output dim mismatch"
        );

        let mut segments = Vec::new();
        if let Some(patches) = input.patches {
            let patches = self.prepare_image_tensor(patches)?;
            let (patches_batch, _, _, _) = patches.shape().dims4()?;
            if patches_batch > 0 {
                let local_proj = self.encode_view(&patches)?;
                let (_, local_seq, local_hidden) = local_proj.shape().dims3()?;
                ensure!(local_hidden == hidden, "local/global hidden mismatch");
                let local_tokens = local_proj.reshape((patches_batch * local_seq, hidden))?;
                segments.push(local_tokens);
            }
        }

        let global_tokens = global_proj.reshape((global_seq, hidden))?;
        segments.push(global_tokens);

        let separator = self
            .adapt_view_separator(global_proj.dtype(), global_proj.device())?
            .reshape((1, hidden))?;
        segments.push(separator);

        Ok(Tensor::cat(&segments, 0)
            .context("failed to concatenate Qwen2 vision tokens")?
            .contiguous()?)
    }

    fn encode_view(&self, image: &Tensor) -> Result<Tensor> {
        let sam_features = self.sam.forward(image)?;
        let qwen2_features = self.decoder.forward(&sam_features)?;
        self.projector.forward(&qwen2_features)
    }

    fn prepare_image_tensor(&self, image: &Tensor) -> Result<Tensor> {
        let mut image = if image.rank() == 3 {
            image.unsqueeze(0)?
        } else {
            image.clone()
        };
        ensure!(
            image.rank() == 4,
            "image tensor must be rank 4 (batch, channels, height, width)"
        );
        let (_batch, channels, _height, _width) = image.shape().dims4()?;
        ensure!(channels == 3, "vision encoder expects 3-channel inputs");
        image = image.contiguous()?;
        Ok(image)
    }

    fn adapt_view_separator(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        let mut token = self.view_separator.clone();
        if !token.device().same_device(device) {
            token = token.to_device(device)?;
        }
        token = into_dtype_if_needed(token, dtype)?;
        Ok(token)
    }
}

struct Qwen2Projector {
    linear: LinearLayer,
    input_dim: usize,
    output_dim: usize,
}

impl Qwen2Projector {
    fn load(vb: &VarBuilder, input_dim: usize, output_dim: usize) -> Result<Self> {
        let linear = LinearLayer::load(vb, output_dim, input_dim, true)?;
        Ok(Self {
            linear,
            input_dim,
            output_dim,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, _, hidden) = input
            .shape()
            .dims3()
            .context("projector expects 3D input")?;
        ensure!(
            hidden == self.input_dim,
            "projector input dim {} does not match expected {}",
            hidden,
            self.input_dim
        );
        let output = linear_forward(&self.linear, input)?;
        let (_, _, out_hidden) = output.shape().dims3()?;
        ensure!(
            out_hidden == self.output_dim,
            "projector output dim {} does not match expected {}",
            out_hidden,
            self.output_dim
        );
        Ok(output)
    }
}

#[derive(Clone)]
struct LinearLayer {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearLayer {
    fn load(vb: &VarBuilder, out_dim: usize, in_dim: usize, bias: bool) -> Result<Self> {
        let weight = vb
            .get((out_dim, in_dim), "weight")
            .with_context(|| "missing linear weight `weight`")?
            .contiguous()
            .context("linear weight must be contiguous")?;
        let bias = if bias && vb.contains_tensor("bias") {
            Some(
                vb.get(out_dim, "bias")
                    .with_context(|| "missing linear bias `bias`")?,
            )
        } else {
            None
        };
        Ok(Self { weight, bias })
    }
}

fn linear_forward(layer: &LinearLayer, input: &Tensor) -> Result<Tensor> {
    let dims = input.shape().dims();
    ensure!(dims.len() >= 2, "linear expects rank >= 2");
    let last_dim = *dims.last().expect("linear expects rank >= 2");
    let (out_dim, in_dim) = layer.weight.shape().dims2()?;
    ensure!(
        in_dim == last_dim,
        "linear weight expects input dim {}, got {}",
        in_dim,
        last_dim
    );
    let leading = dims[..dims.len() - 1].iter().product::<usize>();
    let reshaped = input.reshape((leading, in_dim))?;
    let weight = to_dtype_if_needed(&layer.weight, reshaped.dtype())?;
    let weight_t = weight.transpose(0, 1)?;
    let mut output = reshaped.matmul(&weight_t)?;
    if let Some(bias) = &layer.bias {
        let bias = to_dtype_if_needed(bias, output.dtype())?.reshape((1, out_dim))?;
        output = output.broadcast_add(&bias)?;
    }
    output = into_dtype_if_needed(output, input.dtype())?;
    output
        .reshape(
            dims[..dims.len() - 1]
                .iter()
                .copied()
                .chain(std::iter::once(out_dim))
                .collect::<Vec<_>>(),
        )
        .context("linear output reshape failed")
}

fn build_token_type_ids(batch: usize, seq: usize, device: &Device) -> Result<Tensor> {
    let image = Tensor::zeros((batch, seq), DType::I64, device)?;
    let query = Tensor::ones((batch, seq), DType::I64, device)?;
    Tensor::cat(&[image, query], 1).context("failed to build token_type_ids")
}

#[doc(hidden)]
pub fn build_custom_attention_mask(token_type_ids: &Tensor, dtype: DType) -> Result<Tensor> {
    let (batch, seq) = token_type_ids
        .shape()
        .dims2()
        .context("token_type_ids must have shape [batch, seq]")?;
    let cpu_ids = token_type_ids.to_device(&Device::Cpu)?;
    let type_ids = cpu_ids
        .to_vec2::<i64>()
        .context("token_type_ids to_vec2 failed")?;

    let min_value = match dtype {
        DType::F16 => -65_504.0,
        _ => f32::MIN,
    };
    let mut mask = vec![min_value; batch * seq * seq];

    for (b, row_ids) in type_ids.iter().enumerate() {
        let mut image_positions = Vec::new();
        let mut query_positions = Vec::new();
        for (idx, &token_type) in row_ids.iter().enumerate() {
            if token_type == 0 {
                image_positions.push(idx);
            } else {
                query_positions.push(idx);
            }
        }

        for &i in &image_positions {
            let row = (b * seq + i) * seq;
            for &j in &image_positions {
                mask[row + j] = 0.0;
            }
        }

        for (qi, &i) in query_positions.iter().enumerate() {
            let row = (b * seq + i) * seq;
            for &j in &image_positions {
                mask[row + j] = 0.0;
            }
            for &j in query_positions.iter().take(qi + 1) {
                mask[row + j] = 0.0;
            }
        }
    }

    let mask = Tensor::from_vec(mask, (batch, seq, seq), token_type_ids.device())?;
    let mask = mask.unsqueeze(1)?;
    to_dtype_if_needed(&mask, dtype)
}

fn build_rope_tables(
    seq_len: usize,
    rope_dim: usize,
    theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    ensure!(rope_dim.is_multiple_of(2), "rope_dim must be even");
    let half = rope_dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = i as f32 / half as f32;
        inv_freq.push(1.0f32 / (theta as f32).powf(exponent));
    }

    let pos = Tensor::arange(0i64, seq_len as i64, device)?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let inv = Tensor::from_vec(inv_freq, (1, half), device)?;
    let angles = pos.matmul(&inv)?;
    let cos_half = angles.cos()?;
    let sin_half = angles.sin()?;
    let cos_full = Tensor::cat(&[cos_half.clone(), cos_half], 1)?;
    let sin_full = Tensor::cat(&[sin_half.clone(), sin_half], 1)?;
    let cos = cos_full
        .to_dtype(dtype)?
        .reshape((1, 1, seq_len, rope_dim))?;
    let sin = sin_full
        .to_dtype(dtype)?
        .reshape((1, 1, seq_len, rope_dim))?;
    Ok((cos, sin))
}
