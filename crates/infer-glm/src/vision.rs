use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Module, Tensor, shape::D};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, VarBuilder, conv2d, conv2d_no_bias, layer_norm,
    ops::softmax_last_dim,
};
use deepseek_ocr_core::{
    benchmark::Timer,
    inference::VisionSettings,
    tensor::{into_dtype_if_needed, to_dtype_if_needed},
};
use image::{DynamicImage, RgbImage};

use crate::{
    config::{GlmOcrVisionConfig, GlmPreprocessorConfig},
    transformer::weights::LinearWeights,
};

pub struct GlmImageBatch {
    pub pixel_values: Tensor,
    pub image_grid_thw: Vec<(usize, usize, usize)>,
}

pub struct GlmVisionModel {
    config: Arc<GlmOcrVisionConfig>,
    patch_embed: VisionPatchEmbed,
    blocks: Vec<GlmVisionBlock>,
    post_layernorm: Tensor,
    downsample: Conv2d,
    merger: VisionPatchMerger,
    rotary: VisionRotaryEmbedding,
}

impl GlmVisionModel {
    pub fn load(vb: &VarBuilder, config: Arc<GlmOcrVisionConfig>, dtype: DType) -> Result<Self> {
        let model_vb = vb.pp("model").pp("visual");
        let patch_embed = VisionPatchEmbed::load(config.as_ref(), &model_vb.pp("patch_embed"))?;
        let mut blocks = Vec::with_capacity(config.depth);
        for idx in 0..config.depth {
            let block_vb = model_vb.pp(format!("blocks.{idx}"));
            blocks.push(GlmVisionBlock::load(config.as_ref(), &block_vb)?);
        }
        let post_layernorm = model_vb
            .pp("post_layernorm")
            .get(config.hidden_size, "weight")
            .context("missing model.visual.post_layernorm.weight")?;
        let downsample_cfg = Conv2dConfig {
            stride: config.spatial_merge_size,
            padding: 0,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let downsample = if model_vb.pp("downsample").contains_tensor("bias") {
            conv2d(
                config.hidden_size,
                config.out_hidden_size,
                config.spatial_merge_size,
                downsample_cfg,
                model_vb.pp("downsample"),
            )?
        } else {
            conv2d_no_bias(
                config.hidden_size,
                config.out_hidden_size,
                config.spatial_merge_size,
                downsample_cfg,
                model_vb.pp("downsample"),
            )?
        };
        let rotary = VisionRotaryEmbedding::new(config.as_ref(), vb.device())?;
        let merger = VisionPatchMerger::load(config.as_ref(), &model_vb.pp("merger"), dtype)?;
        Ok(Self {
            config,
            patch_embed,
            blocks,
            post_layernorm,
            downsample,
            merger,
            rotary,
        })
    }

    pub fn encode(&self, pixel_values: &Tensor, grids: &[(usize, usize, usize)]) -> Result<Tensor> {
        let patch_timer = Timer::new("vision.encode.patch_embed");
        let hidden = self.patch_embed.forward(pixel_values)?;
        patch_timer.finish(|event| {
            event.add_field(
                "tokens",
                hidden.dims().first().copied().unwrap_or_default() as u64,
            );
        });
        let layout = VisionSequenceLayout::from_grids(grids, self.config.spatial_merge_size)?;
        let (token_count, hidden_size) = hidden.shape().dims2()?;
        ensure!(
            hidden_size == self.config.hidden_size,
            "patch embed hidden size mismatch: {} vs {}",
            hidden_size,
            self.config.hidden_size
        );
        ensure!(
            token_count == layout.total_patches,
            "patch count mismatch: {} vs {}",
            token_count,
            layout.total_patches
        );

        let (cos, sin) = self.rotary.build_embeddings(grids)?;
        let mut states = hidden;
        let blocks_timer = Timer::new("vision.encode.blocks");
        for (idx, block) in self.blocks.iter().enumerate() {
            let block_timer = Timer::new("vision.encode.block");
            states = block.forward(&states, &layout, &cos, &sin)?;
            block_timer.finish(|event| {
                event.add_field("idx", idx as u64);
            });
        }
        blocks_timer.finish(|event| {
            event.add_field("count", self.blocks.len() as u64);
        });

        let tail_timer = Timer::new("vision.encode.tail");
        states = precise_rms_norm_last_dim(&states, &self.post_layernorm, self.config.rms_norm_eps)
            .context("vision post_layernorm failed")?;

        let merge = self.config.spatial_merge_size;
        let reshaped =
            states.reshape((layout.total_groups, merge, merge, self.config.hidden_size))?;
        let downsample_input = reshaped.permute((0, 3, 1, 2))?;
        let downsampled = self.downsample.forward(&downsample_input)?;
        let merged_tokens =
            downsampled.reshape((layout.total_groups, self.config.out_hidden_size))?;
        let output = self.merger.forward(&merged_tokens)?;
        tail_timer.finish(|event| {
            event.add_field("groups", layout.total_groups as u64);
        });
        Ok(output)
    }
}

pub fn preprocess_images(
    images: &[DynamicImage],
    preprocessor: &GlmPreprocessorConfig,
    device: &Device,
    _vision: VisionSettings,
) -> Result<GlmImageBatch> {
    let mut all_patches = Vec::new();
    let mut grids = Vec::with_capacity(images.len());
    for image in images {
        let processed = preprocess_single_image(image, preprocessor)?;
        grids.push(processed.grid_thw);
        all_patches.extend_from_slice(&processed.flatten_patches);
    }
    let rows = all_patches.len()
        / (3 * preprocessor.temporal_patch_size
            * preprocessor.patch_size
            * preprocessor.patch_size);
    let cols =
        3 * preprocessor.temporal_patch_size * preprocessor.patch_size * preprocessor.patch_size;
    let pixel_values = if rows == 0 {
        Tensor::zeros((0, cols), DType::F32, device)?
    } else {
        Tensor::from_vec(all_patches, (rows, cols), device)?
    };
    Ok(GlmImageBatch {
        pixel_values,
        image_grid_thw: grids,
    })
}

#[derive(Debug)]
struct PreprocessedImage {
    flatten_patches: Vec<f32>,
    grid_thw: (usize, usize, usize),
}

fn preprocess_single_image(
    image: &DynamicImage,
    preprocessor: &GlmPreprocessorConfig,
) -> Result<PreprocessedImage> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let factor = preprocessor.patch_size * preprocessor.spatial_merge_size;
    let (resized_h, resized_w) = smart_resize(
        preprocessor.temporal_patch_size,
        height as usize,
        width as usize,
        preprocessor.temporal_patch_size,
        factor,
        preprocessor.size.shortest_edge,
        preprocessor.size.longest_edge,
    )?;
    let resized = if (resized_w as u32, resized_h as u32) == rgb.dimensions() {
        rgb
    } else {
        resize_rgb_image(&rgb, resized_w as u32, resized_h as u32)?
    };

    let mut chw = vec![0f32; 3 * resized_h * resized_w];
    for y in 0..resized_h {
        for x in 0..resized_w {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            for (c, &channel) in pixel.iter().enumerate().take(3) {
                let mut value = channel as f32;
                if preprocessor.do_rescale {
                    // Match HF `rescale` exactly: multiply in float64 then downcast.
                    value = ((value as f64) * preprocessor.rescale_factor) as f32;
                }
                value = (value - preprocessor.image_mean[c]) / preprocessor.image_std[c];
                let dst = c * resized_h * resized_w + y * resized_w + x;
                chw[dst] = value;
            }
        }
    }

    let temporal = preprocessor.temporal_patch_size.max(1);

    let patch = preprocessor.patch_size;
    let merge = preprocessor.spatial_merge_size;
    let grid_h = resized_h / patch;
    let grid_w = resized_w / patch;
    ensure!(
        grid_h.is_multiple_of(merge) && grid_w.is_multiple_of(merge),
        "grid not divisible by merge size"
    );

    let patch_vec_len = 3 * temporal * patch * patch;
    let mut flatten = Vec::with_capacity(grid_h * grid_w * patch_vec_len);

    for _t in 0..1 {
        for gh in 0..(grid_h / merge) {
            for gw in 0..(grid_w / merge) {
                for mh in 0..merge {
                    for mw in 0..merge {
                        for c in 0..3 {
                            for _tp in 0..temporal {
                                for py in 0..patch {
                                    for px in 0..patch {
                                        let y = (gh * merge + mh) * patch + py;
                                        let x = (gw * merge + mw) * patch + px;
                                        let src = c * resized_h * resized_w + y * resized_w + x;
                                        flatten.push(chw[src]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(PreprocessedImage {
        flatten_patches: flatten,
        grid_thw: (1, grid_h, grid_w),
    })
}

fn smart_resize(
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_factor: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    ensure!(
        num_frames >= temporal_factor,
        "t:{num_frames} must be >= temporal_factor:{temporal_factor}"
    );
    let mut h = height.max(1);
    let mut w = width.max(1);
    if h < factor || w < factor {
        let scale = ((factor as f64) / h as f64).max((factor as f64) / w as f64);
        h = (h as f64 * scale) as usize;
        w = (w as f64 * scale) as usize;
    }
    let aspect = h.max(w) as f64 / h.min(w) as f64;
    ensure!(
        aspect <= 200.0,
        "absolute aspect ratio must be <= 200, got {aspect}"
    );

    let mut h_bar = round_to_multiple(h, factor);
    let mut w_bar = round_to_multiple(w, factor);
    let t_bar = round_to_multiple(num_frames, temporal_factor);

    if t_bar * h_bar * w_bar > max_pixels {
        let beta = ((num_frames * h * w) as f64 / max_pixels as f64).sqrt();
        h_bar = factor.max(((h as f64 / beta).floor() as usize / factor) * factor);
        w_bar = factor.max(((w as f64 / beta).floor() as usize / factor) * factor);
    } else if t_bar * h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (num_frames * h * w) as f64).sqrt();
        h_bar = ((h as f64 * beta).ceil() as usize).div_ceil(factor) * factor;
        w_bar = ((w as f64 * beta).ceil() as usize).div_ceil(factor) * factor;
    }

    Ok((h_bar.max(factor), w_bar.max(factor)))
}

fn round_to_multiple(value: usize, factor: usize) -> usize {
    let rounded = ((value as f32 / factor as f32).round() as usize) * factor;
    rounded.max(factor)
}

const PILLOW_RESAMPLE_PRECISION_BITS: usize = 22;

#[derive(Debug, Clone)]
struct PillowResampleCoeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    coeffs: Vec<i32>,
}

#[inline]
fn pillow_bicubic_filter(mut x: f64) -> f64 {
    const A: f64 = -0.5;
    if x < 0.0 {
        x = -x;
    }
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
    }
    0.0
}

fn precompute_pillow_coeffs(in_size: usize, out_size: usize) -> PillowResampleCoeffs {
    let in0 = 0f32;
    let in1 = in_size as f32;
    let scale = (f64::from(in1 - in0)) / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;

    let scale_int = (1i64 << PILLOW_RESAMPLE_PRECISION_BITS) as f64;
    let mut bounds = Vec::with_capacity(out_size);
    let mut coeffs = vec![0i32; out_size * ksize];

    for xx in 0..out_size {
        let center = f64::from(in0) + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;

        let mut xmin = (center - support + 0.5).trunc() as isize;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5).trunc() as isize;
        if xmax > in_size as isize {
            xmax = in_size as isize;
        }

        let count = (xmax - xmin).max(0) as usize;
        let mut norm = 0.0f64;
        let mut row_weights = vec![0.0f64; count];
        for (x, row_weight) in row_weights.iter_mut().enumerate().take(count) {
            let weight = pillow_bicubic_filter((x as f64 + xmin as f64 - center + 0.5) * ss);
            *row_weight = weight;
            norm += weight;
        }

        let row = &mut coeffs[xx * ksize..(xx + 1) * ksize];
        for w in row.iter_mut() {
            *w = 0;
        }
        for (i, w) in row_weights.into_iter().enumerate() {
            let normalized = if norm != 0.0 { w / norm } else { 0.0 };
            row[i] = if normalized < 0.0 {
                (-0.5 + normalized * scale_int).trunc() as i32
            } else {
                (0.5 + normalized * scale_int).trunc() as i32
            };
        }
        bounds.push((xmin as usize, count));
    }

    PillowResampleCoeffs {
        ksize,
        bounds,
        coeffs,
    }
}

#[inline]
fn clip8(value: i64) -> u8 {
    value.clamp(0, 255) as u8
}

fn resize_rgb_image(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    ensure!(
        width > 0 && height > 0,
        "target dimensions must be positive"
    );
    if image.width() == width && image.height() == height {
        return Ok(image.clone());
    }

    let src_w = image.width() as usize;
    let src_h = image.height() as usize;
    let dst_w = width as usize;
    let dst_h = height as usize;
    let src = image.as_raw();

    let horiz = precompute_pillow_coeffs(src_w, dst_w);
    let vert = precompute_pillow_coeffs(src_h, dst_h);

    let rounding = 1i64 << (PILLOW_RESAMPLE_PRECISION_BITS - 1);

    let mut temp = vec![0u8; src_h * dst_w * 3];
    for y in 0..src_h {
        for x in 0..dst_w {
            let (xmin, count) = horiz.bounds[x];
            let k = &horiz.coeffs[x * horiz.ksize..(x + 1) * horiz.ksize];

            let mut s0 = rounding;
            let mut s1 = rounding;
            let mut s2 = rounding;

            for (tap, &coeff_i32) in k.iter().enumerate().take(count) {
                let coeff = i64::from(coeff_i32);
                let src_idx = (y * src_w + xmin + tap) * 3;
                s0 += i64::from(src[src_idx]) * coeff;
                s1 += i64::from(src[src_idx + 1]) * coeff;
                s2 += i64::from(src[src_idx + 2]) * coeff;
            }

            let dst_idx = (y * dst_w + x) * 3;
            temp[dst_idx] = clip8(s0 >> PILLOW_RESAMPLE_PRECISION_BITS);
            temp[dst_idx + 1] = clip8(s1 >> PILLOW_RESAMPLE_PRECISION_BITS);
            temp[dst_idx + 2] = clip8(s2 >> PILLOW_RESAMPLE_PRECISION_BITS);
        }
    }

    let mut out = vec![0u8; dst_h * dst_w * 3];
    for y in 0..dst_h {
        let (ymin, count) = vert.bounds[y];
        let k = &vert.coeffs[y * vert.ksize..(y + 1) * vert.ksize];

        for x in 0..dst_w {
            let mut s0 = rounding;
            let mut s1 = rounding;
            let mut s2 = rounding;

            for (tap, &coeff_i32) in k.iter().enumerate().take(count) {
                let coeff = i64::from(coeff_i32);
                let src_idx = ((ymin + tap) * dst_w + x) * 3;
                s0 += i64::from(temp[src_idx]) * coeff;
                s1 += i64::from(temp[src_idx + 1]) * coeff;
                s2 += i64::from(temp[src_idx + 2]) * coeff;
            }

            let dst_idx = (y * dst_w + x) * 3;
            out[dst_idx] = clip8(s0 >> PILLOW_RESAMPLE_PRECISION_BITS);
            out[dst_idx + 1] = clip8(s1 >> PILLOW_RESAMPLE_PRECISION_BITS);
            out[dst_idx + 2] = clip8(s2 >> PILLOW_RESAMPLE_PRECISION_BITS);
        }
    }

    RgbImage::from_raw(width, height, out)
        .ok_or_else(|| anyhow::anyhow!("failed to convert resized buffer into image"))
}

#[derive(Debug, Clone)]
struct VisionFrame {
    start: usize,
    len: usize,
}

#[derive(Debug, Clone)]
struct VisionSequenceLayout {
    frames: Vec<VisionFrame>,
    total_patches: usize,
    total_groups: usize,
}

impl VisionSequenceLayout {
    fn from_grids(grids: &[(usize, usize, usize)], merge: usize) -> Result<Self> {
        let mut frames = Vec::new();
        let mut total_patches = 0usize;
        let mut total_groups = 0usize;
        for &(t, h, w) in grids {
            ensure!(
                h.is_multiple_of(merge) && w.is_multiple_of(merge),
                "grid {}x{} not divisible by merge {}",
                h,
                w,
                merge
            );
            let per_frame = h * w;
            let groups_per_frame = (h / merge) * (w / merge);
            for _ in 0..t {
                frames.push(VisionFrame {
                    start: total_patches,
                    len: per_frame,
                });
                total_patches += per_frame;
                total_groups += groups_per_frame;
            }
        }
        Ok(Self {
            frames,
            total_patches,
            total_groups,
        })
    }
}

struct VisionRotaryEmbedding {
    rope_dim: usize,
    merge_size: usize,
    inv_freq: Vec<f32>,
    device: Device,
}

impl VisionRotaryEmbedding {
    fn new(cfg: &GlmOcrVisionConfig, device: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_heads;
        ensure!(
            head_dim.is_multiple_of(4),
            "vision head dim must be divisible by 4"
        );

        // HF reference constructs `GlmOcrVisionRotaryEmbedding(head_dim // 2)`,
        // then uses frequencies over even indices only.
        let rope_dim = head_dim / 2;
        let axis_freq_dim = rope_dim / 2;
        let mut inv_freq = Vec::with_capacity(axis_freq_dim);
        for i in 0..axis_freq_dim {
            let exponent = (i * 2) as f32 / rope_dim as f32;
            inv_freq.push(1.0f32 / 10_000f32.powf(exponent));
        }

        Ok(Self {
            rope_dim,
            merge_size: cfg.spatial_merge_size,
            inv_freq,
            device: device.clone(),
        })
    }

    fn build_embeddings(&self, grids: &[(usize, usize, usize)]) -> Result<(Tensor, Tensor)> {
        let mut pos_ids = Vec::new();
        let mut max_grid = 0usize;
        for &(t, h, w) in grids {
            max_grid = max_grid.max(h.max(w));
            let h_ids = grouped_axis_ids(h, w, self.merge_size, 0);
            let w_ids = grouped_axis_ids(h, w, self.merge_size, 1);
            ensure!(
                h_ids.len() == h * w && w_ids.len() == h * w,
                "axis id size mismatch"
            );
            for _ in 0..t {
                for idx in 0..(h * w) {
                    pos_ids.push((h_ids[idx], w_ids[idx]));
                }
            }
        }

        let axis_freq_dim = self.inv_freq.len();
        let mut full = Vec::with_capacity(max_grid * axis_freq_dim);
        for pos in 0..max_grid {
            let p = pos as f32;
            for &freq in &self.inv_freq {
                full.push(p * freq);
            }
        }

        // Equivalent to HF: rotary_pos_emb_full[pos_ids].flatten(1)
        // where pos_ids has shape [tokens, 2] (h_id, w_id).
        let mut rotary_vals = Vec::with_capacity(pos_ids.len() * self.rope_dim);
        for &(h_id, w_id) in &pos_ids {
            let h_off = h_id * axis_freq_dim;
            rotary_vals.extend_from_slice(&full[h_off..h_off + axis_freq_dim]);
            let w_off = w_id * axis_freq_dim;
            rotary_vals.extend_from_slice(&full[w_off..w_off + axis_freq_dim]);
        }

        let rotary = Tensor::from_vec(rotary_vals, (pos_ids.len(), self.rope_dim), &self.device)?;
        let emb = Tensor::cat(&[rotary.clone(), rotary], D::Minus1)?;
        Ok((emb.cos()?, emb.sin()?))
    }
}

fn grouped_axis_ids(height: usize, width: usize, merge: usize, axis: usize) -> Vec<usize> {
    let mut ids = Vec::with_capacity(height * width);
    for block_h in 0..(height / merge) {
        for block_w in 0..(width / merge) {
            for inner_h in 0..merge {
                for inner_w in 0..merge {
                    let h = block_h * merge + inner_h;
                    let w = block_w * merge + inner_w;
                    ids.push(if axis == 0 { h } else { w });
                }
            }
        }
    }
    ids
}

struct VisionPatchEmbed {
    weight: Tensor,
    bias: Option<Tensor>,
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
}

impl VisionPatchEmbed {
    fn load(cfg: &GlmOcrVisionConfig, vb: &VarBuilder) -> Result<Self> {
        let proj_vb = vb.pp("proj");
        let weight = proj_vb
            .get(
                (
                    cfg.hidden_size,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ),
                "weight",
            )
            .context("missing vision patch embed weight")?
            .reshape((
                cfg.hidden_size,
                cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size,
            ))?
            .contiguous()?;
        let bias = if proj_vb.contains_tensor("bias") {
            Some(
                proj_vb
                    .get(cfg.hidden_size, "bias")
                    .context("missing vision patch embed bias")?
                    .contiguous()?,
            )
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            in_channels: cfg.in_channels,
            embed_dim: cfg.hidden_size,
        })
    }

    fn forward(&self, flatten_patches: &Tensor) -> Result<Tensor> {
        let (_, cols) = flatten_patches.shape().dims2()?;
        let expected =
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        ensure!(
            cols == expected,
            "patch row width mismatch: {} vs {}",
            cols,
            expected
        );

        let compute_dtype = self.weight.dtype();
        let mut input = to_dtype_if_needed(flatten_patches, compute_dtype)?;
        input = input.contiguous()?;

        let mut weight_t = self.weight.transpose(0, 1)?;
        weight_t = into_dtype_if_needed(weight_t, compute_dtype)?;

        let mut out = input.matmul(&weight_t)?;
        if let Some(bias) = &self.bias {
            let bias = to_dtype_if_needed(bias, out.dtype())?;
            out = out.broadcast_add(&bias.reshape((1, self.embed_dim))?)?;
        }
        Ok(out)
    }
}

struct GlmVisionBlock {
    norm1: Tensor,
    norm2: Tensor,
    attn: GlmVisionAttention,
    mlp: GlmVisionMlp,
    eps: f64,
}

impl GlmVisionBlock {
    fn load(cfg: &GlmOcrVisionConfig, vb: &VarBuilder) -> Result<Self> {
        let norm1 = vb
            .pp("norm1")
            .get(cfg.hidden_size, "weight")
            .context("missing vision norm1 weight")?;
        let norm2 = vb
            .pp("norm2")
            .get(cfg.hidden_size, "weight")
            .context("missing vision norm2 weight")?;
        let attn = GlmVisionAttention::load(cfg, &vb.pp("attn"))?;
        let mlp = GlmVisionMlp::load(cfg, &vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        layout: &VisionSequenceLayout,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let attn_timer = Timer::new("vision.block.attn");
        let normed = precise_rms_norm_last_dim(hidden, &self.norm1, self.eps)
            .context("vision block norm1 failed")?;
        let attn = self.attn.forward(&normed, layout, cos, sin)?;
        attn_timer.finish(|_| {});
        let residual = hidden.add(&attn)?;

        let mlp_timer = Timer::new("vision.block.mlp");
        let normed = precise_rms_norm_last_dim(&residual, &self.norm2, self.eps)
            .context("vision block norm2 failed")?;
        let mlp = self.mlp.forward(&normed)?;
        mlp_timer.finish(|_| {});
        Ok(residual.add(&mlp)?)
    }
}

const VISION_ATTN_QUERY_CHUNK: usize = 1024;

struct GlmVisionAttention {
    qkv: LinearWeights,
    proj: LinearWeights,
    q_norm: Tensor,
    k_norm: Tensor,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl GlmVisionAttention {
    fn load(cfg: &GlmOcrVisionConfig, vb: &VarBuilder) -> Result<Self> {
        let qkv = LinearWeights::load(
            vb.pp("qkv"),
            cfg.hidden_size * 3,
            cfg.hidden_size,
            cfg.attention_bias,
        )?;
        let proj = LinearWeights::load(
            vb.pp("proj"),
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.attention_bias,
        )?;
        let q_norm = vb
            .pp("q_norm")
            .get(cfg.hidden_size / cfg.num_heads, "weight")
            .context("missing vision q_norm weight")?;
        let k_norm = vb
            .pp("k_norm")
            .get(cfg.hidden_size / cfg.num_heads, "weight")
            .context("missing vision k_norm weight")?;
        Ok(Self {
            qkv,
            proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_heads,
            head_dim: cfg.hidden_size / cfg.num_heads,
            scaling: (cfg.hidden_size / cfg.num_heads) as f64,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        layout: &VisionSequenceLayout,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let qkv = self.qkv.forward_2d(hidden)?;
        let (tokens, width) = qkv.shape().dims2()?;
        ensure!(
            width == self.num_heads * self.head_dim * 3,
            "unexpected qkv width"
        );
        let reshaped = qkv.reshape((tokens, 3, self.num_heads, self.head_dim))?;
        let q = reshaped
            .narrow(1, 0, 1)?
            .reshape((tokens, self.num_heads, self.head_dim))?;
        let k = reshaped
            .narrow(1, 1, 1)?
            .reshape((tokens, self.num_heads, self.head_dim))?;
        let v = reshaped
            .narrow(1, 2, 1)?
            .reshape((tokens, self.num_heads, self.head_dim))?;

        let q = precise_rms_norm_last_dim(&q, &self.q_norm, 1e-5)?;
        let k = precise_rms_norm_last_dim(&k, &self.k_norm, 1e-5)?;
        let (q, k) = apply_vision_rotary(&q, &k, cos, sin)?;

        let mut outputs = Vec::with_capacity(layout.frames.len());
        for frame in &layout.frames {
            let qf = q.narrow(0, frame.start, frame.len)?;
            let kf = k.narrow(0, frame.start, frame.len)?;
            let vf = v.narrow(0, frame.start, frame.len)?;
            // Candle bmm requires contiguous RHS. Keep RHS contiguous while
            // preserving non-contiguous lhs views to avoid extra copies.
            let qh = qf.transpose(0, 1)?;
            let kh = kf.transpose(0, 1)?;
            let vh = vf.transpose(0, 1)?.contiguous()?;
            let kt = kh.transpose(1, 2)?.contiguous()?;
            let compute_dtype = match qh.dtype() {
                DType::F16 | DType::BF16 => DType::F32,
                d => d,
            };
            let qh = into_dtype_if_needed(qh, compute_dtype)?;
            let kt = into_dtype_if_needed(kt, compute_dtype)?;
            let vh = into_dtype_if_needed(vh, compute_dtype)?;
            let qh = (qh * (1.0f64 / self.scaling.sqrt()))?;

            // Softmax is row-wise over keys; chunking queries keeps numerics while
            // avoiding large [heads, q_len, k_len] score allocations on Metal/CPU.
            let mut ctx_chunks = Vec::new();
            for q_start in (0..frame.len).step_by(VISION_ATTN_QUERY_CHUNK) {
                let q_chunk_len = (frame.len - q_start).min(VISION_ATTN_QUERY_CHUNK);
                let q_chunk = qh.narrow(1, q_start, q_chunk_len)?;
                let scores = q_chunk.matmul(&kt)?;
                let probs = softmax_last_dim(&scores)?;
                let ctx_chunk = into_dtype_if_needed(probs.matmul(&vh)?, hidden.dtype())?;
                ctx_chunks.push(ctx_chunk);
            }

            let ctx = if ctx_chunks.len() == 1 {
                ctx_chunks.pop().expect("ctx_chunks has one element")
            } else {
                let chunk_refs: Vec<&Tensor> = ctx_chunks.iter().collect();
                Tensor::cat(&chunk_refs, 1)?
            };
            let ctx = ctx
                .transpose(0, 1)?
                .reshape((frame.len, self.num_heads * self.head_dim))?;
            outputs.push(ctx);
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        let merged = Tensor::cat(&refs, 0)?;
        self.proj.forward_2d(&merged)
    }
}

fn apply_vision_rotary(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_dtype = q.dtype();
    let k_dtype = k.dtype();
    let qf = q.to_dtype(DType::F32)?;
    let kf = k.to_dtype(DType::F32)?;
    let cos = cos.unsqueeze(1)?.to_dtype(DType::F32)?;
    let sin = sin.unsqueeze(1)?.to_dtype(DType::F32)?;
    let q_rot = qf
        .broadcast_mul(&cos)?
        .add(&rotate_half_last_dim(&qf)?.broadcast_mul(&sin)?)?;
    let k_rot = kf
        .broadcast_mul(&cos)?
        .add(&rotate_half_last_dim(&kf)?.broadcast_mul(&sin)?)?;
    Ok((q_rot.to_dtype(q_dtype)?, k_rot.to_dtype(k_dtype)?))
}

fn rotate_half_last_dim(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    ensure!(last % 2 == 0, "rotate_half expects even dim, got {last}");
    let half = last / 2;
    let left = x.narrow(D::Minus1, 0, half)?;
    let right = x.narrow(D::Minus1, half, half)?;
    Ok(Tensor::cat(&[right.neg()?, left], D::Minus1)?)
}

struct GlmVisionMlp {
    gate: LinearWeights,
    up: LinearWeights,
    down: LinearWeights,
}

impl GlmVisionMlp {
    fn load(cfg: &GlmOcrVisionConfig, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: LinearWeights::load(
                vb.pp("gate_proj"),
                cfg.intermediate_size,
                cfg.hidden_size,
                cfg.attention_bias,
            )?,
            up: LinearWeights::load(
                vb.pp("up_proj"),
                cfg.intermediate_size,
                cfg.hidden_size,
                cfg.attention_bias,
            )?,
            down: LinearWeights::load(
                vb.pp("down_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.attention_bias,
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward_2d(hidden)?.silu()?;
        let up = self.up.forward_2d(hidden)?;
        let fused = gate.broadcast_mul(&up)?;
        self.down.forward_2d(&fused)
    }
}

struct VisionPatchMerger {
    proj: LinearWeights,
    norm: LayerNorm,
    gate: LinearWeights,
    up: LinearWeights,
    down: LinearWeights,
}

impl VisionPatchMerger {
    fn load(cfg: &GlmOcrVisionConfig, vb: &VarBuilder, _dtype: DType) -> Result<Self> {
        let norm = layer_norm(cfg.out_hidden_size, 1e-5, vb.pp("post_projection_norm"))?;
        Ok(Self {
            proj: LinearWeights::load(
                vb.pp("proj"),
                cfg.out_hidden_size,
                cfg.out_hidden_size,
                false,
            )?,
            norm,
            gate: LinearWeights::load(
                vb.pp("gate_proj"),
                cfg.out_hidden_size * cfg.in_channels,
                cfg.out_hidden_size,
                false,
            )?,
            up: LinearWeights::load(
                vb.pp("up_proj"),
                cfg.out_hidden_size * cfg.in_channels,
                cfg.out_hidden_size,
                false,
            )?,
            down: LinearWeights::load(
                vb.pp("down_proj"),
                cfg.out_hidden_size,
                cfg.out_hidden_size * cfg.in_channels,
                false,
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let projected = self.proj.forward_2d(hidden)?;
        let normalized = self.norm.forward(&projected)?;
        let activated = normalized.gelu_erf()?;
        let gate = self.gate.forward_2d(&activated)?.silu()?;
        let up = self.up.forward_2d(&activated)?;
        let fused = gate.broadcast_mul(&up)?;
        self.down.forward_2d(&fused)
    }
}

fn precise_rms_norm_last_dim(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let input_dtype = input.dtype();
    let x = to_dtype_if_needed(input, DType::F32)?;
    let hidden = x.dim(D::Minus1)?;
    let variance = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
    let inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x.broadcast_mul(&inv)?;
    let weight = to_dtype_if_needed(weight, DType::F32)?;
    let out = normed.broadcast_mul(&weight)?;
    into_dtype_if_needed(out, input_dtype)
}
