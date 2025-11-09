use anyhow::{Context, Result, anyhow};
use candle_core::{Device, Tensor};
use fast_image_resize as fir;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

use crate::config::PaddleOcrVisionConfig;

pub const DEFAULT_MIN_PIXELS: usize = 147_384;
pub const DEFAULT_MAX_PIXELS: usize = 2_822_400;
pub const DEFAULT_IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const DEFAULT_IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

#[derive(Debug, Clone)]
pub struct SiglipPreprocessConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f32,
}

impl SiglipPreprocessConfig {
    pub fn from_vision_config(cfg: &PaddleOcrVisionConfig) -> Self {
        // The official PaddleOCR-VL processor keeps `temporal_patch_size = 1` for single
        // images even though the exported config reports `temporal_patch_size = 2`. Using
        // the raw config value would double the SigLIP token budget and break parity, so
        // we intentionally clamp the temporal dimension to one here.
        Self {
            patch_size: cfg.patch_size,
            merge_size: cfg.spatial_merge_size,
            temporal_patch_size: 1,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
            image_mean: DEFAULT_IMAGE_MEAN,
            image_std: DEFAULT_IMAGE_STD,
            rescale_factor: 1.0 / 255.0,
        }
    }

    pub fn with_max_image_size(mut self, image_size: u32) -> Self {
        if image_size > 0 {
            let max_pixels = (image_size as usize).saturating_mul(image_size as usize);
            self.max_pixels = self.max_pixels.min(max_pixels.max(self.min_pixels));
        }
        self
    }

    pub fn with_min_max(mut self, min_pixels: usize, max_pixels: usize) -> Self {
        self.min_pixels = min_pixels;
        self.max_pixels = max_pixels;
        self
    }

    pub fn with_normalization(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.image_mean = mean;
        self.image_std = std;
        self
    }
}

impl Default for SiglipPreprocessConfig {
    fn default() -> Self {
        Self {
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 1,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
            image_mean: DEFAULT_IMAGE_MEAN,
            image_std: DEFAULT_IMAGE_STD,
            rescale_factor: 1.0 / 255.0,
        }
    }
}

#[derive(Debug)]
pub struct SiglipImagePatches {
    pub patches: Tensor,
    pub grid_thw: (usize, usize, usize),
    pub height: usize,
    pub width: usize,
    pub position_ids: Vec<i64>,
    pub height_ids: Vec<i64>,
    pub width_ids: Vec<i64>,
}

pub fn preprocess_image(
    image: &DynamicImage,
    device: &Device,
    config: &SiglipPreprocessConfig,
) -> Result<SiglipImagePatches> {
    let rgb = image.to_rgb8();
    let (orig_width, orig_height) = rgb.dimensions();
    let factor = (config.patch_size * config.merge_size) as u32;
    let (resized_height, resized_width) = smart_resize(
        orig_height,
        orig_width,
        factor,
        config.min_pixels as u32,
        config.max_pixels as u32,
    )?;

    let resized: RgbImage = if (orig_width, orig_height) == (resized_width, resized_height) {
        rgb
    } else {
        resize_rgb_image(&rgb, resized_width, resized_height)?
    };

    let normalized = normalise_rgb(&resized, config)?;
    let (grid_t, grid_h, grid_w) =
        compute_grids(resized_height as usize, resized_width as usize, config)?;
    let patch_vec = patches_from_normalised(
        &normalized,
        resized_width as usize,
        resized_height as usize,
        config,
    )?;
    let data = maybe_tile_temporal(patch_vec, config.temporal_patch_size);
    let num_patches = grid_t * grid_h * grid_w;
    let channels = 3usize;
    let dims = (num_patches, channels, config.patch_size, config.patch_size);
    let tensor = Tensor::from_vec(data, dims, device)?;

    let (position_ids, height_ids, width_ids) = build_position_metadata((grid_t, grid_h, grid_w));

    Ok(SiglipImagePatches {
        patches: tensor,
        grid_thw: (grid_t, grid_h, grid_w),
        height: resized_height as usize,
        width: resized_width as usize,
        position_ids,
        height_ids,
        width_ids,
    })
}

fn normalise_rgb(
    image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    config: &SiglipPreprocessConfig,
) -> Result<Vec<f32>> {
    let rescale = config.rescale_factor;
    let mean = config.image_mean;
    let std = config.image_std;
    if mean.iter().any(|&m| !m.is_finite()) || std.iter().any(|&s| s <= 0.0 || !s.is_finite()) {
        return Err(anyhow!("invalid mean/std for normalisation"));
    }
    let mut data = Vec::with_capacity((image.width() * image.height() * 3) as usize);
    for pixel in image.pixels() {
        let channels = pixel.0;
        for (idx, &value) in channels.iter().enumerate() {
            let scaled = (value as f32) * rescale;
            let normalised = (scaled - mean[idx]) / std[idx];
            data.push(normalised);
        }
    }
    Ok(data)
}

fn compute_grids(
    height: usize,
    width: usize,
    config: &SiglipPreprocessConfig,
) -> Result<(usize, usize, usize)> {
    let patch = config.patch_size;
    anyhow::ensure!(
        height % patch == 0 && width % patch == 0,
        "resized dimensions ({height}, {width}) not divisible by patch size {patch}"
    );
    let grid_h = height / patch;
    let grid_w = width / patch;
    let grid_t = config.temporal_patch_size.max(1);
    Ok((grid_t, grid_h, grid_w))
}

fn patches_from_normalised(
    data: &[f32],
    width: usize,
    height: usize,
    config: &SiglipPreprocessConfig,
) -> Result<Vec<f32>> {
    let patch = config.patch_size;
    let channels = 3usize;
    let row_stride = width * channels;
    let mut patches = Vec::with_capacity(width * height * channels);
    let grid_h = height / patch;
    let grid_w = width / patch;
    for gh in 0..grid_h {
        for gw in 0..grid_w {
            for channel in 0..channels {
                for py in 0..patch {
                    for px in 0..patch {
                        let y = gh * patch + py;
                        let x = gw * patch + px;
                        let idx = y * row_stride + x * channels + channel;
                        patches.push(data[idx]);
                    }
                }
            }
        }
    }
    Ok(patches)
}

fn maybe_tile_temporal(data: Vec<f32>, temporal: usize) -> Vec<f32> {
    if temporal <= 1 {
        return data;
    }
    let per_frame = data.len();
    let mut all = Vec::with_capacity(per_frame * temporal);
    for _ in 0..temporal {
        all.extend_from_slice(&data);
    }
    all
}

fn build_position_metadata(grid: (usize, usize, usize)) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (t, h, w) = grid;
    let total = t * h * w;
    let mut position_ids = Vec::with_capacity(total);
    let mut height_ids = Vec::with_capacity(total);
    let mut width_ids = Vec::with_capacity(total);
    for _frame in 0..t {
        for row in 0..h {
            for col in 0..w {
                position_ids.push((row * w + col) as i64);
                height_ids.push(row as i64);
                width_ids.push(col as i64);
            }
        }
    }
    (position_ids, height_ids, width_ids)
}

fn resize_rgb_image(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    if width == 0 || height == 0 {
        return Err(anyhow!("target dimensions must be positive"));
    }
    let mut owned = image.clone().into_raw();
    let src = fir::images::Image::from_slice_u8(
        image.width(),
        image.height(),
        owned.as_mut_slice(),
        fir::PixelType::U8x3,
    )
    .context("failed to build fast_image_resize source image")?;
    let mut dst = fir::images::Image::new(width, height, fir::PixelType::U8x3);
    let options = fir::ResizeOptions::new()
        .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom));
    let mut resizer = fir::Resizer::new();
    resizer
        .resize(&src, &mut dst, &options)
        .map_err(|err| anyhow!("fast_image_resize failed: {err}"))?;
    let buffer = dst.into_vec();
    ImageBuffer::from_raw(width, height, buffer)
        .ok_or_else(|| anyhow!("failed to convert resized buffer into image"))
}

pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32)> {
    let factor = factor.max(1) as f64;
    let mut h = height.max(1) as f64;
    let mut w = width.max(1) as f64;
    if h < factor {
        w = ((w * factor) / h).round();
        h = factor;
    }
    if w < factor {
        h = ((h * factor) / w).round();
        w = factor;
    }
    let aspect = h.max(w) / h.min(w);
    anyhow::ensure!(aspect <= 200.0, "aspect ratio exceeds limit ({aspect})");
    let mut h_bar = (h / factor).round() * factor;
    let mut w_bar = (w / factor).round() * factor;
    let area = h_bar * w_bar;
    let max_pixels = max_pixels.max(1) as f64;
    let min_pixels = min_pixels.max(1) as f64;
    if area > max_pixels {
        let beta = ((h * w) / max_pixels).sqrt();
        h_bar = ((h / beta) / factor).floor() * factor;
        w_bar = ((w / beta) / factor).floor() * factor;
    } else if area < min_pixels {
        let beta = (min_pixels / (h * w)).sqrt();
        h_bar = ((h * beta) / factor).ceil() * factor;
        w_bar = ((w * beta) / factor).ceil() * factor;
    }
    anyhow::ensure!(
        h_bar >= factor && w_bar >= factor,
        "resized dimensions smaller than factor"
    );
    Ok((h_bar as u32, w_bar as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use image::RgbImage;

    #[test]
    fn resize_preserves_factor() {
        let factor = 28;
        let (h, w) = smart_resize(
            320,
            512,
            factor,
            DEFAULT_MIN_PIXELS as u32,
            DEFAULT_MAX_PIXELS as u32,
        )
        .unwrap();
        assert_eq!(h % factor, 0);
        assert_eq!(w % factor, 0);
    }

    #[test]
    fn preprocess_constant_image() {
        let mut img = RgbImage::new(28, 28);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([128, 128, 128]);
        }
        let dyn_image = DynamicImage::ImageRgb8(img);
        let device = Device::Cpu;
        let config = SiglipPreprocessConfig {
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 1,
            min_pixels: 28 * 28,
            max_pixels: 28 * 28,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            rescale_factor: 1.0 / 255.0,
        };
        let patches = preprocess_image(&dyn_image, &device, &config).unwrap();
        assert_eq!(patches.grid_thw, (1, 2, 2));
        assert_eq!(patches.height, 28);
        assert_eq!(patches.width, 28);
        let (n, c, h, w) = patches.patches.dims4().unwrap();
        assert_eq!((n, c, h, w), (4, 3, 14, 14));
        let sum = patches
            .patches
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let mean_val = sum / (4.0 * 3.0 * 196.0);
        let expected = ((128.0 / 255.0) - 0.5) / 0.5;
        assert!(
            (mean_val - expected).abs() < 1e-6,
            "mean value {mean_val}, expected {expected}"
        );
    }
}
