use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use candle_core::{DType, Device, Tensor};
use fast_image_resize as fir;
use image::{DynamicImage, ImageBuffer, RgbImage};
use serde::Deserialize;

/// Parameters governing how raw images are resized and normalised prior to patchification.
#[derive(Debug, Clone)]
pub struct DotsPreprocessConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl DotsPreprocessConfig {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let owned;
        let path = match path {
            Some(path) => path,
            None => {
                owned = default_preprocess_path();
                &owned
            }
        };
        let bytes = std::fs::read(path).with_context(|| {
            format!(
                "failed to read dots.ocr preprocessor_config.json from {}",
                path.display()
            )
        })?;
        let raw: RawPreprocessConfig = serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "failed to parse dots.ocr preprocessor_config.json at {}",
                path.display()
            )
        })?;
        Ok(raw.into())
    }

    pub fn factor(&self) -> u32 {
        (self.patch_size * self.merge_size) as u32
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawPreprocessConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    #[serde(alias = "merge_size", alias = "spatial_merge_size")]
    pub merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl From<RawPreprocessConfig> for DotsPreprocessConfig {
    fn from(value: RawPreprocessConfig) -> Self {
        Self {
            patch_size: value.patch_size,
            temporal_patch_size: value.temporal_patch_size,
            merge_size: value.merge_size,
            min_pixels: value.min_pixels,
            max_pixels: value.max_pixels,
            image_mean: value.image_mean,
            image_std: value.image_std,
        }
    }
}

fn default_preprocess_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dots.ocr/preprocessor_config.json")
}

#[derive(Debug)]
pub struct DotsPreprocessResult {
    pub pixel_values: Tensor,
    pub grid_thw: [u32; 3],
    pub height: u32,
    pub width: u32,
}

pub fn preprocess_image(
    image: &DynamicImage,
    device: &Device,
    cfg: &DotsPreprocessConfig,
) -> Result<DotsPreprocessResult> {
    let rgb = image.to_rgb8();
    let (orig_w, orig_h) = rgb.dimensions();
    let factor = cfg.factor();
    let (resized_h, resized_w) = smart_resize(
        orig_h,
        orig_w,
        factor,
        cfg.min_pixels as u32,
        cfg.max_pixels as u32,
    )?;
    let resized: RgbImage = if (orig_w, orig_h) == (resized_w, resized_h) {
        rgb
    } else {
        resize_rgb_image(&rgb, resized_w, resized_h)?
    };

    let normalised = normalise_rgb(&resized, cfg)?;
    let grid_h = (resized_h as usize) / cfg.patch_size;
    let grid_w = (resized_w as usize) / cfg.patch_size;
    ensure!(
        grid_h > 0 && grid_w > 0,
        "invalid resized dimensions: {}x{}",
        resized_h,
        resized_w
    );
    let grid_t = cfg.temporal_patch_size.max(1);
    let patches =
        patches_from_normalised(&normalised, resized_w as usize, resized_h as usize, cfg)?;
    let tiled = maybe_tile_temporal(patches, grid_t);
    let patch_shape = (grid_t * grid_h * grid_w, 3, cfg.patch_size, cfg.patch_size);
    let tensor = Tensor::from_vec(tiled, patch_shape, device)?;
    Ok(DotsPreprocessResult {
        pixel_values: tensor,
        grid_thw: [grid_t as u32, grid_h as u32, grid_w as u32],
        height: resized_h,
        width: resized_w,
    })
}

pub fn preprocess_images(
    images: &[DynamicImage],
    device: &Device,
    cfg: &DotsPreprocessConfig,
) -> Result<(Tensor, Vec<[u32; 3]>)> {
    if images.is_empty() {
        let zeros = Tensor::zeros((0, 3, cfg.patch_size, cfg.patch_size), DType::F32, device)?;
        return Ok((zeros, Vec::new()));
    }

    let mut tensors = Vec::with_capacity(images.len());
    let mut grids = Vec::with_capacity(images.len());
    for image in images {
        let result = preprocess_image(image, device, cfg)?;
        grids.push(result.grid_thw);
        tensors.push(result.pixel_values);
    }
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let pixel_values = if refs.len() == 1 {
        tensors.into_iter().next().unwrap()
    } else {
        Tensor::cat(&refs, 0)?
    };
    Ok((pixel_values, grids))
}

fn normalise_rgb(image: &RgbImage, cfg: &DotsPreprocessConfig) -> Result<Vec<f32>> {
    let rescale = 1.0 / 255.0;
    let mut data = Vec::with_capacity((image.width() * image.height() * 3) as usize);
    for pixel in image.pixels() {
        for (idx, &value) in pixel.0.iter().enumerate() {
            let scaled = (value as f32) * rescale;
            let normalised = (scaled - cfg.image_mean[idx]) / cfg.image_std[idx];
            if !normalised.is_finite() {
                return Err(anyhow!("normalised pixel is not finite"));
            }
            data.push(normalised);
        }
    }
    Ok(data)
}

fn patches_from_normalised(
    data: &[f32],
    width: usize,
    height: usize,
    cfg: &DotsPreprocessConfig,
) -> Result<Vec<f32>> {
    let patch = cfg.patch_size;
    ensure!(width % patch == 0 && height % patch == 0);
    let channels = 3usize;
    let row_stride = width * channels;
    let mut patches = Vec::with_capacity(width * height * channels);
    let grid_h = height / patch;
    let grid_w = width / patch;
    ensure!(
        grid_h % cfg.merge_size == 0 && grid_w % cfg.merge_size == 0,
        "grid {}x{} not divisible by merge size {}",
        grid_h,
        grid_w,
        cfg.merge_size
    );
    let blocks_h = grid_h / cfg.merge_size;
    let blocks_w = grid_w / cfg.merge_size;
    for bh in 0..blocks_h {
        for bw in 0..blocks_w {
            for ih in 0..cfg.merge_size {
                for iw in 0..cfg.merge_size {
                    let gh = bh * cfg.merge_size + ih;
                    let gw = bw * cfg.merge_size + iw;
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
        }
    }
    Ok(patches)
}
fn maybe_tile_temporal(data: Vec<f32>, temporal: usize) -> Vec<f32> {
    if temporal <= 1 {
        return data;
    }
    let per = data.len();
    let mut all = Vec::with_capacity(per * temporal);
    for _ in 0..temporal {
        all.extend_from_slice(&data);
    }
    all
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
    ensure!(aspect <= 200.0, "aspect ratio exceeds limit ({aspect})");
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
    ensure!(h_bar >= factor && w_bar >= factor);
    Ok((h_bar as u32, w_bar as u32))
}

fn resize_rgb_image(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    ensure!(width > 0 && height > 0);
    let mut src_buf = image.clone().into_raw();
    let src = fir::images::Image::from_slice_u8(
        image.width(),
        image.height(),
        src_buf.as_mut_slice(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use image::Rgb;

    #[test]
    fn load_default_preprocess_config() {
        let cfg = DotsPreprocessConfig::load(None).expect("config available");
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.temporal_patch_size, 1);
        assert_eq!(cfg.merge_size, 2);
        assert_eq!(cfg.min_pixels, 3136);
        assert_eq!(cfg.max_pixels, 11_289_600);
    }

    #[test]
    fn preprocess_constant_image() {
        let cfg = DotsPreprocessConfig {
            patch_size: 14,
            temporal_patch_size: 1,
            merge_size: 2,
            min_pixels: 28 * 28,
            max_pixels: 28 * 28,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        let mut img = RgbImage::new(28, 28);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([128, 128, 128]);
        }
        let dyn_image = DynamicImage::ImageRgb8(img);
        let device = Device::Cpu;
        let out = preprocess_image(&dyn_image, &device, &cfg).expect("preprocess works");
        assert_eq!(out.grid_thw, [1, 2, 2]);
        let (n, c, h, w) = out.pixel_values.dims4().unwrap();
        assert_eq!((n, c, h, w), (4, 3, 14, 14));
        let val = out
            .pixel_values
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let avg = val / (4.0 * 3.0 * 196.0);
        let expected = ((128.0 / 255.0) - 0.5) / 0.5;
        assert!(
            (avg - expected).abs() < 1e-6,
            "avg={avg} expected={expected}"
        );
    }
}
