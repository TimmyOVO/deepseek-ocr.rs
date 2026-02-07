use std::collections::BTreeSet;

use image::{DynamicImage, GenericImageView, RgbImage};

use super::resample::resize_bicubic;

#[derive(Debug, Clone, Copy)]
pub struct PreprocessParams {
    pub tile_size: u32,
    pub base_size: u32,
    pub min_num: u32,
    pub max_num: u32,
    pub small_image_no_crop_threshold: Option<u32>,
}

impl PreprocessParams {
    pub fn ocr1(base_size: u32, tile_size: u32) -> Self {
        Self {
            tile_size,
            base_size,
            min_num: 2,
            max_num: 9,
            small_image_no_crop_threshold: Some(tile_size),
        }
    }

    pub fn ocr2(base_size: u32, tile_size: u32) -> Self {
        Self {
            tile_size,
            base_size,
            min_num: 2,
            max_num: 6,
            small_image_no_crop_threshold: Some(tile_size),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicPreprocessResult {
    pub tiles: Vec<DynamicImage>,
    pub ratio: (u32, u32),
}

impl DynamicPreprocessResult {
    pub fn grid(&self) -> (u32, u32) {
        self.ratio
    }
}

pub fn dynamic_preprocess(
    image: &DynamicImage,
    min_num: u32,
    max_num: u32,
    image_size: u32,
    use_thumbnail: bool,
) -> DynamicPreprocessResult {
    let params = PreprocessParams {
        tile_size: image_size,
        base_size: image_size,
        min_num,
        max_num,
        small_image_no_crop_threshold: None,
    };
    dynamic_preprocess_with_params(image, &params, use_thumbnail)
}

pub fn dynamic_preprocess_with_params(
    image: &DynamicImage,
    params: &PreprocessParams,
    use_thumbnail: bool,
) -> DynamicPreprocessResult {
    let (orig_width, orig_height) = image.dimensions();
    if let Some(threshold) = params.small_image_no_crop_threshold
        && orig_width <= threshold
        && orig_height <= threshold
    {
        return DynamicPreprocessResult {
            tiles: Vec::new(),
            ratio: (1, 1),
        };
    }

    let aspect_ratio = orig_width as f64 / orig_height as f64;

    let mut target_ratios: BTreeSet<(u32, u32)> = BTreeSet::new();
    for n in params.min_num..=params.max_num {
        for i in 1..=n {
            for j in 1..=n {
                if i * j <= params.max_num && i * j >= params.min_num {
                    target_ratios.insert((i, j));
                }
            }
        }
    }

    let mut target_aspect_ratio = (1, 1);
    let mut best_ratio_diff = f64::MAX;
    let area = (orig_width * orig_height) as f64;

    for (w_ratio, h_ratio) in &target_ratios {
        let target_ratio = *w_ratio as f64 / *h_ratio as f64;
        let ratio_diff = (aspect_ratio - target_ratio).abs();
        if ratio_diff < best_ratio_diff {
            best_ratio_diff = ratio_diff;
            target_aspect_ratio = (*w_ratio, *h_ratio);
        } else if (ratio_diff - best_ratio_diff).abs() < f64::EPSILON
            && area > 0.5f64 * (params.tile_size * params.tile_size * *w_ratio * *h_ratio) as f64
        {
            target_aspect_ratio = (*w_ratio, *h_ratio);
        }
    }

    let target_width = params.tile_size * target_aspect_ratio.0;
    let target_height = params.tile_size * target_aspect_ratio.1;
    let base_rgb: RgbImage = image.to_rgb8();
    let resized_rgb = resize_bicubic(&base_rgb, target_width, target_height);
    let resized = DynamicImage::ImageRgb8(resized_rgb);

    let mut tiles = Vec::new();
    let tiles_w = target_width / params.tile_size;
    let tiles_h = target_height / params.tile_size;
    for i in 0..tiles_w * tiles_h {
        let x = (i % tiles_w) * params.tile_size;
        let y = (i / tiles_w) * params.tile_size;
        let tile = resized.crop_imm(x, y, params.tile_size, params.tile_size);
        tiles.push(tile);
    }

    if use_thumbnail && tiles.len() > 1 {
        let thumb_rgb = resize_bicubic(&base_rgb, params.tile_size, params.tile_size);
        tiles.push(DynamicImage::ImageRgb8(thumb_rgb));
    }

    DynamicPreprocessResult {
        tiles,
        ratio: target_aspect_ratio,
    }
}
