use std::collections::BTreeSet;

use image::{DynamicImage, GenericImageView, RgbImage};

use super::resample::resize_bicubic;

#[derive(Debug, Clone)]
pub struct DynamicPreprocessResult {
    pub tiles: Vec<DynamicImage>,
    pub ratio: (u32, u32),
}

pub fn dynamic_preprocess(
    image: &DynamicImage,
    min_num: u32,
    max_num: u32,
    image_size: u32,
    use_thumbnail: bool,
) -> DynamicPreprocessResult {
    let (orig_width, orig_height) = image.dimensions();
    let aspect_ratio = orig_width as f64 / orig_height as f64;

    let mut target_ratios: BTreeSet<(u32, u32)> = BTreeSet::new();
    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                if i * j <= max_num && i * j >= min_num {
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
        } else if (ratio_diff - best_ratio_diff).abs() < f64::EPSILON {
            if area > 0.5f64 * (image_size * image_size * *w_ratio * *h_ratio) as f64 {
                target_aspect_ratio = (*w_ratio, *h_ratio);
            }
        }
    }

    let target_width = image_size * target_aspect_ratio.0;
    let target_height = image_size * target_aspect_ratio.1;
    let base_rgb: RgbImage = image.to_rgb8();
    let resized_rgb = resize_bicubic(&base_rgb, target_width, target_height);
    let resized = DynamicImage::ImageRgb8(resized_rgb);

    let mut tiles = Vec::new();
    let tiles_w = target_width / image_size;
    let tiles_h = target_height / image_size;
    for i in 0..tiles_w * tiles_h {
        let x = (i % tiles_w) * image_size;
        let y = (i / tiles_w) * image_size;
        let tile = resized.crop_imm(x, y, image_size, image_size);
        tiles.push(tile);
    }

    if use_thumbnail && tiles.len() > 1 {
        let thumb_rgb = resize_bicubic(&base_rgb, image_size, image_size);
        tiles.push(DynamicImage::ImageRgb8(thumb_rgb));
    }

    DynamicPreprocessResult {
        tiles,
        ratio: target_aspect_ratio,
    }
}
