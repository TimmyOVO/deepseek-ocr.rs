use candle_core::Device;
use deepseek_ocr_infer_dots::vision::preprocess::{DotsPreprocessConfig, preprocess_image};
use image::{DynamicImage, Rgb, RgbImage};

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
    let (n, c, h, w) = out.pixel_values.dims4().expect("4d tensor");
    assert_eq!((n, c, h, w), (4, 3, 14, 14));
    let val = out
        .pixel_values
        .sum_all()
        .expect("sum")
        .to_scalar::<f32>()
        .expect("scalar");
    let avg = val / (4.0 * 3.0 * 196.0);
    let expected = ((128.0 / 255.0) - 0.5) / 0.5;
    assert!(
        (avg - expected).abs() < 1e-6,
        "avg={avg} expected={expected}"
    );
}
