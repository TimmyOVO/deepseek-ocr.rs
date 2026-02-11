use candle_core::Device;
use deepseek_ocr_infer_paddleocr::vision::{SiglipPreprocessConfig, preprocess_image, smart_resize};
use image::{DynamicImage, Rgb, RgbImage};

const DEFAULT_MIN_PIXELS: usize = 147_384;
const DEFAULT_MAX_PIXELS: usize = 2_822_400;

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
    .expect("resize");
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
    let patches = preprocess_image(&dyn_image, &device, &config).expect("preprocess");
    assert_eq!(patches.grid_thw, (1, 2, 2));
    assert_eq!(patches.height, 28);
    assert_eq!(patches.width, 28);
    let (n, c, h, w) = patches.patches.dims4().expect("dims");
    assert_eq!((n, c, h, w), (4, 3, 14, 14));
    let sum = patches
        .patches
        .sum_all()
        .expect("sum")
        .to_scalar::<f32>()
        .expect("scalar");
    let mean_val = sum / (4.0 * 3.0 * 196.0);
    let expected = ((128.0 / 255.0) - 0.5) / 0.5;
    assert!(
        (mean_val - expected).abs() < 1e-6,
        "mean value {mean_val}, expected {expected}"
    );
}
