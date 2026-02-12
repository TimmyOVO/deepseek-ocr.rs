use std::sync::Arc;

use candle_core::{DType, Device};
use deepseek_ocr_config::{AppConfig, InferenceOverride, InferenceSettings, LocalFileSystem};
use deepseek_ocr_server::state::AppState;

fn build_state(
    base_inference: InferenceSettings,
    inference_overrides: InferenceOverride,
) -> AppState {
    let fs = LocalFileSystem::new("deepseek-ocr-server-tests");
    let config = Arc::new(AppConfig::default());
    AppState::bootstrap(
        fs,
        config,
        Device::Cpu,
        DType::F32,
        base_inference,
        inference_overrides,
    )
    .expect("bootstrap state")
}

#[test]
fn ocr2_uses_its_model_default_image_size() {
    let base = InferenceSettings {
        image_size: 640,
        ..InferenceSettings::default()
    };
    let state = build_state(base, InferenceOverride::default());

    let (vision, _) = state
        .per_model_inference_settings("deepseek-ocr-2")
        .expect("resolve ocr2 settings");
    assert_eq!(vision.base_size, 1024);
    assert_eq!(vision.image_size, 768);
    assert!(vision.crop_mode);
}

#[test]
fn server_cli_overrides_model_defaults() {
    let base = InferenceSettings::default();
    let overrides = InferenceOverride {
        image_size: Some(896),
        base_size: Some(960),
        ..InferenceOverride::default()
    };
    let state = build_state(base, overrides);

    let (vision, _) = state
        .per_model_inference_settings("deepseek-ocr-2")
        .expect("resolve overridden settings");
    assert_eq!(vision.base_size, 960);
    assert_eq!(vision.image_size, 896);
}
