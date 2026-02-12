use deepseek_ocr_config::{AppConfig, ConfigOverrides, InferenceOverride};
use deepseek_ocr_core::{DecodeParameters, DecodeParametersPatch};

#[test]
fn inference_defaults_embed_decode_defaults() {
    let cfg = AppConfig::default();

    assert_eq!(cfg.inference.decode.max_new_tokens, 512);
    assert!(!cfg.inference.decode.do_sample);
    assert_eq!(cfg.inference.decode.temperature, 0.0);
    assert_eq!(cfg.inference.decode.top_p, Some(1.0));
    assert_eq!(cfg.inference.decode.top_k, None);
    assert_eq!(cfg.inference.decode.repetition_penalty, 1.0);
    assert_eq!(cfg.inference.decode.no_repeat_ngram_size, Some(20));
    assert_eq!(cfg.inference.decode.seed, None);
    assert!(cfg.inference.decode.use_cache);
}

#[test]
fn decode_patch_updates_only_selected_fields() {
    let mut decode = DecodeParameters::default();
    decode += &DecodeParametersPatch {
        max_new_tokens: Some(1024),
        top_p: Some(0.9),
        top_k: Some(32),
        use_cache: Some(false),
        ..Default::default()
    };

    assert_eq!(decode.max_new_tokens, 1024);
    assert_eq!(decode.top_p, Some(0.9));
    assert_eq!(decode.top_k, Some(32));
    assert!(!decode.use_cache);

    assert!(!decode.do_sample);
    assert_eq!(decode.temperature, 0.0);
    assert_eq!(decode.repetition_penalty, 1.0);
    assert_eq!(decode.no_repeat_ngram_size, Some(20));
}

#[test]
fn config_override_priority_cli_over_model_defaults() {
    let mut cfg = AppConfig::default();

    let overrides = ConfigOverrides {
        model_id: Some("deepseek-ocr-2".to_string()),
        inference: InferenceOverride {
            decode: DecodeParametersPatch {
                max_new_tokens: Some(2048),
                use_cache: Some(false),
                temperature: Some(0.7),
                do_sample: Some(true),
                ..Default::default()
            },
            ..InferenceOverride::default()
        },
        ..ConfigOverrides::default()
    };

    cfg.apply_overrides(&overrides);

    assert_eq!(cfg.inference.base_size, 1024);
    assert_eq!(cfg.inference.image_size, 768);
    assert!(cfg.inference.crop_mode);

    assert_eq!(cfg.inference.decode.max_new_tokens, 2048);
    assert!(!cfg.inference.decode.use_cache);
    assert_eq!(cfg.inference.decode.temperature, 0.7);
    assert!(cfg.inference.decode.do_sample);
}
