use deepseek_ocr_infer_dots::config::load_dots_config;

#[test]
fn load_default_config_from_repo() {
    let cfg = load_dots_config(None).expect("config should be readable");
    assert_eq!(cfg.model_type, "dots_ocr");
    assert_eq!(cfg.image_token_id, 151665);
    assert_eq!(cfg.video_token_id, 151656);
    assert_eq!(cfg.text.num_hidden_layers, 28);
    assert_eq!(cfg.vision.num_hidden_layers, 42);
}
