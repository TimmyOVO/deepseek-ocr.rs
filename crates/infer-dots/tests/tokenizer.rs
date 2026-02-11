use std::path::PathBuf;

use deepseek_ocr_infer_dots::tokenizer::{
    DotsImageTokens, IMAGE_END_TOKEN, IMAGE_START_TOKEN, load_tokenizer_config,
};
use tokenizers::Tokenizer;

fn tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dots.ocr/tokenizer.json")
}

fn tokenizer_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dots.ocr/tokenizer_config.json")
}

#[test]
fn resolve_tokens_from_repo_tokenizer() {
    let tokenizer = Tokenizer::from_file(tokenizer_path()).expect("dots.ocr tokenizer.json");
    let cfg = deepseek_ocr_infer_dots::config::load_dots_config(None).expect("config");
    let tokens = DotsImageTokens::resolve(&tokenizer, &cfg).expect("tokens present");
    assert_eq!(tokens.pad, 151665);
    assert_eq!(
        tokenizer.id_to_token(tokens.start),
        Some(IMAGE_START_TOKEN.into())
    );
    assert_eq!(tokenizer.id_to_token(tokens.end), Some(IMAGE_END_TOKEN.into()));
}

#[test]
fn load_tokenizer_config_file() {
    let path = tokenizer_config_path();
    assert!(
        path.exists(),
        "tokenizer_config path `{}` missing",
        path.display()
    );
    let contents = std::fs::read_to_string(&path).expect("tokenizer_config readable");
    assert!(
        contents.contains("\"chat_template\""),
        "tokenizer_config missing chat_template key"
    );
    let cfg = load_tokenizer_config(None).expect("tokenizer_config loads");
    assert!(
        cfg.chat_template
            .as_ref()
            .is_some_and(|template| template.contains("<|assistant|>")),
        "chat_template missing or malformed"
    );
}
