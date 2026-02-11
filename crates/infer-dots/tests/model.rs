use std::{path::PathBuf, sync::Arc};

use deepseek_ocr_infer_dots::{
    config::load_dots_config,
    model::{build_prompt_inputs, vision_token_count},
    tokenizer::DotsImageTokens,
};
use tokenizers::Tokenizer;

fn tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dots.ocr/tokenizer.json")
}

#[test]
fn prompt_builder_inserts_image_tokens() -> anyhow::Result<()> {
    let tokenizer = Tokenizer::from_file(tokenizer_path()).expect("dots.ocr tokenizer load");
    let cfg = Arc::new(load_dots_config(None)?);
    let tokens = DotsImageTokens::resolve(&tokenizer, &cfg)?;
    let counts = vec![4usize];
    let (ids, mask) = build_prompt_inputs(&tokenizer, "User: <image> Answer:", &counts, &tokens)?;
    assert_eq!(mask.len(), ids.len());
    let pad_id = tokens.pad as i64;
    let pad_positions: Vec<_> = ids
        .iter()
        .enumerate()
        .filter_map(|(idx, &id)| (id == pad_id).then_some(idx))
        .collect();
    assert_eq!(pad_positions.len(), 4);
    for idx in pad_positions {
        assert_eq!(mask[idx], 1);
    }
    Ok(())
}

#[test]
fn vision_token_count_respects_merge() -> anyhow::Result<()> {
    let tokens = vision_token_count([1, 4, 4], 2)?;
    assert_eq!(tokens, 4);
    Ok(())
}
