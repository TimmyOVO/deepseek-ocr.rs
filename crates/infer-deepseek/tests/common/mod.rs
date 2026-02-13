use anyhow::Result;
use deepseek_ocr_infer_deepseek::config::DeepseekOcrConfig;
use std::path::PathBuf;

pub fn load_fixture(filename: &str) -> Result<DeepseekOcrConfig> {
    let path = fixtures_dir().join(filename);
    let raw = std::fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&raw)?)
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
