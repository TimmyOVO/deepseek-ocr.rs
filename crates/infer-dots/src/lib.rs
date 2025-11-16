use anyhow::{Result, bail};
use deepseek_ocr_core::{ModelKind, ModelLoadArgs, OcrEngine};

pub mod config;
pub mod model;
pub mod tokenizer;
pub mod transformer;
pub mod vision;

pub use model::DotsOcrModel;

pub fn load_model(args: ModelLoadArgs) -> Result<Box<dyn OcrEngine>> {
    if args.kind != ModelKind::DotsOcr {
        bail!(
            "ModelKind::{:?} cannot be loaded by the Dots OCR engine",
            args.kind
        );
    }

    let model = DotsOcrModel::load(args)?;
    Ok(Box::new(model))
}
