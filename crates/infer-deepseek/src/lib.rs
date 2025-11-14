pub mod config;
pub mod model;
pub mod quant_snapshot;
pub mod quantization;
pub mod transformer;
pub mod vision;

pub use model::{DeepseekOcrModel, GenerateOptions, OwnedVisionInput, VisionInput, load_model};
