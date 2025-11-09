pub mod config;
pub mod transformer;
pub mod vision;
pub mod weights;

mod model;

pub use model::{PaddleOcrModel, load_model};
