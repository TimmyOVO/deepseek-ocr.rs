mod deepseek_ocr;
mod dots_ocr;
pub(crate) mod helpers;
mod paddleocr_vl;

pub use deepseek_ocr::DeepSeekOcrAdapter;
pub use dots_ocr::DotsOcrAdapter;
pub use paddleocr_vl::PaddleOcrVlAdapter;
