use std::{env, path::PathBuf};

use anyhow::Result;
use deepseek_ocr_infer_paddleocr::weights::summarize_weights;

fn main() -> Result<()> {
    let mut limit: usize = 25;
    let mut path: Option<PathBuf> = None;
    for arg in env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--limit=") {
            limit = value.parse().unwrap_or(limit);
            continue;
        }
        path = Some(PathBuf::from(arg));
    }

    let weights_path = path.unwrap_or_else(|| PathBuf::from("PaddleOCR-VL/model.safetensors"));
    let summary = summarize_weights(&weights_path, limit)?;

    println!("weights: {}", summary.path.display());
    println!(
        "size: {} ({} bytes)",
        human_bytes(summary.file_size),
        summary.file_size
    );
    println!("tensors: {}", summary.tensor_count);
    if limit == 0 || summary.sample.is_empty() {
        return Ok(());
    }
    println!("\nfirst {} tensors:", summary.sample.len());
    for (idx, record) in summary.sample.iter().enumerate() {
        println!(
            "{:>4}. {:<72} {:>6?} {:<30} elems={:<10} bytes={}",
            idx + 1,
            record.name,
            record.dtype,
            format_shape(&record.shape),
            record.num_elements,
            record.num_bytes,
        );
    }
    Ok(())
}

fn format_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        return "[]".to_string();
    }
    let inner = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", inner)
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes == 0 {
        return "0 B".into();
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}
