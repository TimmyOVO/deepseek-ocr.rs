use std::{
    fs::File,
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::{Context, Result};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};

#[derive(Debug, Clone)]
pub struct TensorRecord {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub num_elements: usize,
    pub num_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct WeightsSummary {
    pub path: PathBuf,
    pub file_size: u64,
    pub modified: Option<SystemTime>,
    pub tensor_count: usize,
    pub sample: Vec<TensorRecord>,
}

pub fn summarize_weights(path: impl AsRef<Path>, sample_limit: usize) -> Result<WeightsSummary> {
    let path = path.as_ref();
    let file = File::open(path)
        .with_context(|| format!("failed to open weights file {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed to mmap {}", path.display()))?;
    let tensors =
        SafeTensors::deserialize(&mmap).with_context(|| "failed to parse safetensors header")?;
    let mut sample = Vec::new();
    for (index, (name, tensor)) in tensors.iter().enumerate() {
        if index >= sample_limit {
            break;
        }
        let shape = tensor.shape().to_vec();
        let num_elements: usize = shape.iter().product();
        let dtype = tensor.dtype();
        let num_bytes = num_elements.checked_mul(dtype.size()).unwrap_or(0);
        sample.push(TensorRecord {
            name: name.to_owned(),
            dtype,
            shape,
            num_elements,
            num_bytes,
        });
    }

    Ok(WeightsSummary {
        path: path.to_path_buf(),
        file_size: metadata.len(),
        modified: metadata.modified().ok(),
        tensor_count: tensors.len(),
        sample,
    })
}
