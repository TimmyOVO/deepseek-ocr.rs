use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    mem,
    path::{Path, PathBuf},
    slice,
};

use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, GgmlType as CandleGgmlType};

use deepseek_ocr_dsq::{DsqBiasDType, DsqTensorDType};
use half::{bf16, f16};
use tempfile::NamedTempFile;
use thiserror::Error;

const DSQ_MAGIC: &[u8; 7] = b"DSQSNAP";
const DSQ_VERSION: u32 = 1;
const Q8_BLOCK: usize = 32;
const Q4K_BLOCK: usize = 256;
const Q4K_BLOCK_BYTES: usize = mem::size_of::<BlockQ4K>(); // 144 bytes per block (K-scale layout)
const Q6K_BLOCK: usize = 256;
const Q6K_BLOCK_BYTES: usize = mem::size_of::<BlockQ6K>();
const Q8_BLOCK_BYTES: usize = Q8_BLOCK + 2;

/// Errors produced while exporting a DSQ snapshot.
#[derive(Debug, Error)]
pub enum DsqWriterError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("tensor `{0}` already exists")]
    DuplicateTensor(String),
    #[error("writer expected {expected} values for `{name}`, got {found}")]
    DimensionMismatch {
        name: String,
        expected: usize,
        found: usize,
    },
    #[error("tensor `{name}` requires in_dim divisible by {block}, got {in_dim}")]
    InvalidBlock {
        name: String,
        in_dim: usize,
        block: usize,
    },
    #[error("tensor `{name}` bias length {found} does not match out_dim {out_dim}")]
    BiasLengthMismatch {
        name: String,
        out_dim: usize,
        found: usize,
    },
    #[error("value `{what}` exceeds DSQ limits")]
    ValueOverflow { what: &'static str },
    #[error("metadata section size exceeds addressable range")]
    MetadataTooLarge,
    #[error("quantization failed: {0}")]
    Quantization(String),
}

pub type Result<T> = std::result::Result<T, DsqWriterError>;

/// Static metadata shared by the snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    pub candle_version: String,
    pub model_id: String,
    pub backend: String,
    pub default_qdtype: DsqTensorDType,
}

impl SnapshotMetadata {
    fn block_size(&self) -> Result<u32> {
        self.default_qdtype
            .block_size()
            .and_then(|size| u32::try_from(size).ok())
            .ok_or(DsqWriterError::ValueOverflow { what: "block_size" })
    }
}

#[derive(Debug)]
struct PendingRecord {
    name: String,
    out_dim: u32,
    in_dim: u32,
    q_dtype: DsqTensorDType,
    q_offset: u64,
    q_len: u64,
    bias_offset: Option<u64>,
    bias_len: Option<u64>,
    bias_dtype: Option<DsqBiasDType>,
}

/// Streaming DSQ writer that collects tensor records and emits the final file.
#[derive(Debug)]
pub struct DsqWriter {
    output_path: PathBuf,
    metadata: SnapshotMetadata,
    payload: NamedTempFile,
    payload_len: u64,
    records: Vec<PendingRecord>,
}

impl DsqWriter {
    /// Create a new writer targeting `output_path`.
    pub fn new(path: impl AsRef<Path>, metadata: SnapshotMetadata) -> Result<Self> {
        let output_path = path.as_ref().with_extension("dsq");
        let _ = metadata.block_size()?; // validate upfront
        Ok(Self {
            output_path,
            metadata,
            payload: NamedTempFile::new()?,
            payload_len: 0,
            records: Vec::new(),
        })
    }

    /// Quantize a dense matrix into Q8_0 blocks and append it as a tensor record.
    pub fn add_q8_tensor(
        &mut self,
        name: impl Into<String>,
        out_dim: usize,
        in_dim: usize,
        weights: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<()> {
        let name = name.into();
        if self.records.iter().any(|rec| rec.name == name) {
            return Err(DsqWriterError::DuplicateTensor(name.clone()));
        }
        if in_dim % Q8_BLOCK != 0 {
            return Err(DsqWriterError::InvalidBlock {
                name: name.clone(),
                in_dim,
                block: Q8_BLOCK,
            });
        }
        let expected = out_dim
            .checked_mul(in_dim)
            .ok_or(DsqWriterError::ValueOverflow {
                what: "tensor elements",
            })?;
        if weights.len() != expected {
            return Err(DsqWriterError::DimensionMismatch {
                name: name.clone(),
                expected,
                found: weights.len(),
            });
        }
        if let Some(bias_vals) = bias {
            if bias_vals.len() != out_dim {
                return Err(DsqWriterError::BiasLengthMismatch {
                    name: name.clone(),
                    out_dim,
                    found: bias_vals.len(),
                });
            }
        }
        let qbytes = quantize_q8_0(weights, out_dim, in_dim)?;
        let bias_bytes = bias.map(encode_bias_values);
        self.add_quantized_tensor_internal(
            name,
            out_dim,
            in_dim,
            DsqTensorDType::Q8_0,
            &qbytes,
            bias_bytes
                .as_deref()
                .map(|slice| (slice, DsqBiasDType::F32)),
        )
    }

    /// Quantize a dense matrix into Q4_K blocks and append it as a tensor record.
    pub fn add_q4k_tensor(
        &mut self,
        name: impl Into<String>,
        out_dim: usize,
        in_dim: usize,
        weights: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<()> {
        let name = name.into();
        if self.records.iter().any(|rec| rec.name == name) {
            return Err(DsqWriterError::DuplicateTensor(name.clone()));
        }
        if in_dim % Q4K_BLOCK != 0 {
            return Err(DsqWriterError::InvalidBlock {
                name: name.clone(),
                in_dim,
                block: Q4K_BLOCK,
            });
        }
        let expected = out_dim
            .checked_mul(in_dim)
            .ok_or(DsqWriterError::ValueOverflow {
                what: "tensor elements",
            })?;
        if weights.len() != expected {
            return Err(DsqWriterError::DimensionMismatch {
                name: name.clone(),
                expected,
                found: weights.len(),
            });
        }
        if let Some(bias_vals) = bias {
            if bias_vals.len() != out_dim {
                return Err(DsqWriterError::BiasLengthMismatch {
                    name: name.clone(),
                    out_dim,
                    found: bias_vals.len(),
                });
            }
        }
        let qbytes = quantize_q4k(weights, out_dim, in_dim)?;
        let bias_bytes = bias.map(encode_bias_values);
        self.add_quantized_tensor_internal(
            name,
            out_dim,
            in_dim,
            DsqTensorDType::Q4K,
            &qbytes,
            bias_bytes
                .as_deref()
                .map(|slice| (slice, DsqBiasDType::F32)),
        )
    }

    /// Quantize a dense matrix into Q6_K blocks and append it as a tensor record.
    pub fn add_q6k_tensor(
        &mut self,
        name: impl Into<String>,
        out_dim: usize,
        in_dim: usize,
        weights: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<()> {
        let name = name.into();
        if self.records.iter().any(|rec| rec.name == name) {
            return Err(DsqWriterError::DuplicateTensor(name.clone()));
        }
        if in_dim % Q6K_BLOCK != 0 {
            return Err(DsqWriterError::InvalidBlock {
                name: name.clone(),
                in_dim,
                block: Q6K_BLOCK,
            });
        }
        let expected = out_dim
            .checked_mul(in_dim)
            .ok_or(DsqWriterError::ValueOverflow {
                what: "tensor elements",
            })?;
        if weights.len() != expected {
            return Err(DsqWriterError::DimensionMismatch {
                name: name.clone(),
                expected,
                found: weights.len(),
            });
        }
        if let Some(bias_vals) = bias {
            if bias_vals.len() != out_dim {
                return Err(DsqWriterError::BiasLengthMismatch {
                    name: name.clone(),
                    out_dim,
                    found: bias_vals.len(),
                });
            }
        }
        let qbytes = quantize_q6k(weights, out_dim, in_dim)?;
        let bias_bytes = bias.map(encode_bias_values);
        self.add_quantized_tensor_internal(
            name,
            out_dim,
            in_dim,
            DsqTensorDType::Q6K,
            &qbytes,
            bias_bytes
                .as_deref()
                .map(|slice| (slice, DsqBiasDType::F32)),
        )
    }

    /// Append a dense F32 tensor without quantization.
    pub fn add_f32_tensor(
        &mut self,
        name: impl Into<String>,
        out_dim: usize,
        in_dim: usize,
        weights: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<()> {
        self.add_float_tensor(name, out_dim, in_dim, DsqTensorDType::F32, weights, bias)
    }

    /// Append a dense BF16 tensor without quantization.
    pub fn add_bf16_tensor(
        &mut self,
        name: impl Into<String>,
        out_dim: usize,
        in_dim: usize,
        weights: &[bf16],
        bias: Option<&[f32]>,
    ) -> Result<()> {
        self.add_float_tensor(name, out_dim, in_dim, DsqTensorDType::BF16, weights, bias)
    }

    /// Append a dense F16 tensor without quantization.
    pub fn add_f16_tensor(
        &mut self,
        name: impl Into<String>,
        out_dim: usize,
        in_dim: usize,
        weights: &[f16],
        bias: Option<&[f32]>,
    ) -> Result<()> {
        self.add_float_tensor(name, out_dim, in_dim, DsqTensorDType::F16, weights, bias)
    }

    fn add_float_tensor<T: Copy>(
        &mut self,
        name: impl Into<String>,
        out_dim: usize,
        in_dim: usize,
        dtype: DsqTensorDType,
        weights: &[T],
        bias: Option<&[f32]>,
    ) -> Result<()> {
        debug_assert!(dtype.elem_size_bytes().is_some());
        let name = name.into();
        if self.records.iter().any(|rec| rec.name == name) {
            return Err(DsqWriterError::DuplicateTensor(name));
        }
        let expected = out_dim
            .checked_mul(in_dim)
            .ok_or(DsqWriterError::ValueOverflow {
                what: "tensor elements",
            })?;
        if weights.len() != expected {
            return Err(DsqWriterError::DimensionMismatch {
                name: name.clone(),
                expected,
                found: weights.len(),
            });
        }
        if let Some(bias_vals) = bias {
            if bias_vals.len() != out_dim {
                return Err(DsqWriterError::BiasLengthMismatch {
                    name: name.clone(),
                    out_dim,
                    found: bias_vals.len(),
                });
            }
        }
        let bias_bytes = bias.map(encode_bias_values);
        let qbytes = slice_as_bytes(weights);
        self.add_quantized_tensor_internal(
            name,
            out_dim,
            in_dim,
            dtype,
            qbytes,
            bias_bytes
                .as_deref()
                .map(|slice| (slice, DsqBiasDType::F32)),
        )
    }

    /// Append a tensor whose quantized bytes were produced externally.
    pub fn add_quantized_bytes(
        &mut self,
        name: impl Into<String>,
        out_dim: usize,
        in_dim: usize,
        dtype: DsqTensorDType,
        qbytes: &[u8],
        bias: Option<(&[u8], DsqBiasDType)>,
    ) -> Result<()> {
        let name = name.into();
        if dtype.elem_size_bytes().is_some() {
            return Err(DsqWriterError::Quantization(format!(
                "add_quantized_bytes expects quantized dtype, got {dtype}"
            )));
        }
        if self.records.iter().any(|rec| rec.name == name) {
            return Err(DsqWriterError::DuplicateTensor(name));
        }
        let block = ensure_aligned(&name, dtype, in_dim)?;
        let expected_qbytes = expected_qbyte_len(dtype, out_dim, in_dim, block)?;
        if qbytes.len() != expected_qbytes {
            return Err(DsqWriterError::DimensionMismatch {
                name: name.clone(),
                expected: expected_qbytes,
                found: qbytes.len(),
            });
        }
        if let Some((bytes, bias_dtype)) = bias {
            let elem_size = bias_dtype_size(bias_dtype)?;
            let expected_bias = out_dim
                .checked_mul(elem_size)
                .ok_or(DsqWriterError::ValueOverflow { what: "bias bytes" })?;
            if bytes.len() != expected_bias {
                return Err(DsqWriterError::BiasLengthMismatch {
                    name: name.clone(),
                    out_dim,
                    found: bytes.len() / elem_size,
                });
            }
        }
        self.add_quantized_tensor_internal(name, out_dim, in_dim, dtype, qbytes, bias)
    }

    /// Finish writing and persist the final `.dsq` file.
    pub fn finalize(mut self) -> Result<()> {
        self.payload.flush()?;
        let header_bytes = self.encode_header()?;
        let records_len: usize = self.records.iter().map(record_entry_len).sum();
        let metadata_len = header_bytes
            .len()
            .checked_add(records_len)
            .ok_or(DsqWriterError::MetadataTooLarge)?;
        let metadata_len_u64 =
            u64::try_from(metadata_len).map_err(|_| DsqWriterError::MetadataTooLarge)?;
        let record_bytes = self.encode_records(metadata_len_u64)?;
        let file = File::create(&self.output_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&header_bytes)?;
        writer.write_all(&record_bytes)?;
        let mut payload_reader = BufReader::new(self.payload.reopen()?);
        io::copy(&mut payload_reader, &mut writer)?;
        writer.flush()?;
        Ok(())
    }

    fn add_quantized_tensor_internal(
        &mut self,
        name: String,
        out_dim: usize,
        in_dim: usize,
        q_dtype: DsqTensorDType,
        qbytes: &[u8],
        bias: Option<(&[u8], DsqBiasDType)>,
    ) -> Result<()> {
        let out_dim_u32 = u32::try_from(out_dim)
            .map_err(|_| DsqWriterError::ValueOverflow { what: "out_dim" })?;
        let in_dim_u32 =
            u32::try_from(in_dim).map_err(|_| DsqWriterError::ValueOverflow { what: "in_dim" })?;
        let q_offset = self.reserve_payload(qbytes)?;
        let (bias_offset, bias_len, bias_dtype) = match bias {
            Some((bytes, dtype)) => {
                let offset = self.reserve_payload(bytes)?;
                (Some(offset), Some(bytes.len() as u64), Some(dtype))
            }
            None => (None, None, None),
        };
        self.records.push(PendingRecord {
            name,
            out_dim: out_dim_u32,
            in_dim: in_dim_u32,
            q_dtype,
            q_offset,
            q_len: qbytes.len() as u64,
            bias_offset,
            bias_len,
            bias_dtype,
        });
        Ok(())
    }

    fn reserve_payload(&mut self, bytes: &[u8]) -> Result<u64> {
        let offset = self.payload_len;
        self.payload.write_all(bytes)?;
        self.payload_len = self.payload_len.checked_add(bytes.len() as u64).ok_or(
            DsqWriterError::ValueOverflow {
                what: "payload length",
            },
        )?;
        Ok(offset)
    }

    fn encode_header(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.extend_from_slice(DSQ_MAGIC);
        buf.extend_from_slice(&DSQ_VERSION.to_le_bytes());
        write_string(&mut buf, &self.metadata.candle_version);
        write_string(&mut buf, &self.metadata.model_id);
        write_string(&mut buf, &self.metadata.backend);
        buf.extend_from_slice(&self.metadata.default_qdtype.as_u32().to_le_bytes());
        buf.extend_from_slice(&self.metadata.block_size()?.to_le_bytes());
        let tensor_count =
            u32::try_from(self.records.len()).map_err(|_| DsqWriterError::ValueOverflow {
                what: "tensor_count",
            })?;
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        Ok(buf)
    }

    fn encode_records(&self, metadata_len: u64) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        for record in &self.records {
            write_string(&mut buf, &record.name);
            buf.extend_from_slice(&record.out_dim.to_le_bytes());
            buf.extend_from_slice(&record.in_dim.to_le_bytes());
            buf.extend_from_slice(&record.q_dtype.as_u32().to_le_bytes());
            let q_offset = record
                .q_offset
                .checked_add(metadata_len)
                .ok_or(DsqWriterError::ValueOverflow { what: "q_offset" })?;
            buf.extend_from_slice(&q_offset.to_le_bytes());
            buf.extend_from_slice(&record.q_len.to_le_bytes());
            if let (Some(offset), Some(len), Some(dtype)) =
                (record.bias_offset, record.bias_len, record.bias_dtype)
            {
                let bias_offset =
                    offset
                        .checked_add(metadata_len)
                        .ok_or(DsqWriterError::ValueOverflow {
                            what: "bias_offset",
                        })?;
                buf.extend_from_slice(&bias_offset.to_le_bytes());
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&dtype.as_u32().to_le_bytes());
            } else {
                buf.extend_from_slice(&0u64.to_le_bytes());
                buf.extend_from_slice(&0u64.to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        Ok(buf)
    }
}

fn record_entry_len(record: &PendingRecord) -> usize {
    52 + record.name.len()
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    let len = value.len() as u32;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

pub fn encode_bias_values(values: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        buf.extend_from_slice(&value.to_le_bytes());
    }
    buf
}

fn slice_as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * mem::size_of::<T>(),
        )
    }
}

pub fn quantize_q8_0(weights: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>> {
    if cols % Q8_BLOCK != 0 {
        return Err(DsqWriterError::InvalidBlock {
            name: "quantize_q8_0".into(),
            in_dim: cols,
            block: Q8_BLOCK,
        });
    }
    if weights.len() != rows * cols {
        return Err(DsqWriterError::DimensionMismatch {
            name: "quantize_q8_0".into(),
            expected: rows * cols,
            found: weights.len(),
        });
    }
    let blocks_per_row = cols / Q8_BLOCK;
    let mut result =
        Vec::with_capacity(rows * blocks_per_row * (Q8_BLOCK + std::mem::size_of::<f16>()));
    for row in 0..rows {
        let start = row * cols;
        let row_slice = &weights[start..start + cols];
        for block in row_slice.chunks_exact(Q8_BLOCK) {
            let mut amax = 0f32;
            for &value in block {
                let abs = value.abs();
                if abs > amax {
                    amax = abs;
                }
            }
            let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            result.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
            if scale == 0.0 {
                result.extend(std::iter::repeat(0u8).take(Q8_BLOCK));
                continue;
            }
            let inv = 1.0 / scale;
            for &value in block {
                let quant = (value * inv).round().clamp(-128.0, 127.0) as i32;
                result.push((quant as i8) as u8);
            }
        }
    }
    Ok(result)
}

pub fn quantize_q4k(weights: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>> {
    if cols % Q4K_BLOCK != 0 {
        return Err(DsqWriterError::InvalidBlock {
            name: "quantize_q4k".into(),
            in_dim: cols,
            block: Q4K_BLOCK,
        });
    }
    if weights.len() != rows * cols {
        return Err(DsqWriterError::DimensionMismatch {
            name: "quantize_q4k".into(),
            expected: rows * cols,
            found: weights.len(),
        });
    }
    let blocks_per_row = cols / Q4K_BLOCK;
    let total_blocks = rows
        .checked_mul(blocks_per_row)
        .ok_or(DsqWriterError::ValueOverflow { what: "q4k blocks" })?;
    let mut result = Vec::with_capacity(total_blocks * Q4K_BLOCK_BYTES);
    for row in 0..rows {
        let start = row * cols;
        let row_slice = &weights[start..start + cols];
        let mut blocks = vec![<BlockQ4K as CandleGgmlType>::zeros(); blocks_per_row];
        <BlockQ4K as CandleGgmlType>::from_float(row_slice, &mut blocks)
            .map_err(|err| DsqWriterError::Quantization(err.to_string()))?;
        let bytes = unsafe {
            slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * Q4K_BLOCK_BYTES)
        };
        result.extend_from_slice(bytes);
    }
    Ok(result)
}

pub fn quantize_q6k(weights: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>> {
    if cols % Q6K_BLOCK != 0 {
        return Err(DsqWriterError::InvalidBlock {
            name: "quantize_q6k".into(),
            in_dim: cols,
            block: Q6K_BLOCK,
        });
    }
    if weights.len() != rows * cols {
        return Err(DsqWriterError::DimensionMismatch {
            name: "quantize_q6k".into(),
            expected: rows * cols,
            found: weights.len(),
        });
    }
    let blocks_per_row = cols / Q6K_BLOCK;
    let total_blocks = rows
        .checked_mul(blocks_per_row)
        .ok_or(DsqWriterError::ValueOverflow { what: "q6k blocks" })?;
    let mut result = Vec::with_capacity(total_blocks * Q6K_BLOCK_BYTES);
    for row in 0..rows {
        let start = row * cols;
        let row_slice = &weights[start..start + cols];
        let mut blocks = vec![<BlockQ6K as CandleGgmlType>::zeros(); blocks_per_row];
        <BlockQ6K as CandleGgmlType>::from_float(row_slice, &mut blocks)
            .map_err(|err| DsqWriterError::Quantization(err.to_string()))?;
        let bytes = unsafe {
            slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * Q6K_BLOCK_BYTES)
        };
        result.extend_from_slice(bytes);
    }
    Ok(result)
}

fn ensure_aligned(name: &str, dtype: DsqTensorDType, in_dim: usize) -> Result<usize> {
    let Some(block) = dtype.block_size() else {
        return Err(DsqWriterError::Quantization(format!(
            "dtype {dtype} is not quantized"
        )));
    };
    if in_dim % block != 0 {
        return Err(DsqWriterError::InvalidBlock {
            name: name.to_string(),
            in_dim,
            block,
        });
    }
    Ok(block)
}

fn expected_qbyte_len(
    dtype: DsqTensorDType,
    out_dim: usize,
    in_dim: usize,
    block: usize,
) -> Result<usize> {
    let blocks_per_row = in_dim / block;
    let per_block = match dtype {
        DsqTensorDType::Q8_0 => Q8_BLOCK_BYTES,
        DsqTensorDType::Q4K => Q4K_BLOCK_BYTES,
        DsqTensorDType::Q6K => Q6K_BLOCK_BYTES,
        other => unreachable!("expected quantized dtype, got {other:?}"),
    };
    let row_bytes = blocks_per_row
        .checked_mul(per_block)
        .ok_or(DsqWriterError::ValueOverflow { what: "row bytes" })?;
    out_dim
        .checked_mul(row_bytes)
        .ok_or(DsqWriterError::ValueOverflow {
            what: "tensor bytes",
        })
}

fn bias_dtype_size(dtype: DsqBiasDType) -> Result<usize> {
    let size = match dtype {
        DsqBiasDType::U8 => 1,
        DsqBiasDType::U32 => 4,
        DsqBiasDType::I64 => 8,
        DsqBiasDType::F16 => 2,
        DsqBiasDType::F32 => 4,
        DsqBiasDType::F64 => 8,
        DsqBiasDType::BF16 => 2,
    };
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, GgmlType as CandleGgmlType};
    use deepseek_ocr_dsq::DsqReader;
    use tempfile::tempdir;

    #[test]
    fn writes_q8_tensor() {
        let tmp = tempdir().unwrap();
        let output = tmp.path().join("snapshot");
        let metadata = SnapshotMetadata {
            candle_version: "candle-test".to_string(),
            model_id: "unit-test".to_string(),
            backend: "CPU".to_string(),
            default_qdtype: DsqTensorDType::Q8_0,
        };
        let mut writer = DsqWriter::new(&output, metadata).unwrap();
        let out_dim = 2;
        let in_dim = 32;
        let mut weights = vec![0f32; out_dim * in_dim];
        for (idx, value) in weights.iter_mut().enumerate() {
            *value = (idx as f32 * 0.25) - 3.0;
        }
        let bias = vec![0.5f32, -0.25f32];
        writer
            .add_q8_tensor("linear.weight", out_dim, in_dim, &weights, Some(&bias))
            .unwrap();
        writer.finalize().unwrap();
        let dsq_path = output.with_extension("dsq");
        let reader = DsqReader::open(&dsq_path).unwrap();
        assert_eq!(reader.header().tensor_count, 1);
        let record = reader.tensor("linear.weight").unwrap();
        assert_eq!(record.out_dim, out_dim);
        assert_eq!(record.in_dim, in_dim);
        assert!(matches!(record.q_dtype, DsqTensorDType::Q8_0));
        let expected_q_len = out_dim * (in_dim / Q8_BLOCK) * (Q8_BLOCK + 2);
        assert_eq!(record.q_len as usize, expected_q_len);
        let bias_bytes = reader.bias_bytes(record).unwrap().unwrap();
        assert_eq!(bias_bytes.len(), out_dim * std::mem::size_of::<f32>());
        let mut decoded_bias = Vec::new();
        for chunk in bias_bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            decoded_bias.push(f32::from_le_bytes(arr));
        }
        assert_eq!(decoded_bias, bias);
        // Basic sanity: quantized payload is not all zeros.
        let payload = reader.tensor_bytes(record).unwrap();
        assert_eq!(payload.len(), expected_q_len);
        assert!(payload.iter().any(|&b| b != 0));
    }

    #[test]
    fn writes_q4k_tensor() {
        let tmp = tempdir().unwrap();
        let output = tmp.path().join("snapshot_q4");
        let metadata = SnapshotMetadata {
            candle_version: "candle-test".to_string(),
            model_id: "unit-test".to_string(),
            backend: "CPU".to_string(),
            default_qdtype: DsqTensorDType::Q4K,
        };
        let mut writer = DsqWriter::new(&output, metadata).unwrap();
        let out_dim = 1;
        let in_dim = 256;
        let weights: Vec<f32> = (0..(out_dim * in_dim))
            .map(|idx| (idx as f32 * 0.1) - 4.0)
            .collect();
        writer
            .add_q4k_tensor("proj.weight", out_dim, in_dim, &weights, None)
            .unwrap();
        writer.finalize().unwrap();
        let dsq_path = output.with_extension("dsq");
        let reader = DsqReader::open(&dsq_path).unwrap();
        let record = reader.tensor("proj.weight").unwrap();
        assert_eq!(record.out_dim, out_dim);
        assert_eq!(record.in_dim, in_dim);
        assert!(matches!(record.q_dtype, DsqTensorDType::Q4K));
        let expected_q_len = out_dim * (in_dim / Q4K_BLOCK) * Q4K_BLOCK_BYTES;
        assert_eq!(record.q_len as usize, expected_q_len);
        let payload = reader.tensor_bytes(record).unwrap();
        assert_eq!(payload.len(), expected_q_len);
        assert!(payload.iter().any(|&b| b != 0));
    }

    #[test]
    fn writes_q6k_tensor() {
        let tmp = tempdir().unwrap();
        let output = tmp.path().join("snapshot_q6");
        let metadata = SnapshotMetadata {
            candle_version: "candle-test".to_string(),
            model_id: "unit-test".to_string(),
            backend: "CPU".to_string(),
            default_qdtype: DsqTensorDType::Q6K,
        };
        let mut writer = DsqWriter::new(&output, metadata).unwrap();
        let out_dim = 2;
        let in_dim = 256;
        let weights: Vec<f32> = (0..(out_dim * in_dim))
            .map(|idx| (idx as f32 * 0.05) - 1.0)
            .collect();
        writer
            .add_q6k_tensor("attn.weight", out_dim, in_dim, &weights, None)
            .unwrap();
        writer.finalize().unwrap();
        let dsq_path = output.with_extension("dsq");
        let reader = DsqReader::open(&dsq_path).unwrap();
        let record = reader.tensor("attn.weight").unwrap();
        assert_eq!(record.out_dim, out_dim);
        assert_eq!(record.in_dim, in_dim);
        assert!(matches!(record.q_dtype, DsqTensorDType::Q6K));
        let expected_q_len = out_dim * (in_dim / Q6K_BLOCK) * Q6K_BLOCK_BYTES;
        assert_eq!(record.q_len as usize, expected_q_len);
        let payload = reader.tensor_bytes(record).unwrap();
        assert_eq!(payload.len(), expected_q_len);
        assert!(payload.iter().any(|&b| b != 0));
    }

    #[test]
    fn q6k_bytes_match_candle_from_float() {
        // Deterministic weights for 1x256 and 2x256
        let shapes = [(1usize, 256usize), (2usize, 256usize)];
        for (rows, cols) in shapes {
            let total = rows * cols;
            let weights: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.0375) - 2.0).collect();

            // DSQ writer output
            let dsq_bytes = quantize_q6k(&weights, rows, cols).expect("quantize_q6k");

            // Candle reference blocks -> bytes
            let blocks_per_row = cols / Q6K_BLOCK;
            let mut ref_all = Vec::with_capacity(rows * blocks_per_row * Q6K_BLOCK_BYTES);
            for r in 0..rows {
                let start = r * cols;
                let row_slice = &weights[start..start + cols];
                let mut blocks = vec![<BlockQ6K as CandleGgmlType>::zeros(); blocks_per_row];
                <BlockQ6K as CandleGgmlType>::from_float(row_slice, &mut blocks).unwrap();
                let bytes = unsafe {
                    slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        blocks.len() * Q6K_BLOCK_BYTES,
                    )
                };
                ref_all.extend_from_slice(bytes);
            }
            assert_eq!(dsq_bytes, ref_all, "Q6_K bytes mismatch for {rows}x{cols}");
        }
    }

    #[test]
    fn q4k_bytes_match_candle_from_float() {
        // Deterministic weights for 1x256 and 2x256
        let shapes = [(1usize, 256usize), (2usize, 256usize)];
        for (rows, cols) in shapes {
            let total = rows * cols;
            let weights: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.051) - 1.25).collect();

            // DSQ writer output
            let dsq_bytes = quantize_q4k(&weights, rows, cols).expect("quantize_q4k");

            // Candle reference blocks -> bytes
            let blocks_per_row = cols / Q4K_BLOCK;
            let mut ref_all = Vec::with_capacity(rows * blocks_per_row * Q4K_BLOCK_BYTES);
            for r in 0..rows {
                let start = r * cols;
                let row_slice = &weights[start..start + cols];
                let mut blocks = vec![<BlockQ4K as CandleGgmlType>::zeros(); blocks_per_row];
                <BlockQ4K as CandleGgmlType>::from_float(row_slice, &mut blocks).unwrap();
                let bytes = unsafe {
                    slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        blocks.len() * Q4K_BLOCK_BYTES,
                    )
                };
                ref_all.extend_from_slice(bytes);
            }
            assert_eq!(dsq_bytes, ref_all, "Q4_K bytes mismatch for {rows}x{cols}");
        }
    }

    #[test]
    fn writes_f32_tensor_payload() {
        let tmp = tempdir().unwrap();
        let output = tmp.path().join("snapshot_f32");
        let metadata = SnapshotMetadata {
            candle_version: "candle-test".to_string(),
            model_id: "unit-test".to_string(),
            backend: "CPU".to_string(),
            default_qdtype: DsqTensorDType::Q8_0,
        };
        let mut writer = DsqWriter::new(&output, metadata).unwrap();
        let out_dim = 2;
        let in_dim = 3;
        let weights: Vec<f32> = vec![0.5, -1.25, 2.0, 0.125, -0.75, 1.5];
        let bias = vec![0.25f32, -0.5f32];
        writer
            .add_f32_tensor("dense.weight", out_dim, in_dim, &weights, Some(&bias))
            .unwrap();
        writer.finalize().unwrap();
        let dsq_path = output.with_extension("dsq");
        let reader = DsqReader::open(&dsq_path).unwrap();
        let record = reader.tensor("dense.weight").unwrap();
        assert!(matches!(record.q_dtype, DsqTensorDType::F32));
        assert_eq!(record.q_len as usize, out_dim * in_dim * 4);
        let payload = reader.tensor_bytes(record).unwrap();
        assert_eq!(payload, super::slice_as_bytes(&weights));
        let bias_bytes = reader.bias_bytes(record).unwrap().unwrap();
        assert_eq!(bias_bytes.len(), bias.len() * std::mem::size_of::<f32>());
    }

    #[test]
    fn writes_bf16_tensor_payload() {
        let tmp = tempdir().unwrap();
        let output = tmp.path().join("snapshot_bf16");
        let metadata = SnapshotMetadata {
            candle_version: "candle-test".to_string(),
            model_id: "unit-test".to_string(),
            backend: "CPU".to_string(),
            default_qdtype: DsqTensorDType::Q8_0,
        };
        let mut writer = DsqWriter::new(&output, metadata).unwrap();
        let out_dim = 1;
        let in_dim = 4;
        let weights: Vec<bf16> = vec![
            bf16::from_f32(0.5),
            bf16::from_f32(-1.25),
            bf16::from_f32(0.0),
            bf16::from_f32(2.0),
        ];
        writer
            .add_bf16_tensor("bf16.weight", out_dim, in_dim, &weights, None)
            .unwrap();
        writer.finalize().unwrap();
        let reader = DsqReader::open(output.with_extension("dsq")).unwrap();
        let record = reader.tensor("bf16.weight").unwrap();
        assert!(matches!(record.q_dtype, DsqTensorDType::BF16));
        assert_eq!(record.q_len as usize, out_dim * in_dim * 2);
        let payload = reader.tensor_bytes(record).unwrap();
        assert_eq!(payload, super::slice_as_bytes(&weights));
    }

    #[test]
    fn writes_f16_tensor_payload() {
        let tmp = tempdir().unwrap();
        let output = tmp.path().join("snapshot_f16");
        let metadata = SnapshotMetadata {
            candle_version: "candle-test".to_string(),
            model_id: "unit-test".to_string(),
            backend: "CPU".to_string(),
            default_qdtype: DsqTensorDType::Q8_0,
        };
        let mut writer = DsqWriter::new(&output, metadata).unwrap();
        let out_dim = 1;
        let in_dim = 4;
        let weights: Vec<f16> = vec![
            f16::from_f32(0.25),
            f16::from_f32(-2.0),
            f16::from_f32(1.0),
            f16::from_f32(3.5),
        ];
        writer
            .add_f16_tensor("f16.weight", out_dim, in_dim, &weights, None)
            .unwrap();
        writer.finalize().unwrap();
        let reader = DsqReader::open(output.with_extension("dsq")).unwrap();
        let record = reader.tensor("f16.weight").unwrap();
        assert!(matches!(record.q_dtype, DsqTensorDType::F16));
        assert_eq!(record.q_len as usize, out_dim * in_dim * 2);
        let payload = reader.tensor_bytes(record).unwrap();
        assert_eq!(payload, super::slice_as_bytes(&weights));
    }
}
