use std::io::Write;

use deepseek_ocr_dsq::{DsqBiasDType, DsqError, DsqReader, DsqTensorDType, Result};
use tempfile::NamedTempFile;

const DSQ_MAGIC: &[u8; 7] = b"DSQSNAP";
const DSQ_VERSION: u32 = 1;

fn write_string(buf: &mut Vec<u8>, value: &str) {
    let len = value.len() as u32;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

struct SnapshotSpec<'a> {
    header_dtype: DsqTensorDType,
    record_dtype: DsqTensorDType,
    block_size: u32,
    name: &'a str,
    out_dim: usize,
    in_dim: usize,
    q_bytes: &'a [u8],
    bias_bytes: Option<&'a [u8]>,
}

fn build_snapshot_bytes_with_version(version: u32, spec: SnapshotSpec<'_>) -> Vec<u8> {
    let mut file = Vec::new();
    file.extend_from_slice(DSQ_MAGIC);
    file.extend_from_slice(&version.to_le_bytes());
    write_string(&mut file, "candle-test");
    write_string(&mut file, "model-id");
    write_string(&mut file, "CPU");
    file.extend_from_slice(&(spec.header_dtype.as_u32()).to_le_bytes());
    file.extend_from_slice(&spec.block_size.to_le_bytes());
    file.extend_from_slice(&1u32.to_le_bytes());
    let record_size = (4 + spec.name.len()) + (4 * 3) + (8 * 4) + 4;
    let metadata_len = file.len() + record_size;
    let q_offset = metadata_len as u64;
    let q_len = spec.q_bytes.len() as u64;
    let bias_len = spec.bias_bytes.map(|bytes| bytes.len() as u64).unwrap_or(0);
    let bias_offset = q_offset + q_len;
    write_string(&mut file, spec.name);
    file.extend_from_slice(&(spec.out_dim as u32).to_le_bytes());
    file.extend_from_slice(&(spec.in_dim as u32).to_le_bytes());
    file.extend_from_slice(&(spec.record_dtype.as_u32()).to_le_bytes());
    file.extend_from_slice(&q_offset.to_le_bytes());
    file.extend_from_slice(&q_len.to_le_bytes());
    if spec.bias_bytes.is_some() {
        file.extend_from_slice(&bias_offset.to_le_bytes());
        file.extend_from_slice(&bias_len.to_le_bytes());
        file.extend_from_slice(&(DsqBiasDType::F32.as_u32()).to_le_bytes());
    } else {
        file.extend_from_slice(&0u64.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes());
    }
    file.extend_from_slice(spec.q_bytes);
    if let Some(bias) = spec.bias_bytes {
        file.extend_from_slice(bias);
    }
    file
}

fn build_snapshot_bytes(spec: SnapshotSpec<'_>) -> Vec<u8> {
    build_snapshot_bytes_with_version(DSQ_VERSION, spec)
}

#[test]
fn parses_valid_snapshot() -> Result<()> {
    let out_dim = 64;
    let in_dim = 96;
    let blocks_per_row = in_dim / 32;
    let q_len = out_dim * blocks_per_row * (32 + 2);
    let q_bytes = vec![0xAB; q_len];
    let bias_bytes = vec![0u8; out_dim * 4];
    let bytes = build_snapshot_bytes(SnapshotSpec {
        header_dtype: DsqTensorDType::Q8_0,
        record_dtype: DsqTensorDType::Q8_0,
        block_size: 32,
        name: "layer.q_proj.weight",
        out_dim,
        in_dim,
        q_bytes: &q_bytes,
        bias_bytes: Some(&bias_bytes),
    });
    let mut file = NamedTempFile::new().expect("tempfile");
    file.write_all(&bytes).expect("write bytes");
    file.flush().expect("flush bytes");
    let reader = DsqReader::open(file.path())?;
    assert_eq!(reader.header().tensor_count, 1);
    let record = reader.tensor("layer.q_proj.weight").expect("tensor exists");
    assert_eq!(record.out_dim, out_dim);
    assert_eq!(record.in_dim, in_dim);
    assert!(matches!(record.q_dtype, DsqTensorDType::Q8_0));
    let q_slice = reader.tensor_bytes(record)?;
    assert_eq!(q_slice.len(), q_bytes.len());
    assert_eq!(q_slice[0], 0xAB);
    let bias_slice = reader.bias_bytes(record)?.expect("bias exists");
    assert_eq!(bias_slice.len(), bias_bytes.len());
    Ok(())
}

#[test]
fn rejects_unaligned_q8() {
    let out_dim: usize = 64;
    let in_dim: usize = 30;
    let q_len = out_dim * in_dim.div_ceil(32) * (32 + 2);
    let q_bytes = vec![0xCD; q_len];
    let bytes = build_snapshot_bytes(SnapshotSpec {
        header_dtype: DsqTensorDType::Q8_0,
        record_dtype: DsqTensorDType::Q8_0,
        block_size: 32,
        name: "bad",
        out_dim,
        in_dim,
        q_bytes: &q_bytes,
        bias_bytes: None,
    });
    let mut file = NamedTempFile::new().expect("tempfile");
    file.write_all(&bytes).expect("write bytes");
    file.flush().expect("flush bytes");
    let err = DsqReader::open(file.path()).expect_err("must fail");
    match err {
        DsqError::Validation(msg) => {
            assert!(msg.contains("not divisible"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn parses_q4k_snapshot() -> Result<()> {
    let out_dim = 32;
    let in_dim = 512;
    let q_bytes = vec![0xEE; 4096];
    let bytes = build_snapshot_bytes(SnapshotSpec {
        header_dtype: DsqTensorDType::Q4K,
        record_dtype: DsqTensorDType::Q4K,
        block_size: 256,
        name: "layer.o_proj.weight",
        out_dim,
        in_dim,
        q_bytes: &q_bytes,
        bias_bytes: None,
    });
    let mut file = NamedTempFile::new().expect("tempfile");
    file.write_all(&bytes).expect("write bytes");
    file.flush().expect("flush bytes");
    let reader = DsqReader::open(file.path())?;
    let record = reader.tensor("layer.o_proj.weight").expect("tensor exists");
    assert!(matches!(record.q_dtype, DsqTensorDType::Q4K));
    assert_eq!(record.in_dim, in_dim);
    assert_eq!(reader.tensor_bytes(record)?.len(), q_bytes.len());
    Ok(())
}

#[test]
fn parses_q6k_snapshot() -> Result<()> {
    let out_dim = 32;
    let in_dim = 512;
    let q_bytes = vec![0xAA; 4096];
    let bytes = build_snapshot_bytes(SnapshotSpec {
        header_dtype: DsqTensorDType::Q6K,
        record_dtype: DsqTensorDType::Q6K,
        block_size: 256,
        name: "layer.k_proj.weight",
        out_dim,
        in_dim,
        q_bytes: &q_bytes,
        bias_bytes: None,
    });
    let mut file = NamedTempFile::new().expect("tempfile");
    file.write_all(&bytes).expect("write bytes");
    file.flush().expect("flush bytes");
    let reader = DsqReader::open(file.path())?;
    let record = reader.tensor("layer.k_proj.weight").expect("tensor exists");
    assert!(matches!(record.q_dtype, DsqTensorDType::Q6K));
    assert_eq!(record.in_dim, in_dim);
    assert_eq!(reader.tensor_bytes(record)?.len(), q_bytes.len());
    Ok(())
}

#[test]
fn parses_f32_snapshot() -> Result<()> {
    let out_dim = 2;
    let in_dim = 3;
    let q_bytes = vec![0x11; out_dim * in_dim * 4];
    let bytes = build_snapshot_bytes(SnapshotSpec {
        header_dtype: DsqTensorDType::Q8_0,
        record_dtype: DsqTensorDType::F32,
        block_size: 32,
        name: "float.weight",
        out_dim,
        in_dim,
        q_bytes: &q_bytes,
        bias_bytes: None,
    });
    let mut file = NamedTempFile::new().expect("tempfile");
    file.write_all(&bytes).expect("write bytes");
    file.flush().expect("flush bytes");
    let reader = DsqReader::open(file.path())?;
    let record = reader.tensor("float.weight").expect("tensor exists");
    assert!(matches!(record.q_dtype, DsqTensorDType::F32));
    assert_eq!(record.in_dim, in_dim);
    assert_eq!(reader.tensor_bytes(record)?.len(), q_bytes.len());
    Ok(())
}

#[test]
fn rejects_float_with_wrong_byte_len() {
    let out_dim = 2;
    let in_dim = 3;
    let q_bytes = vec![0x22; out_dim * in_dim * 4 - 1];
    let bytes = build_snapshot_bytes(SnapshotSpec {
        header_dtype: DsqTensorDType::Q8_0,
        record_dtype: DsqTensorDType::F32,
        block_size: 32,
        name: "bad.float",
        out_dim,
        in_dim,
        q_bytes: &q_bytes,
        bias_bytes: None,
    });
    let mut file = NamedTempFile::new().expect("tempfile");
    file.write_all(&bytes).expect("write bytes");
    file.flush().expect("flush bytes");
    let err = DsqReader::open(file.path()).expect_err("must fail");
    match err {
        DsqError::Validation(msg) => {
            assert!(msg.contains("expected"), "unexpected message: {msg}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
