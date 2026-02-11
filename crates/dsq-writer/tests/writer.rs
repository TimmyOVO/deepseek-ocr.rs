use std::slice;

use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, GgmlType as CandleGgmlType};
use deepseek_ocr_dsq::{DsqReader, DsqTensorDType};
use deepseek_ocr_dsq_writer::{
    DsqWriter, SnapshotMetadata, quantize_q4k, quantize_q6k, slice_as_bytes_public,
};
use half::{bf16, f16};
use tempfile::tempdir;

const Q8_BLOCK: usize = 32;
const Q4K_BLOCK: usize = 256;
const Q6K_BLOCK: usize = 256;
const Q4K_BLOCK_BYTES: usize = std::mem::size_of::<BlockQ4K>();
const Q6K_BLOCK_BYTES: usize = std::mem::size_of::<BlockQ6K>();

#[test]
fn writes_q8_tensor() {
    let tmp = tempdir().expect("tempdir");
    let output = tmp.path().join("snapshot");
    let metadata = SnapshotMetadata {
        candle_version: "candle-test".to_string(),
        model_id: "unit-test".to_string(),
        backend: "CPU".to_string(),
        default_qdtype: DsqTensorDType::Q8_0,
    };
    let mut writer = DsqWriter::new(&output, metadata).expect("writer");
    let out_dim = 2;
    let in_dim = 32;
    let mut weights = vec![0f32; out_dim * in_dim];
    for (idx, value) in weights.iter_mut().enumerate() {
        *value = (idx as f32 * 0.25) - 3.0;
    }
    let bias = vec![0.5f32, -0.25f32];
    writer
        .add_q8_tensor("linear.weight", out_dim, in_dim, &weights, Some(&bias))
        .expect("add tensor");
    writer.finalize().expect("finalize");
    let dsq_path = output.with_extension("dsq");
    let reader = DsqReader::open(&dsq_path).expect("open dsq");
    assert_eq!(reader.header().tensor_count, 1);
    let record = reader.tensor("linear.weight").expect("record");
    assert_eq!(record.out_dim, out_dim);
    assert_eq!(record.in_dim, in_dim);
    assert!(matches!(record.q_dtype, DsqTensorDType::Q8_0));
    let expected_q_len = out_dim * (in_dim / Q8_BLOCK) * (Q8_BLOCK + 2);
    assert_eq!(record.q_len as usize, expected_q_len);
    let bias_bytes = reader.bias_bytes(record).expect("bias read").expect("bias");
    assert_eq!(bias_bytes.len(), out_dim * std::mem::size_of::<f32>());
    let mut decoded_bias = Vec::new();
    for chunk in bias_bytes.chunks_exact(std::mem::size_of::<f32>()) {
        let mut arr = [0u8; 4];
        arr.copy_from_slice(chunk);
        decoded_bias.push(f32::from_le_bytes(arr));
    }
    assert_eq!(decoded_bias, bias);
    let payload = reader.tensor_bytes(record).expect("payload");
    assert_eq!(payload.len(), expected_q_len);
    assert!(payload.iter().any(|&byte| byte != 0));
}

#[test]
fn writes_q4k_tensor() {
    let tmp = tempdir().expect("tempdir");
    let output = tmp.path().join("snapshot_q4");
    let metadata = SnapshotMetadata {
        candle_version: "candle-test".to_string(),
        model_id: "unit-test".to_string(),
        backend: "CPU".to_string(),
        default_qdtype: DsqTensorDType::Q4K,
    };
    let mut writer = DsqWriter::new(&output, metadata).expect("writer");
    let out_dim = 1;
    let in_dim = 256;
    let weights: Vec<f32> = (0..(out_dim * in_dim))
        .map(|idx| (idx as f32 * 0.1) - 4.0)
        .collect();
    writer
        .add_q4k_tensor("proj.weight", out_dim, in_dim, &weights, None)
        .expect("add tensor");
    writer.finalize().expect("finalize");
    let dsq_path = output.with_extension("dsq");
    let reader = DsqReader::open(&dsq_path).expect("open dsq");
    let record = reader.tensor("proj.weight").expect("record");
    assert_eq!(record.out_dim, out_dim);
    assert_eq!(record.in_dim, in_dim);
    assert!(matches!(record.q_dtype, DsqTensorDType::Q4K));
    let expected_q_len = out_dim * (in_dim / Q4K_BLOCK) * Q4K_BLOCK_BYTES;
    assert_eq!(record.q_len as usize, expected_q_len);
    let payload = reader.tensor_bytes(record).expect("payload");
    assert_eq!(payload.len(), expected_q_len);
    assert!(payload.iter().any(|&byte| byte != 0));
}

#[test]
fn writes_q6k_tensor() {
    let tmp = tempdir().expect("tempdir");
    let output = tmp.path().join("snapshot_q6");
    let metadata = SnapshotMetadata {
        candle_version: "candle-test".to_string(),
        model_id: "unit-test".to_string(),
        backend: "CPU".to_string(),
        default_qdtype: DsqTensorDType::Q6K,
    };
    let mut writer = DsqWriter::new(&output, metadata).expect("writer");
    let out_dim = 2;
    let in_dim = 256;
    let weights: Vec<f32> = (0..(out_dim * in_dim))
        .map(|idx| (idx as f32 * 0.05) - 1.0)
        .collect();
    writer
        .add_q6k_tensor("attn.weight", out_dim, in_dim, &weights, None)
        .expect("add tensor");
    writer.finalize().expect("finalize");
    let dsq_path = output.with_extension("dsq");
    let reader = DsqReader::open(&dsq_path).expect("open dsq");
    let record = reader.tensor("attn.weight").expect("record");
    assert_eq!(record.out_dim, out_dim);
    assert_eq!(record.in_dim, in_dim);
    assert!(matches!(record.q_dtype, DsqTensorDType::Q6K));
    let expected_q_len = out_dim * (in_dim / Q6K_BLOCK) * Q6K_BLOCK_BYTES;
    assert_eq!(record.q_len as usize, expected_q_len);
    let payload = reader.tensor_bytes(record).expect("payload");
    assert_eq!(payload.len(), expected_q_len);
    assert!(payload.iter().any(|&byte| byte != 0));
}

#[test]
fn q6k_bytes_match_candle_from_float() {
    let shapes = [(1usize, 256usize), (2usize, 256usize)];
    for (rows, cols) in shapes {
        let total = rows * cols;
        let weights: Vec<f32> = (0..total).map(|idx| ((idx as f32) * 0.0375) - 2.0).collect();
        let dsq_bytes = quantize_q6k(&weights, rows, cols).expect("quantize_q6k");

        let blocks_per_row = cols / Q6K_BLOCK;
        let mut reference = Vec::with_capacity(rows * blocks_per_row * Q6K_BLOCK_BYTES);
        for row_idx in 0..rows {
            let start = row_idx * cols;
            let row_slice = &weights[start..start + cols];
            let mut blocks = vec![<BlockQ6K as CandleGgmlType>::zeros(); blocks_per_row];
            <BlockQ6K as CandleGgmlType>::from_float(row_slice, &mut blocks);
            let bytes = unsafe {
                slice::from_raw_parts(
                    blocks.as_ptr() as *const u8,
                    blocks.len() * Q6K_BLOCK_BYTES,
                )
            };
            reference.extend_from_slice(bytes);
        }
        assert_eq!(dsq_bytes, reference, "Q6_K bytes mismatch for {rows}x{cols}");
    }
}

#[test]
fn q4k_bytes_match_candle_from_float() {
    let shapes = [(1usize, 256usize), (2usize, 256usize)];
    for (rows, cols) in shapes {
        let total = rows * cols;
        let weights: Vec<f32> = (0..total).map(|idx| ((idx as f32) * 0.051) - 1.25).collect();
        let dsq_bytes = quantize_q4k(&weights, rows, cols).expect("quantize_q4k");

        let blocks_per_row = cols / Q4K_BLOCK;
        let mut reference = Vec::with_capacity(rows * blocks_per_row * Q4K_BLOCK_BYTES);
        for row_idx in 0..rows {
            let start = row_idx * cols;
            let row_slice = &weights[start..start + cols];
            let mut blocks = vec![<BlockQ4K as CandleGgmlType>::zeros(); blocks_per_row];
            <BlockQ4K as CandleGgmlType>::from_float(row_slice, &mut blocks);
            let bytes = unsafe {
                slice::from_raw_parts(
                    blocks.as_ptr() as *const u8,
                    blocks.len() * Q4K_BLOCK_BYTES,
                )
            };
            reference.extend_from_slice(bytes);
        }
        assert_eq!(dsq_bytes, reference, "Q4_K bytes mismatch for {rows}x{cols}");
    }
}

#[test]
fn writes_f32_tensor_payload() {
    let tmp = tempdir().expect("tempdir");
    let output = tmp.path().join("snapshot_f32");
    let metadata = SnapshotMetadata {
        candle_version: "candle-test".to_string(),
        model_id: "unit-test".to_string(),
        backend: "CPU".to_string(),
        default_qdtype: DsqTensorDType::Q8_0,
    };
    let mut writer = DsqWriter::new(&output, metadata).expect("writer");
    let out_dim = 2;
    let in_dim = 3;
    let weights: Vec<f32> = vec![0.5, -1.25, 2.0, 0.125, -0.75, 1.5];
    let bias = vec![0.25f32, -0.5f32];
    writer
        .add_f32_tensor("dense.weight", out_dim, in_dim, &weights, Some(&bias))
        .expect("add tensor");
    writer.finalize().expect("finalize");
    let dsq_path = output.with_extension("dsq");
    let reader = DsqReader::open(&dsq_path).expect("open dsq");
    let record = reader.tensor("dense.weight").expect("record");
    assert!(matches!(record.q_dtype, DsqTensorDType::F32));
    assert_eq!(record.q_len as usize, out_dim * in_dim * 4);
    let payload = reader.tensor_bytes(record).expect("payload");
    assert_eq!(payload, slice_as_bytes_public(&weights));
    let bias_bytes = reader.bias_bytes(record).expect("bias read").expect("bias");
    assert_eq!(bias_bytes.len(), bias.len() * std::mem::size_of::<f32>());
}

#[test]
fn writes_bf16_tensor_payload() {
    let tmp = tempdir().expect("tempdir");
    let output = tmp.path().join("snapshot_bf16");
    let metadata = SnapshotMetadata {
        candle_version: "candle-test".to_string(),
        model_id: "unit-test".to_string(),
        backend: "CPU".to_string(),
        default_qdtype: DsqTensorDType::Q8_0,
    };
    let mut writer = DsqWriter::new(&output, metadata).expect("writer");
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
        .expect("add tensor");
    writer.finalize().expect("finalize");
    let reader = DsqReader::open(output.with_extension("dsq")).expect("open dsq");
    let record = reader.tensor("bf16.weight").expect("record");
    assert!(matches!(record.q_dtype, DsqTensorDType::BF16));
    assert_eq!(record.q_len as usize, out_dim * in_dim * 2);
    let payload = reader.tensor_bytes(record).expect("payload");
    assert_eq!(payload, slice_as_bytes_public(&weights));
}

#[test]
fn writes_f16_tensor_payload() {
    let tmp = tempdir().expect("tempdir");
    let output = tmp.path().join("snapshot_f16");
    let metadata = SnapshotMetadata {
        candle_version: "candle-test".to_string(),
        model_id: "unit-test".to_string(),
        backend: "CPU".to_string(),
        default_qdtype: DsqTensorDType::Q8_0,
    };
    let mut writer = DsqWriter::new(&output, metadata).expect("writer");
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
        .expect("add tensor");
    writer.finalize().expect("finalize");
    let reader = DsqReader::open(output.with_extension("dsq")).expect("open dsq");
    let record = reader.tensor("f16.weight").expect("record");
    assert!(matches!(record.q_dtype, DsqTensorDType::F16));
    assert_eq!(record.q_len as usize, out_dim * in_dim * 2);
    let payload = reader.tensor_bytes(record).expect("payload");
    assert_eq!(payload, slice_as_bytes_public(&weights));
}
