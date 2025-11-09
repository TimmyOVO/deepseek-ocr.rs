mod common;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};
use common::test_utils::{
    build_global_view_from_path, global_view_tensor_from_path, with_shared_ocr_model,
    workspace_path, workspace_root,
};
use ndarray::{Array2, ArrayD};
use ndarray_npy::NpzReader;
use serde_json::Value;
use std::fs::File;

use deepseek_ocr_infer_deepseek::vision::dynamic_preprocess;
use image::{DynamicImage, ImageBuffer, Rgb, open};

#[test]
fn dump_rust_global_view_image() -> Result<()> {
    let baseline_dir = workspace_path("baselines/sample");
    let metadata_path = baseline_dir.join("baseline.json");
    let metadata: Value = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    let base_size = metadata
        .get("base_size")
        .and_then(Value::as_u64)
        .context("baseline.json missing base_size")? as u32;
    let image_rel = metadata
        .get("image")
        .and_then(Value::as_str)
        .context("baseline.json missing image path")?;
    let image_path = workspace_root().join(image_rel);

    let rust_image = build_global_view_from_path(&image_path, base_size)?;
    let output_path = baseline_dir.join("rust_global_view_image0.png");
    rust_image.save(&output_path)?;
    println!("saved Rust global view image to {}", output_path.display());
    Ok(())
}
fn load_array_from_npz(reader: &mut NpzReader<File>, name: &str) -> Result<ArrayD<f32>> {
    reader
        .by_name(name)
        .or_else(|_| reader.by_name(&format!("{name}.npy")))
        .with_context(|| format!("array {name} missing from npz"))
}

#[test]
fn compare_global_view_preprocessing() -> Result<()> {
    let baseline_dir = workspace_path("baselines/sample");
    let metadata_path = baseline_dir.join("baseline.json");
    let metadata: Value = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    let base_size = metadata
        .get("base_size")
        .and_then(Value::as_u64)
        .context("baseline.json missing base_size")? as u32;
    let image_rel = metadata
        .get("image")
        .and_then(Value::as_str)
        .context("baseline.json missing image path")?;
    let image_path = workspace_root().join(image_rel);

    let npz_path = baseline_dir.join("image_tensors.npz");
    let mut reader = NpzReader::new(File::open(&npz_path)?)
        .with_context(|| format!("failed to open npz {}", npz_path.display()))?;
    let array = load_array_from_npz(&mut reader, "global_view_image0")?;
    let dims: Vec<usize> = array.shape().iter().map(|&d| d as usize).collect();
    let len = array.len();
    let (flat, offset) = array.into_raw_vec_and_offset();
    if let Some(off) = offset {
        ensure!(
            off == 0,
            "Python global view tensor stored with non-zero offset {off:?}"
        );
    }
    let device = Device::Cpu;
    let python_tensor = Tensor::from_vec(flat, len, &device)?.reshape(dims.clone())?;

    let rust_tensor = global_view_tensor_from_path(&image_path, base_size, &device, DType::F32)?
        .to_dtype(DType::F32)?;

    let diff_tensor = rust_tensor.sub(&python_tensor)?;
    let diff_abs = diff_tensor.abs()?;
    let diff = diff_abs.max_all()?.to_scalar::<f32>()?;
    let diff_flat = diff_abs.flatten_all()?.to_vec1::<f32>()?;
    let rust_flat = rust_tensor.flatten_all()?.to_vec1::<f32>()?;
    let python_flat = python_tensor.flatten_all()?.to_vec1::<f32>()?;
    if let Some((idx, max_val)) = diff_flat
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
        let height = dims[1];
        let width = dims[2];
        let channel = idx / (height * width);
        let rem = idx % (height * width);
        let row = rem / width;
        let col = rem % width;
        let rust_val = rust_flat[idx];
        let python_val = python_flat[idx];
        println!(
            "max diff between Rust and Python global view tensors: {max_val} (channel {}, row {}, col {}): rust {}, python {}",
            channel, row, col, rust_val, python_val
        );
    } else {
        println!("max diff between Rust and Python global view tensors: {diff}");
    }
    Ok(())
}

fn diff_images(
    a: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    b: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> (u8, usize) {
    let mut max_diff = 0u8;
    let mut max_idx = 0usize;
    for (idx, (&lhs, &rhs)) in a.as_raw().iter().zip(b.as_raw().iter()).enumerate() {
        let diff = lhs.abs_diff(rhs);
        if diff > max_diff {
            max_diff = diff;
            max_idx = idx;
        }
    }
    (max_diff, max_idx)
}

fn flatten_to_2d(tensor: &Tensor) -> Result<Tensor> {
    match tensor.rank() {
        2 => tensor
            .contiguous()
            .context("tensor not contiguous")
            .map_err(Into::into),
        3 => {
            let (d0, d1, d2) = tensor.shape().dims3()?;
            tensor
                .reshape((d0 * d1, d2))?
                .contiguous()
                .context("flattened tensor not contiguous")
                .map_err(Into::into)
        }
        other => bail!("unsupported tensor rank {} for flattening", other),
    }
}

fn tensor_f32_contiguous(tensor: &Tensor) -> Result<Tensor> {
    let tensor = if tensor.dtype() == DType::F32 {
        tensor.clone()
    } else {
        tensor.to_dtype(DType::F32)?
    };
    tensor
        .contiguous()
        .context("tensor not contiguous")
        .map_err(Into::into)
}

fn max_abs_diff_tensor_any(tensor: &Tensor, expected: &ArrayD<f32>) -> Result<(f32, Vec<usize>)> {
    let tensor = tensor_f32_contiguous(tensor)?;
    let dims_tensor = tensor.shape().dims();
    let dims_expected = expected.shape();
    ensure!(
        dims_tensor.len() == dims_expected.len(),
        "rank mismatch: tensor {} vs expected {}",
        dims_tensor.len(),
        dims_expected.len()
    );
    for (axis, (&lhs, &rhs)) in dims_tensor.iter().zip(dims_expected.iter()).enumerate() {
        ensure!(
            lhs == rhs,
            "dimension mismatch at axis {}: tensor {} vs expected {}",
            axis,
            lhs,
            rhs
        );
    }

    let lhs_flat = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut max_diff = 0f32;
    let mut max_index = 0usize;
    for (idx, (&lhs, &rhs)) in lhs_flat.iter().zip(expected.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = idx;
        }
    }

    let mut coords = vec![0usize; dims_expected.len()];
    if dims_expected.len() > 0 {
        let mut remainder = max_index;
        for axis in (0..dims_expected.len()).rev() {
            let dim = dims_expected[axis];
            coords[axis] = remainder % dim;
            remainder /= dim;
        }
    }

    Ok((max_diff, coords))
}

fn max_abs_diff_tensor(tensor: &Tensor, expected: &Array2<f32>) -> Result<(f32, usize, usize)> {
    let (rows, cols) = tensor
        .shape()
        .dims2()
        .context("tensor must be 2D for comparison")?;
    ensure!(
        rows == expected.nrows(),
        "row mismatch: tensor {}, expected {}",
        rows,
        expected.nrows()
    );
    ensure!(
        cols == expected.ncols(),
        "column mismatch: tensor {}, expected {}",
        cols,
        expected.ncols()
    );
    let flat = tensor
        .reshape((rows * cols,))?
        .to_vec1::<f32>()
        .context("failed to extract tensor data")?;
    let mut max_diff = 0f32;
    let mut max_index = 0usize;
    for (idx, (lhs, rhs)) in flat.iter().zip(expected.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = idx;
        }
    }
    let row = max_index / cols;
    let col = max_index % cols;
    Ok((max_diff, row, col))
}

#[test]
fn compare_local_crops_preprocessing() -> Result<()> {
    let baseline_dir = workspace_path("baselines/sample");
    let metadata_path = baseline_dir.join("baseline.json");
    let metadata: Value = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    let image_rel = metadata
        .get("image")
        .and_then(Value::as_str)
        .context("baseline.json missing image path")?;
    let image_path = workspace_root().join(image_rel);
    let image_size = metadata
        .get("image_size")
        .and_then(Value::as_u64)
        .unwrap_or(640) as u32;

    let source_rgb = open(&image_path)?.to_rgb8();
    let dynamic = DynamicImage::ImageRgb8(source_rgb);
    let preprocess = dynamic_preprocess(&dynamic, 2, 9, image_size, false);
    ensure!(
        !preprocess.tiles.is_empty(),
        "dynamic_preprocess produced no tiles"
    );

    let python_dir = baseline_dir.join("images");
    ensure!(
        python_dir.exists(),
        "baseline images directory missing at {}",
        python_dir.display()
    );

    let mut overall_max = 0u8;
    for (idx, tile) in preprocess.tiles.iter().enumerate() {
        let rust_rgb = tile.to_rgb8();
        let python_path = python_dir.join(format!("local_crop_image0_{idx}.png"));
        ensure!(
            python_path.exists(),
            "missing python crop {}",
            python_path.display()
        );
        let python_rgb = open(&python_path)?.to_rgb8();
        ensure!(
            rust_rgb.dimensions() == python_rgb.dimensions(),
            "dimension mismatch for crop {idx}: rust {:?}, python {:?}",
            rust_rgb.dimensions(),
            python_rgb.dimensions()
        );
        let (max_diff, max_idx) = diff_images(&rust_rgb, &python_rgb);
        println!("crop {idx}: max pixel diff {max_diff} at flat index {max_idx}");
        overall_max = overall_max.max(max_diff);
    }

    println!("overall local crop max diff: {overall_max}");
    Ok(())
}

#[test]
fn compare_clip_sam_tokens_against_reference() -> Result<()> {
    let baseline_dir = workspace_path("baselines/sample");
    let metadata_path = baseline_dir.join("baseline.json");
    let metadata: Value = serde_json::from_str(&std::fs::read_to_string(&metadata_path)?)?;
    let image_rel = metadata
        .get("image")
        .and_then(Value::as_str)
        .context("baseline.json missing image path")?;
    let image_path = workspace_root().join(image_rel);
    let vision_rel = metadata
        .get("vision_embeddings_path")
        .and_then(Value::as_str)
        .unwrap_or("baselines/sample/vision_embeddings.npz");
    let vision_path = workspace_root().join(vision_rel);
    let clip_trace_rel = metadata
        .get("clip_trace_path")
        .and_then(Value::as_str)
        .unwrap_or("baselines/sample/clip_trace.npz");
    let clip_trace_path = workspace_root().join(clip_trace_rel);
    ensure!(
        clip_trace_path.exists(),
        "clip trace npz missing at {}",
        clip_trace_path.display()
    );
    let sam_trace_rel = metadata
        .get("sam_trace_path")
        .and_then(Value::as_str)
        .unwrap_or("baselines/sample/sam_trace.npz");
    let sam_trace_path = workspace_root().join(sam_trace_rel);
    ensure!(
        sam_trace_path.exists(),
        "sam trace npz missing at {}",
        sam_trace_path.display()
    );

    let image = open(&image_path)?;
    let base_size = metadata
        .get("base_size")
        .and_then(Value::as_u64)
        .unwrap_or(1024) as u32;
    let image_size = metadata
        .get("image_size")
        .and_then(Value::as_u64)
        .unwrap_or(640) as u32;
    let crop_mode = metadata
        .get("crop_mode")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    with_shared_ocr_model(|model| {
        let owned_input =
            model.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
        let vision_input = owned_input.as_ref();
        let debug = model.compute_vision_debug_features(&vision_input)?;

        let mut reader = NpzReader::new(File::open(&vision_path)?)
            .with_context(|| format!("failed to open {}", vision_path.display()))?;
        let mut clip_reader = NpzReader::new(File::open(&clip_trace_path)?)
            .with_context(|| format!("failed to open {}", clip_trace_path.display()))?;
        let global_clip_py: Array2<f32> = reader
            .by_name("global_clip_tokens_image0.npy")
            .context("missing global_clip_tokens_image0.npy")?;
        let global_sam_py: Array2<f32> = reader
            .by_name("global_sam_tokens_image0.npy")
            .context("missing global_sam_tokens_image0.npy")?;
        let local_clip_py: Array2<f32> = reader
            .by_name("local_clip_tokens_image0.npy")
            .unwrap_or_else(|_| Array2::<f32>::zeros((0, 0)));
        let local_sam_py: Array2<f32> = reader
            .by_name("local_sam_tokens_image0.npy")
            .unwrap_or_else(|_| Array2::<f32>::zeros((0, 0)));

        let global_clip = debug
            .global_clip
            .to_dtype(DType::F32)?
            .contiguous()
            .context("global clip not contiguous")?;
        let (clip_batch, clip_seq, clip_hidden) = global_clip.shape().dims3()?;
        ensure!(clip_batch == 1, "global clip batch {} != 1", clip_batch);
        ensure!(
            clip_seq > 1,
            "global clip sequence length too small: {clip_seq}"
        );
        let global_clip_flat = global_clip
            .narrow(1, 1, clip_seq - 1)?
            .reshape((clip_seq - 1, clip_hidden))?
            .contiguous()
            .context("global clip flat not contiguous")?;
        let (global_clip_diff, gc_row, gc_col) =
            max_abs_diff_tensor(&global_clip_flat, &global_clip_py)?;
        println!("global_clip diff: {global_clip_diff} @ ({gc_row}, {gc_col})");

        let global_sam = debug
            .global_sam
            .to_dtype(DType::F32)?
            .contiguous()
            .context("global sam not contiguous")?;
        let (sam_batch, sam_channels, sam_height, sam_width) = global_sam.shape().dims4()?;
        ensure!(sam_batch == 1, "global sam batch {} != 1", sam_batch);
        let global_sam_flat = global_sam
            .reshape((sam_batch, sam_channels, sam_height * sam_width))?
            .permute((0, 2, 1))?
            .reshape((sam_batch * sam_height * sam_width, sam_channels))?
            .contiguous()
            .context("global sam flat not contiguous")?;
        let (global_sam_diff, gs_row, gs_col) =
            max_abs_diff_tensor(&global_sam_flat, &global_sam_py)?;
        println!("global_sam diff: {global_sam_diff} @ ({gs_row}, {gs_col})");

        let mut sam_reader_global = NpzReader::new(File::open(&sam_trace_path)?)
            .with_context(|| format!("failed to open {}", sam_trace_path.display()))?;
        let global_patch_py: ArrayD<f32> = sam_reader_global
            .by_name("global_patch_embed_image0.npy")
            .context("missing global_patch_embed_image0.npy")?;
        let (global_patch_diff, global_patch_idx) =
            max_abs_diff_tensor_any(&debug.global_sam_trace.patch_embed, &global_patch_py)?;
        println!(
            "global_sam patch_embed diff: {global_patch_diff} @ {:?}",
            global_patch_idx
        );

        if let Some(pos_tensor) = &debug.global_sam_trace.pos_added {
            let global_pos_py: ArrayD<f32> = sam_reader_global
                .by_name("global_pos_added_image0.npy")
                .context("missing global_pos_added_image0.npy")?;
            let (pos_diff, pos_idx) = max_abs_diff_tensor_any(pos_tensor, &global_pos_py)?;
            println!("global_sam pos_added diff: {pos_diff} @ {:?}", pos_idx);
        }

        for (idx, tensor) in debug.global_sam_trace.block_outputs.iter().enumerate() {
            let key = format!("global_block{idx}_image0.npy");
            let block_py: ArrayD<f32> = sam_reader_global
                .by_name(&key)
                .with_context(|| format!("missing {key}"))?;
            let (block_diff, block_idx) = max_abs_diff_tensor_any(tensor, &block_py)?;
            println!(
                "global_sam block[{idx}] diff: {block_diff} @ {:?}",
                block_idx
            );
        }

        let global_neck_conv1_py: ArrayD<f32> = sam_reader_global
            .by_name("global_neck_conv1_image0.npy")
            .context("missing global_neck_conv1_image0.npy")?;
        let (neck_conv1_diff, neck_conv1_idx) =
            max_abs_diff_tensor_any(&debug.global_sam_trace.neck_conv1, &global_neck_conv1_py)?;
        println!(
            "global_sam neck_conv1 diff: {neck_conv1_diff} @ {:?}",
            neck_conv1_idx
        );

        let global_neck_norm1_py: ArrayD<f32> = sam_reader_global
            .by_name("global_neck_norm1_image0.npy")
            .context("missing global_neck_norm1_image0.npy")?;
        let (neck_norm1_diff, neck_norm1_idx) =
            max_abs_diff_tensor_any(&debug.global_sam_trace.neck_norm1, &global_neck_norm1_py)?;
        println!(
            "global_sam neck_norm1 diff: {neck_norm1_diff} @ {:?}",
            neck_norm1_idx
        );

        let global_neck_conv2_py: ArrayD<f32> = sam_reader_global
            .by_name("global_neck_conv2_image0.npy")
            .context("missing global_neck_conv2_image0.npy")?;
        let (neck_conv2_diff, neck_conv2_idx) =
            max_abs_diff_tensor_any(&debug.global_sam_trace.neck_conv2, &global_neck_conv2_py)?;
        println!(
            "global_sam neck_conv2 diff: {neck_conv2_diff} @ {:?}",
            neck_conv2_idx
        );

        let global_neck_norm2_py: ArrayD<f32> = sam_reader_global
            .by_name("global_neck_norm2_image0.npy")
            .context("missing global_neck_norm2_image0.npy")?;
        let (neck_norm2_diff, neck_norm2_idx) =
            max_abs_diff_tensor_any(&debug.global_sam_trace.neck_norm2, &global_neck_norm2_py)?;
        println!(
            "global_sam neck_norm2 diff: {neck_norm2_diff} @ {:?}",
            neck_norm2_idx
        );

        let global_net2_py: ArrayD<f32> = sam_reader_global
            .by_name("global_net2_image0.npy")
            .context("missing global_net2_image0.npy")?;
        let (net2_diff, net2_idx) =
            max_abs_diff_tensor_any(&debug.global_sam_trace.net2, &global_net2_py)?;
        println!("global_sam net2 diff: {net2_diff} @ {:?}", net2_idx);

        let global_net3_py: ArrayD<f32> = sam_reader_global
            .by_name("global_net3_image0.npy")
            .context("missing global_net3_image0.npy")?;
        let (net3_diff, net3_idx) =
            max_abs_diff_tensor_any(&debug.global_sam_trace.net3, &global_net3_py)?;
        println!("global_sam net3 diff: {net3_diff} @ {:?}", net3_idx);

        drop(sam_reader_global);
        let global_trace = &debug.global_clip_trace;
        let global_embeddings_tensor = flatten_to_2d(
            &global_trace
                .embeddings
                .to_dtype(DType::F32)?
                .contiguous()
                .context("global embeddings not contiguous")?,
        )?;
        let global_embeddings_py: Array2<f32> = clip_reader
            .by_name("global_embeddings_image0.npy")
            .context("missing global_embeddings_image0.npy")?;
        let (emb_diff, emb_row, emb_col) =
            max_abs_diff_tensor(&global_embeddings_tensor, &global_embeddings_py)?;
        println!("global_embeddings diff: {emb_diff} @ ({emb_row}, {emb_col})");

        let global_pre_ln_tensor = flatten_to_2d(
            &global_trace
                .pre_layernorm
                .to_dtype(DType::F32)?
                .contiguous()
                .context("global pre_layernorm not contiguous")?,
        )?;
        let global_pre_ln_py: Array2<f32> = clip_reader
            .by_name("global_pre_layernorm_image0.npy")
            .context("missing global_pre_layernorm_image0.npy")?;
        let (pre_ln_diff, pre_ln_row, pre_ln_col) =
            max_abs_diff_tensor(&global_pre_ln_tensor, &global_pre_ln_py)?;
        println!("global_pre_layernorm diff: {pre_ln_diff} @ ({pre_ln_row}, {pre_ln_col})");

        for (layer_idx, layer_tensor) in global_trace.layer_outputs.iter().enumerate() {
            let layer_tensor = flatten_to_2d(
                &layer_tensor
                    .to_dtype(DType::F32)?
                    .contiguous()
                    .with_context(|| {
                        format!("global clip layer {layer_idx} output not contiguous")
                    })?,
            )?;
            let key = format!("global_block{layer_idx}_image0.npy");
            let layer_py: Array2<f32> = clip_reader
                .by_name(&key)
                .with_context(|| format!("missing {key}"))?;
            let (layer_diff, layer_row, layer_col) = max_abs_diff_tensor(&layer_tensor, &layer_py)?;
            println!("global_block[{layer_idx}] diff: {layer_diff} @ ({layer_row}, {layer_col})");
        }

        drop(clip_reader);

        if local_clip_py.is_empty() {
            println!("local_clip diff: baseline has no local clips");
        } else {
            let local_trace = debug
                .local_clip_trace
                .as_ref()
                .context("expected local clip trace data")?;
            let mut clip_local_reader = NpzReader::new(File::open(&clip_trace_path)?)
                .with_context(|| format!("failed to open {}", clip_trace_path.display()))?;
            let local_clip = debug
                .local_clip
                .context("expected local clip tokens")?
                .to_dtype(DType::F32)?
                .contiguous()
                .context("local clip not contiguous")?;
            let (local_patches, local_seq, local_hidden) = local_clip.shape().dims3()?;
            ensure!(
                local_seq > 1,
                "local clip sequence length too small: {local_seq}"
            );
            let local_clip_flat = local_clip
                .narrow(1, 1, local_seq - 1)?
                .reshape((local_patches * (local_seq - 1), local_hidden))?
                .contiguous()
                .context("local clip flat not contiguous")?;
            let (local_clip_diff, lc_row, lc_col) =
                max_abs_diff_tensor(&local_clip_flat, &local_clip_py)?;
            println!("local_clip diff: {local_clip_diff} @ ({lc_row}, {lc_col})");

            let local_embeddings_tensor = flatten_to_2d(
                &local_trace
                    .embeddings
                    .to_dtype(DType::F32)?
                    .contiguous()
                    .context("local embeddings not contiguous")?,
            )?;
            let local_embeddings_py: Array2<f32> = clip_local_reader
                .by_name("local_embeddings_image0.npy")
                .context("missing local_embeddings_image0.npy")?;
            let (local_emb_diff, le_row, le_col) =
                max_abs_diff_tensor(&local_embeddings_tensor, &local_embeddings_py)?;
            println!("local_embeddings diff: {local_emb_diff} @ ({le_row}, {le_col})");

            let local_pre_ln_tensor = flatten_to_2d(
                &local_trace
                    .pre_layernorm
                    .to_dtype(DType::F32)?
                    .contiguous()
                    .context("local pre_layernorm not contiguous")?,
            )?;
            let local_pre_ln_py: Array2<f32> = clip_local_reader
                .by_name("local_pre_layernorm_image0.npy")
                .context("missing local_pre_layernorm_image0.npy")?;
            let (local_pre_ln_diff, lpl_row, lpl_col) =
                max_abs_diff_tensor(&local_pre_ln_tensor, &local_pre_ln_py)?;
            println!("local_pre_layernorm diff: {local_pre_ln_diff} @ ({lpl_row}, {lpl_col})");

            for (layer_idx, layer_tensor) in local_trace.layer_outputs.iter().enumerate() {
                let layer_tensor = flatten_to_2d(
                    &layer_tensor
                        .to_dtype(DType::F32)?
                        .contiguous()
                        .with_context(|| {
                            format!("local clip layer {layer_idx} output not contiguous")
                        })?,
                )?;
                let key = format!("local_block{layer_idx}_image0.npy");
                let layer_py: Array2<f32> = clip_local_reader
                    .by_name(&key)
                    .with_context(|| format!("missing {key}"))?;
                let (layer_diff, layer_row, layer_col) =
                    max_abs_diff_tensor(&layer_tensor, &layer_py)?;
                println!(
                    "local_block[{layer_idx}] diff: {layer_diff} @ ({layer_row}, {layer_col})"
                );
            }

            drop(clip_local_reader);

            if let Some(local_sam_trace) = debug.local_sam_trace.as_ref() {
                let mut sam_reader_local = NpzReader::new(File::open(&sam_trace_path)?)
                    .with_context(|| format!("failed to open {}", sam_trace_path.display()))?;
                let local_patch_py: ArrayD<f32> = sam_reader_local
                    .by_name("local_patch_embed_image0.npy")
                    .context("missing local_patch_embed_image0.npy")?;
                let (local_patch_diff, local_patch_idx) =
                    max_abs_diff_tensor_any(&local_sam_trace.patch_embed, &local_patch_py)?;
                println!(
                    "local_sam patch_embed diff: {local_patch_diff} @ {:?}",
                    local_patch_idx
                );

                if let Some(pos_tensor) = &local_sam_trace.pos_added {
                    let local_pos_py: ArrayD<f32> = sam_reader_local
                        .by_name("local_pos_added_image0.npy")
                        .context("missing local_pos_added_image0.npy")?;
                    let (local_pos_diff, local_pos_idx) =
                        max_abs_diff_tensor_any(pos_tensor, &local_pos_py)?;
                    println!(
                        "local_sam pos_added diff: {local_pos_diff} @ {:?}",
                        local_pos_idx
                    );
                }

                for (idx, tensor) in local_sam_trace.block_outputs.iter().enumerate() {
                    let key = format!("local_block{idx}_image0.npy");
                    let block_py: ArrayD<f32> = sam_reader_local
                        .by_name(&key)
                        .with_context(|| format!("missing {key}"))?;
                    let (block_diff, block_idx) = max_abs_diff_tensor_any(tensor, &block_py)?;
                    println!(
                        "local_sam block[{idx}] diff: {block_diff} @ {:?}",
                        block_idx
                    );
                }

                let local_neck_conv1_py: ArrayD<f32> = sam_reader_local
                    .by_name("local_neck_conv1_image0.npy")
                    .context("missing local_neck_conv1_image0.npy")?;
                let (neck_conv1_diff, neck_conv1_idx) =
                    max_abs_diff_tensor_any(&local_sam_trace.neck_conv1, &local_neck_conv1_py)?;
                println!(
                    "local_sam neck_conv1 diff: {neck_conv1_diff} @ {:?}",
                    neck_conv1_idx
                );

                let local_neck_norm1_py: ArrayD<f32> = sam_reader_local
                    .by_name("local_neck_norm1_image0.npy")
                    .context("missing local_neck_norm1_image0.npy")?;
                let (neck_norm1_diff, neck_norm1_idx) =
                    max_abs_diff_tensor_any(&local_sam_trace.neck_norm1, &local_neck_norm1_py)?;
                println!(
                    "local_sam neck_norm1 diff: {neck_norm1_diff} @ {:?}",
                    neck_norm1_idx
                );

                let local_neck_conv2_py: ArrayD<f32> = sam_reader_local
                    .by_name("local_neck_conv2_image0.npy")
                    .context("missing local_neck_conv2_image0.npy")?;
                let (neck_conv2_diff, neck_conv2_idx) =
                    max_abs_diff_tensor_any(&local_sam_trace.neck_conv2, &local_neck_conv2_py)?;
                println!(
                    "local_sam neck_conv2 diff: {neck_conv2_diff} @ {:?}",
                    neck_conv2_idx
                );

                let local_neck_norm2_py: ArrayD<f32> = sam_reader_local
                    .by_name("local_neck_norm2_image0.npy")
                    .context("missing local_neck_norm2_image0.npy")?;
                let (neck_norm2_diff, neck_norm2_idx) =
                    max_abs_diff_tensor_any(&local_sam_trace.neck_norm2, &local_neck_norm2_py)?;
                println!(
                    "local_sam neck_norm2 diff: {neck_norm2_diff} @ {:?}",
                    neck_norm2_idx
                );

                let local_net2_py: ArrayD<f32> = sam_reader_local
                    .by_name("local_net2_image0.npy")
                    .context("missing local_net2_image0.npy")?;
                let (net2_diff, net2_idx) =
                    max_abs_diff_tensor_any(&local_sam_trace.net2, &local_net2_py)?;
                println!("local_sam net2 diff: {net2_diff} @ {:?}", net2_idx);

                let local_net3_py: ArrayD<f32> = sam_reader_local
                    .by_name("local_net3_image0.npy")
                    .context("missing local_net3_image0.npy")?;
                let (net3_diff, net3_idx) =
                    max_abs_diff_tensor_any(&local_sam_trace.net3, &local_net3_py)?;
                println!("local_sam net3 diff: {net3_diff} @ {:?}", net3_idx);
            }
        }

        if local_sam_py.is_empty() {
            println!("local_sam diff: baseline has no local sam tokens");
        } else {
            let local_sam = debug
                .local_sam
                .context("expected local SAM tokens")?
                .to_dtype(DType::F32)?
                .contiguous()
                .context("local sam not contiguous")?;
            let (local_patches, local_channels, local_height, local_width) =
                local_sam.shape().dims4()?;
            let local_sam_flat = local_sam
                .reshape((local_patches, local_channels, local_height * local_width))?
                .permute((0, 2, 1))?
                .reshape((local_patches * local_height * local_width, local_channels))?
                .contiguous()
                .context("local sam flat not contiguous")?;
            let (local_sam_diff, ls_row, ls_col) =
                max_abs_diff_tensor(&local_sam_flat, &local_sam_py)?;
            println!("local_sam diff: {local_sam_diff} @ ({ls_row}, {ls_col})");
        }

        Ok(())
    })
}
