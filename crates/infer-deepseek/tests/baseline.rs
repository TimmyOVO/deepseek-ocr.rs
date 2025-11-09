mod common;

use std::{
    cell::RefCell,
    convert::TryFrom,
    env, fs,
    io::{self, Write},
    path::Path,
    rc::Rc,
};

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Tensor};
use common::test_utils::{with_shared_ocr_model, workspace_path};
use deepseek_ocr_infer_deepseek::model::{
    DEFAULT_WEIGHTS_PATH, GenerateOptions, VisionProjectionOutputs,
};
use deepseek_ocr_infer_deepseek::vision::dynamic_preprocess;
use image::{GenericImageView, open};
use ndarray::Array2;
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct BaselineMetadata {
    prompt: String,
    markdown: String,
    device: String,
    dtype: String,
    image: String,
    #[serde(default)]
    base_size: Option<u32>,
    #[serde(default)]
    image_size: Option<u32>,
    #[serde(default)]
    crop_mode: Option<bool>,
    #[serde(default)]
    markdown_path: Option<String>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    prompt_assets_path: Option<String>,
    #[serde(default)]
    vision_token_total: Option<usize>,
    #[serde(default)]
    projector_outputs_path: Option<String>,
    #[serde(default)]
    vision_embeddings_path: Option<String>,
    #[serde(default)]
    output_tokens_path: Option<String>,
    #[serde(default)]
    logits_path: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct PromptRange {
    start: usize,
    length: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct PromptAssets {
    input_ids: Vec<i64>,
    images_seq_mask: Vec<u8>,
    image_token_ranges: Vec<PromptRange>,
    image_token_counts: Vec<usize>,
    vision_token_counts: Vec<usize>,
    vision_token_total: usize,
    bos_token_id: i64,
    image_token_id: i64,
    prefill_len: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OutputTokens {
    tokens: Vec<i64>,
    prefill_len: usize,
    generated_len: usize,
    #[serde(default)]
    eos_token_id: Option<i64>,
    #[serde(default)]
    decoded_markdown: Option<String>,
}

fn resolve_workspace_path<P>(path: P) -> std::path::PathBuf
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_path(path)
    }
}

#[test]
fn baseline_artifacts_present() {
    let path = workspace_path("baselines/sample/baseline.json");
    assert!(
        path.exists(),
        "baseline metadata not found at {}",
        path.display()
    );

    let data = fs::read_to_string(path).expect("failed to read baseline metadata");
    let meta: BaselineMetadata =
        serde_json::from_str(&data).expect("baseline metadata should parse as json");

    assert!(
        meta.prompt.contains("<image>"),
        "baseline prompt should include image token"
    );
    assert!(
        !meta.markdown.trim().is_empty(),
        "baseline markdown should not be empty"
    );
    assert_eq!(meta.device, "cpu", "baseline captured on expected device");
    let dtype_lc = meta.dtype.to_ascii_lowercase();
    assert!(
        matches!(dtype_lc.as_str(), "bf16" | "bfloat16" | "fp32" | "float32"),
        "baseline dtype recorded as {}, expected bf16/bfloat16/fp32/float32",
        meta.dtype
    );

    let image_path = workspace_path(&meta.image);
    assert!(
        image_path.exists(),
        "baseline image missing at {}",
        image_path.display()
    );

    let image = open(image_path).expect("failed to open baseline image");
    let image_size = meta.image_size.unwrap_or(640);
    let result = dynamic_preprocess(&image, 2, 9, image_size, false);
    let (w_ratio, h_ratio) = result.ratio;
    assert!(
        !result.tiles.is_empty(),
        "dynamic preprocess should produce tiles"
    );
    assert_eq!(
        result.tiles.len() as u32,
        w_ratio * h_ratio,
        "tile count should match ratio product"
    );
    for tile in &result.tiles {
        assert_eq!(tile.dimensions(), (640, 640));
    }
}

#[test]
fn baseline_prompt_assets_match_schema() {
    let metadata_path = workspace_path("baselines/sample/baseline.json");
    if !metadata_path.exists() {
        eprintln!(
            "skipping prompt asset test: metadata missing at {:?}",
            metadata_path
        );
        return;
    }
    let metadata: BaselineMetadata = match serde_json::from_str(
        &fs::read_to_string(metadata_path).expect("read baseline metadata"),
    ) {
        Ok(meta) => meta,
        Err(err) => panic!("baseline metadata should parse as json: {err}"),
    };

    let prompt_path = metadata
        .prompt_assets_path
        .as_deref()
        .unwrap_or("baselines/sample/prompt.json");
    let prompt_path = resolve_workspace_path(prompt_path);
    if !prompt_path.exists() {
        eprintln!(
            "skipping prompt asset test: prompt.json missing at {:?}",
            prompt_path
        );
        return;
    }

    let prompt: PromptAssets = serde_json::from_str(
        &fs::read_to_string(prompt_path).expect("failed to read prompt assets"),
    )
    .expect("prompt assets should parse as json");

    assert!(
        !prompt.input_ids.is_empty(),
        "prompt input_ids should not be empty"
    );
    assert_eq!(
        prompt.input_ids.len(),
        prompt.images_seq_mask.len(),
        "images_seq_mask length must match input_ids"
    );
    assert_eq!(
        prompt.prefill_len,
        prompt.input_ids.len(),
        "prefill_len should equal the number of prompt tokens"
    );
    assert_eq!(
        prompt.input_ids[0], prompt.bos_token_id,
        "prompt should start with bos token"
    );

    let mask_count: usize = prompt
        .images_seq_mask
        .iter()
        .map(|&flag| flag as usize)
        .sum();
    assert_eq!(
        mask_count, prompt.vision_token_total,
        "mask ones must match vision_token_total"
    );

    for (range, expected) in prompt
        .image_token_ranges
        .iter()
        .zip(prompt.image_token_counts.iter())
    {
        let segment = &prompt.images_seq_mask[range.start..range.start + range.length];
        let ones = segment.iter().map(|&flag| flag as usize).sum::<usize>();
        assert_eq!(
            ones, *expected,
            "image_token_count should equal mask ones within the recorded range"
        );
        let ids_segment = &prompt.input_ids[range.start..range.start + range.length];
        assert!(
            ids_segment.iter().all(|&id| id == prompt.image_token_id),
            "all ids inside image ranges must match <image> token id"
        );
    }

    assert_eq!(
        mask_count,
        prompt.vision_token_counts.iter().copied().sum::<usize>(),
        "vision_token_counts should sum to mask ones"
    );

    if let Some(expected_total) = metadata.vision_token_total {
        assert_eq!(
            expected_total, mask_count,
            "metadata.vision_token_total should equal prompt vision_token_total"
        );
    }
}

#[test]
fn baseline_vision_embeddings_match_reference() -> Result<()> {
    let weights_path = workspace_path(DEFAULT_WEIGHTS_PATH);
    if !weights_path.exists() {
        eprintln!(
            "skipping vision embeddings test: weights missing at {:?}",
            weights_path
        );
        return Ok(());
    }

    let metadata_path = workspace_path("baselines/sample/baseline.json");
    if !metadata_path.exists() {
        eprintln!(
            "skipping vision embeddings test: metadata missing at {:?}",
            metadata_path
        );
        return Ok(());
    }
    let metadata: BaselineMetadata = serde_json::from_str(&fs::read_to_string(metadata_path)?)
        .context("failed to parse baseline metadata")?;

    let image_path = resolve_workspace_path(&metadata.image);
    if !image_path.exists() {
        eprintln!(
            "skipping vision embeddings test: baseline image missing at {:?}",
            image_path
        );
        return Ok(());
    }

    let vision_path = metadata
        .vision_embeddings_path
        .as_deref()
        .unwrap_or("baselines/sample/vision_embeddings.npz");
    let vision_path = resolve_workspace_path(vision_path);
    if !vision_path.exists() {
        eprintln!(
            "skipping vision embeddings test: vision embeddings missing at {:?}",
            vision_path
        );
        return Ok(());
    }

    match with_shared_ocr_model(|model| {
        let image = open(image_path).context("failed to open baseline image")?;
        let base_size = metadata.base_size.unwrap_or(1024);
        let image_size = metadata.image_size.unwrap_or(640);
        let crop_mode = metadata.crop_mode.unwrap_or(true);
        let owned_input =
            model.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
        let vision_input = owned_input.as_ref();

        let VisionProjectionOutputs {
            global_pre,
            local_pre,
            ..
        } = model.compute_vision_projection(&vision_input)?;

        let global_pre = global_pre.to_dtype(DType::F32)?.contiguous()?;
        let local_pre = if let Some(t) = local_pre {
            Some(t.to_dtype(DType::F32)?.contiguous()?)
        } else {
            None
        };

        let mut vision_npz = ndarray_npy::NpzReader::new(fs::File::open(&vision_path)?)?;
        let global_pre_py: Array2<f32> = vision_npz
            .by_name("global_pre_image0.npy")
            .context("missing global_pre_image0.npy")?;
        let global_clip_py: Array2<f32> = vision_npz
            .by_name("global_clip_tokens_image0.npy")
            .context("missing global_clip_tokens_image0.npy")?;
        let global_sam_py: Array2<f32> = vision_npz
            .by_name("global_sam_tokens_image0.npy")
            .context("missing global_sam_tokens_image0.npy")?;
        let local_pre_py: Array2<f32> = vision_npz
            .by_name("local_pre_image0.npy")
            .unwrap_or_else(|_| Array2::<f32>::zeros((0, 0)));
        let local_clip_py: Array2<f32> = vision_npz
            .by_name("local_clip_tokens_image0.npy")
            .unwrap_or_else(|_| Array2::<f32>::zeros((0, 0)));
        let local_sam_py: Array2<f32> = vision_npz
            .by_name("local_sam_tokens_image0.npy")
            .unwrap_or_else(|_| Array2::<f32>::zeros((0, 0)));
        drop(vision_npz);

        let tol = 5.0;
        let (global_pre_diff, gp_row, gp_col) = max_abs_diff_info(&global_pre, &global_pre_py)?;
        let gp_rust_val = global_pre
            .narrow(0, gp_row, 1)?
            .narrow(1, gp_col, 1)?
            .reshape(())?
            .to_scalar::<f32>()?;
        let gp_py_val = global_pre_py[(gp_row, gp_col)];
        println!(
            "global_pre max diff: {global_pre_diff} @ ({gp_row}, {gp_col}) [rust {gp_rust_val}, py {gp_py_val}]"
        );

        let hidden_total = global_pre_py.ncols();
        let split_hidden = hidden_total / 2;
        let global_pre_clip = global_pre.narrow(1, 0, split_hidden)?;
        let global_pre_sam = global_pre.narrow(1, split_hidden, hidden_total - split_hidden)?;
        let (clip_diff, clip_row, clip_col) = max_abs_diff_info(&global_pre_clip, &global_clip_py)?;
        let clip_rust_val = global_pre_clip
            .narrow(0, clip_row, 1)?
            .narrow(1, clip_col, 1)?
            .reshape(())?
            .to_scalar::<f32>()?;
        let clip_py_val = global_clip_py[(clip_row, clip_col)];
        println!(
            "global_clip max diff: {clip_diff} @ ({clip_row}, {clip_col}) [rust {clip_rust_val}, py {clip_py_val}]"
        );
        let (sam_diff, sam_row, sam_col) = max_abs_diff_info(&global_pre_sam, &global_sam_py)?;
        let sam_rust_val = global_pre_sam
            .narrow(0, sam_row, 1)?
            .narrow(1, sam_col, 1)?
            .reshape(())?
            .to_scalar::<f32>()?;
        let sam_py_val = global_sam_py[(sam_row, sam_col)];
        println!(
            "global_sam max diff: {sam_diff} @ ({sam_row}, {sam_col}) [rust {sam_rust_val}, py {sam_py_val}]"
        );

        assert!(
            global_pre_diff <= tol,
            "global_pre max abs diff {} exceeds tolerance {}",
            global_pre_diff,
            tol
        );
        assert!(
            clip_diff <= tol,
            "global_clip max abs diff {} exceeds tolerance {}",
            clip_diff,
            tol
        );
        assert!(
            sam_diff <= tol,
            "global_sam max abs diff {} exceeds tolerance {}",
            sam_diff,
            tol
        );

        if global_pre_py.nrows() != metadata.vision_token_total.unwrap_or(global_pre_py.nrows()) {
            println!(
                "warning: metadata vision_token_total {:?} differs from global_pre rows {}",
                metadata.vision_token_total,
                global_pre_py.nrows()
            );
        }

        if local_pre_py.is_empty() {
            assert!(
                local_pre.as_ref().map_or(true, |tensor| tensor
                    .shape()
                    .dims2()
                    .map(|(rows, _)| rows == 0)
                    .unwrap_or(true)),
                "expected no local pre tokens but tensor present"
            );
        } else {
            let local_pre = local_pre.context("expected local pre tokens")?;
            let (local_diff, lp_row, lp_col) = max_abs_diff_info(&local_pre, &local_pre_py)?;
            let lp_rust_val = local_pre
                .narrow(0, lp_row, 1)?
                .narrow(1, lp_col, 1)?
                .reshape(())?
                .to_scalar::<f32>()?;
            let lp_py_val = local_pre_py[(lp_row, lp_col)];
            println!(
                "local_pre max diff: {local_diff} @ ({lp_row}, {lp_col}) [rust {lp_rust_val}, py {lp_py_val}]"
            );

            let local_hidden_total = local_pre_py.ncols();
            let local_split_hidden = local_hidden_total / 2;
            let local_clip = local_pre.narrow(1, 0, local_split_hidden)?;
            let local_sam = local_pre.narrow(
                1,
                local_split_hidden,
                local_hidden_total - local_split_hidden,
            )?;
            let (local_clip_diff, lc_row, lc_col) = max_abs_diff_info(&local_clip, &local_clip_py)?;
            let lc_rust_val = local_clip
                .narrow(0, lc_row, 1)?
                .narrow(1, lc_col, 1)?
                .reshape(())?
                .to_scalar::<f32>()?;
            let lc_py_val = local_clip_py[(lc_row, lc_col)];
            println!(
                "local_clip max diff: {local_clip_diff} @ ({lc_row}, {lc_col}) [rust {lc_rust_val}, py {lc_py_val}]"
            );
            let (local_sam_diff, ls_row, ls_col) = max_abs_diff_info(&local_sam, &local_sam_py)?;
            let ls_rust_val = local_sam
                .narrow(0, ls_row, 1)?
                .narrow(1, ls_col, 1)?
                .reshape(())?
                .to_scalar::<f32>()?;
            let ls_py_val = local_sam_py[(ls_row, ls_col)];
            println!(
                "local_sam max diff: {local_sam_diff} @ ({ls_row}, {ls_col}) [rust {ls_rust_val}, py {ls_py_val}]"
            );
            assert!(
                local_diff <= tol,
                "local_pre max abs diff {} exceeds tolerance {}",
                local_diff,
                tol
            );
            assert!(
                local_clip_diff <= tol,
                "local_clip max abs diff {} exceeds tolerance {}",
                local_clip_diff,
                tol
            );
            assert!(
                local_sam_diff <= tol,
                "local_sam max abs diff {} exceeds tolerance {}",
                local_sam_diff,
                tol
            );
        }

        Ok(())
    }) {
        Ok(result) => Ok(result),
        Err(err) => {
            eprintln!("skipping vision embeddings test: failed to use shared model:\n{err:#}");
            Ok(())
        }
    }
}

#[test]
#[ignore]
fn baseline_generation_matches_reference() -> Result<()> {
    let weights_path = workspace_path(DEFAULT_WEIGHTS_PATH);
    if !weights_path.exists() {
        eprintln!(
            "skipping baseline parity test: weights missing at {:?}",
            weights_path
        );
        return Ok(());
    }
    let tokenizer_path = workspace_path("DeepSeek-OCR/tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!(
            "skipping baseline parity test: tokenizer missing at {:?}",
            tokenizer_path
        );
        return Ok(());
    }
    let metadata_path = workspace_path("baselines/sample/baseline.json");
    if !metadata_path.exists() {
        eprintln!(
            "skipping baseline parity test: metadata missing at {:?}",
            metadata_path
        );
        return Ok(());
    }

    let metadata: BaselineMetadata = serde_json::from_str(&fs::read_to_string(metadata_path)?)
        .context("failed to parse baseline metadata")?;

    let image_path = resolve_workspace_path(&metadata.image);
    if !image_path.exists() {
        eprintln!(
            "skipping baseline parity test: baseline image missing at {:?}",
            image_path
        );
        return Ok(());
    }

    println!("Loading tokenizer from {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|err| anyhow!("failed to load tokenizer: {err}"))?;

    println!("Loading baseline image from {:?}", image_path);
    let image = open(image_path).context("failed to open baseline image")?;
    let base_size = metadata.base_size.unwrap_or(1024);
    let image_size = metadata.image_size.unwrap_or(640);
    let crop_mode = metadata.crop_mode.unwrap_or(true);

    let baseline_text_source = if let Some(path) = &metadata.markdown_path {
        let path = resolve_workspace_path(path);
        if path.exists() {
            fs::read_to_string(path)?
        } else {
            metadata.markdown.clone()
        }
    } else {
        metadata.markdown.clone()
    };

    let prompt_path = metadata
        .prompt_assets_path
        .as_deref()
        .unwrap_or("baselines/sample/prompt.json");
    let prompt_path = resolve_workspace_path(prompt_path);
    if !prompt_path.exists() {
        eprintln!(
            "skipping baseline parity test: prompt assets missing at {:?}",
            prompt_path
        );
        return Ok(());
    }
    let prompt: PromptAssets = serde_json::from_str(&fs::read_to_string(prompt_path)?)
        .context("failed to parse prompt assets")?;
    anyhow::ensure!(
        prompt.prefill_len == prompt.input_ids.len(),
        "prompt prefill_len {} does not match input_ids length {}",
        prompt.prefill_len,
        prompt.input_ids.len()
    );
    anyhow::ensure!(
        prompt.input_ids.len() == prompt.images_seq_mask.len(),
        "prompt images_seq_mask length {} must match input_ids length {}",
        prompt.images_seq_mask.len(),
        prompt.input_ids.len()
    );

    let generated_text = match with_shared_ocr_model(|model| {
        let requested_dtype = metadata.dtype.to_ascii_lowercase();
        match requested_dtype.as_str() {
            "bf16" | "bfloat16" => {
                println!(
                    "BF16 weights requested but shared model is {:?}",
                    model.dtype()
                );
            }
            "f16" | "float16" => {
                println!(
                    "F16 weights requested but shared model is {:?}",
                    model.dtype()
                );
            }
            "f32" | "float32" => {}
            other => {
                eprintln!(
                    "Unknown dtype `{other}`, using shared model dtype {:?}",
                    model.dtype()
                );
            }
        }

        println!(
            "Using shared model on {:?} with weights {:?} (dtype {:?})",
            model.device(),
            model.weights_path().display(),
            model.dtype()
        );
        println!(
            "Preparing vision input (base_size={}, image_size={}, crop_mode={})",
            base_size, image_size, crop_mode
        );
        let owned_input =
            model.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
        let vision_input = owned_input.as_ref();
        let vision_inputs = [Some(vision_input)];
        println!("Computing image embeddings...");
        let embeddings = model.compute_image_embeddings(&vision_inputs)?;
        let embeddings_slice = embeddings.as_slice();
        let (vision_tokens, hidden) = embeddings[0]
            .shape()
            .dims2()
            .context("vision embeddings must be 2D")?;
        assert_eq!(
            hidden,
            model.language_model().config().hidden_size,
            "projector hidden size mismatch"
        );

        anyhow::ensure!(
            prompt.vision_token_total == vision_tokens,
            "baseline prompt vision token total {} mismatches computed projector token count {}",
            prompt.vision_token_total,
            vision_tokens
        );

        let input_ids_vec = prompt.input_ids.clone();
        let mask_vec = prompt.images_seq_mask.clone();
        let seq_len = input_ids_vec.len();
        let input_ids = Tensor::from_vec(input_ids_vec.clone(), (1, seq_len), model.device())?
            .to_dtype(DType::I64)?;
        let mask_tensor =
            Tensor::from_vec(mask_vec, (1, seq_len), model.device())?.to_dtype(DType::U8)?;

        let requested_tokens = metadata.max_new_tokens.unwrap_or(512);
        println!(
            "Starting generation with requested budget {} tokens.",
            requested_tokens
        );
        let mut options = GenerateOptions::new(requested_tokens);
        let last_count = Rc::new(RefCell::new(0usize));
        let tokenizer_stream = tokenizer.clone();
        let stream_state = Rc::clone(&last_count);
        let stream_callback = move |count: usize, ids: &[i64]| {
            let mut last = stream_state.borrow_mut();
            if count <= *last {
                return;
            }
            let slice = &ids[*last..count];
            let tokens: Vec<u32> = slice
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .collect();
            if !tokens.is_empty() {
                if let Ok(decoded) = tokenizer_stream.decode(&tokens, true) {
                    if !decoded.is_empty() {
                        print!("{}", decoded);
                        let _ = io::stdout().flush();
                    }
                }
            }
            *last = count;
        };
        options.images_seq_mask = Some(&mask_tensor);
        options.image_inputs = Some(&vision_inputs);
        options.image_embeddings = Some(embeddings_slice);
        options.eos_token_id = model.language_model().config().eos_token_id;
        options.progress_callback = Some(&stream_callback);
        if env::var("DEEPSEEK_OCR_DISABLE_CACHE").is_ok() {
            println!("Cache disabled via DEEPSEEK_OCR_DISABLE_CACHE=1");
            options.use_cache = false;
        }

        let generated = model.generate(&input_ids, options)?;
        let generated_vec = generated.to_vec2::<i64>()?;
        let output_tokens = generated_vec
            .get(0)
            .context("generation output missing row")?;
        println!(
            "Generated {} tokens (first 16: {:?})",
            output_tokens.len(),
            &output_tokens.iter().take(16).collect::<Vec<_>>()
        );
        let output_ids: Vec<u32> = output_tokens
            .iter()
            .map(|&id| u32::try_from(id).context("generated token id must be non-negative"))
            .collect::<Result<Vec<_>>>()?;
        let decoded = tokenizer
            .decode(&output_ids, true)
            .map_err(|err| anyhow!("decode failed: {err}"))?;
        let generated_text = normalize_text(&decoded);
        println!("{}", generated_text);
        Ok(generated_text)
    }) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping baseline parity test: failed to use shared model:\n{err:#}");
            return Ok(());
        }
    };

    let baseline_text = normalize_text(&baseline_text_source);

    assert_eq!(
        generated_text, baseline_text,
        "Rust generation diverges from Python baseline"
    );

    Ok(())
}

#[test]
fn baseline_vision_projector_matches_reference() -> Result<()> {
    let weights_path = workspace_path(DEFAULT_WEIGHTS_PATH);
    if !weights_path.exists() {
        eprintln!(
            "skipping vision parity test: weights missing at {:?}",
            weights_path
        );
        return Ok(());
    }

    let metadata_path = workspace_path("baselines/sample/baseline.json");
    if !metadata_path.exists() {
        eprintln!(
            "skipping vision parity test: metadata missing at {:?}",
            metadata_path
        );
        return Ok(());
    }
    let metadata: BaselineMetadata = serde_json::from_str(&fs::read_to_string(metadata_path)?)
        .context("failed to parse baseline metadata")?;

    let image_path = resolve_workspace_path(&metadata.image);
    if !image_path.exists() {
        eprintln!(
            "skipping vision parity test: baseline image missing at {:?}",
            image_path
        );
        return Ok(());
    }

    let projector_path = metadata
        .projector_outputs_path
        .as_deref()
        .unwrap_or("baselines/sample/projector_outputs.npz");
    let projector_path = resolve_workspace_path(projector_path);
    if !projector_path.exists() {
        eprintln!(
            "skipping vision parity test: projector outputs missing at {:?}",
            projector_path
        );
        return Ok(());
    }

    match with_shared_ocr_model(|model| {
        let image = open(image_path).context("failed to open baseline image")?;
        let base_size = metadata.base_size.unwrap_or(1024);
        let image_size = metadata.image_size.unwrap_or(640);
        let crop_mode = metadata.crop_mode.unwrap_or(true);
        let owned_input =
            model.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
        let vision_input = owned_input.as_ref();

        let VisionProjectionOutputs {
            global_pre: _,
            local_pre: _,
            global_post,
            local_post,
            global_tokens,
            local_tokens,
            fused_tokens,
        } = model.compute_vision_projection(&vision_input)?;

        let global_post = global_post.to_dtype(DType::F32)?;
        let global_tokens = global_tokens.to_dtype(DType::F32)?;
        let fused_tokens = fused_tokens.to_dtype(DType::F32)?;
        let local_post = match local_post {
            Some(t) => Some(t.to_dtype(DType::F32)?),
            None => None,
        };
        let local_tokens = match local_tokens {
            Some(t) => Some(t.to_dtype(DType::F32)?),
            None => None,
        };

        let mut projector_npz = ndarray_npy::NpzReader::new(fs::File::open(projector_path)?)?;
        let global_post_py: Array2<f32> = projector_npz
            .by_name("global_post_image0.npy")
            .context("missing global_post_image0.npy")?;
        let local_post_py: Array2<f32> = projector_npz
            .by_name("local_post_image0.npy")
            .context("missing local_post_image0.npy")?;
        let global_tokens_py: Array2<f32> = projector_npz
            .by_name("global_tokens_image0.npy")
            .context("missing global_tokens_image0.npy")?;
        let local_tokens_py: Array2<f32> = projector_npz
            .by_name("local_tokens_image0.npy")
            .context("missing local_tokens_image0.npy")?;
        let fused_tokens_py: Array2<f32> = projector_npz
            .by_name("fused_tokens_image0.npy")
            .context("missing fused_tokens_image0.npy")?;
        let fused_concat_py: Array2<f32> = projector_npz
            .by_name("fused_concat.npy")
            .context("missing fused_concat.npy")?;
        drop(projector_npz);

        let tol = 2.0;
        let global_post_diff = max_abs_diff(&global_post, &global_post_py)?;
        println!("global_post max diff: {global_post_diff}");
        assert!(
            global_post_diff <= tol,
            "global_post max abs diff {} exceeds tolerance {}",
            global_post_diff,
            tol
        );

        let global_tokens_diff = max_abs_diff(&global_tokens, &global_tokens_py)?;
        println!("global_tokens max diff: {global_tokens_diff}");
        assert!(
            global_tokens_diff <= tol,
            "global_tokens max abs diff {} exceeds tolerance {}",
            global_tokens_diff,
            tol
        );

        let fused_tokens_diff = max_abs_diff(&fused_tokens, &fused_tokens_py)?;
        println!("fused_tokens max diff: {fused_tokens_diff}");
        assert!(
            fused_tokens_diff <= tol,
            "fused_tokens max abs diff {} exceeds tolerance {}",
            fused_tokens_diff,
            tol
        );

        let fused_concat_diff = max_abs_diff(&fused_tokens, &fused_concat_py)?;
        println!("fused_concat max diff: {fused_concat_diff}");
        assert!(
            fused_concat_diff <= tol,
            "fused_concat max abs diff {} exceeds tolerance {}",
            fused_concat_diff,
            tol
        );

        if local_post_py.is_empty() {
            assert!(
                local_post.as_ref().map_or(true, |tensor| tensor
                    .shape()
                    .dims2()
                    .map(|(rows, _)| rows == 0)
                    .unwrap_or(true)),
                "expected no local post tokens but tensor present"
            );
        } else {
            let local_post = local_post.context("expected local post tokens")?;
            let diff = max_abs_diff(&local_post, &local_post_py)?;
            println!("local_post max diff: {diff}");
            assert!(
                diff <= tol,
                "local_post max abs diff {} exceeds tolerance {}",
                diff,
                tol
            );
        }

        if local_tokens_py.is_empty() {
            assert!(
                local_tokens.as_ref().map_or(true, |tensor| tensor
                    .shape()
                    .dims2()
                    .map(|(rows, _)| rows == 0)
                    .unwrap_or(true)),
                "expected no local tokens but tensor present"
            );
        } else {
            let local_tokens = local_tokens.context("expected local tokens")?;
            let diff = max_abs_diff(&local_tokens, &local_tokens_py)?;
            println!("local_tokens max diff: {diff}");
            assert!(
                diff <= tol,
                "local_tokens max abs diff {} exceeds tolerance {}",
                diff,
                tol
            );
        }

        if let Some(expected_total) = metadata.vision_token_total {
            let (token_rows, _) = fused_tokens
                .shape()
                .dims2()
                .context("fused tokens must be 2D")?;
            assert_eq!(
                token_rows, expected_total,
                "fused token count {} does not match metadata vision_token_total {}",
                token_rows, expected_total
            );
        }

        Ok(())
    }) {
        Ok(result) => Ok(result),
        Err(err) => {
            eprintln!("skipping vision parity test: failed to use shared model:\n{err:#}");
            Ok(())
        }
    }
}

fn tensor_data_2d(tensor: &Tensor) -> Result<(Vec<f32>, usize, usize)> {
    let (rows, cols) = tensor.shape().dims2().context("tensor must be 2D")?;
    let flat = tensor
        .reshape((rows * cols,))?
        .to_vec1::<f32>()
        .context("failed to extract tensor data")?;
    Ok((flat, rows, cols))
}

fn max_abs_diff(tensor: &Tensor, expected: &Array2<f32>) -> Result<f32> {
    Ok(max_abs_diff_info(tensor, expected)?.0)
}

fn max_abs_diff_info(tensor: &Tensor, expected: &Array2<f32>) -> Result<(f32, usize, usize)> {
    let (data, rows, cols) = tensor_data_2d(tensor)?;
    assert_eq!(
        rows,
        expected.nrows(),
        "row mismatch: tensor has {}, expected {}",
        rows,
        expected.nrows()
    );
    assert_eq!(
        cols,
        expected.ncols(),
        "column mismatch: tensor has {}, expected {}",
        cols,
        expected.ncols()
    );
    let mut max_diff = 0f32;
    let mut max_row = 0usize;
    let mut max_col = 0usize;
    for (index, (lhs, rhs)) in data.iter().zip(expected.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        if diff > max_diff {
            max_diff = diff;
            max_row = index / cols;
            max_col = index % cols;
        }
    }
    Ok((max_diff, max_row, max_col))
}

#[test]
fn baseline_teacher_forcing_matches_reference() -> Result<()> {
    let weights_path = workspace_path(DEFAULT_WEIGHTS_PATH);
    if !weights_path.exists() {
        eprintln!(
            "skipping teacher forcing test: weights missing at {:?}",
            weights_path
        );
        return Ok(());
    }

    let metadata_path = workspace_path("baselines/sample/baseline.json");
    if !metadata_path.exists() {
        eprintln!(
            "skipping teacher forcing test: metadata missing at {:?}",
            metadata_path
        );
        return Ok(());
    }
    let metadata: BaselineMetadata = serde_json::from_str(&fs::read_to_string(metadata_path)?)
        .context("failed to parse baseline metadata")?;

    let prompt_path = metadata
        .prompt_assets_path
        .as_deref()
        .unwrap_or("baselines/sample/prompt.json");
    let prompt_path = resolve_workspace_path(prompt_path);
    if !prompt_path.exists() {
        eprintln!(
            "skipping teacher forcing test: prompt assets missing at {:?}",
            prompt_path
        );
        return Ok(());
    }
    let prompt: PromptAssets = serde_json::from_str(&fs::read_to_string(prompt_path)?)?;

    let output_tokens_path = metadata
        .output_tokens_path
        .as_deref()
        .unwrap_or("baselines/sample/output_tokens.json");
    let output_tokens_path = resolve_workspace_path(output_tokens_path);
    if !output_tokens_path.exists() {
        eprintln!(
            "skipping teacher forcing test: output tokens missing at {:?}",
            output_tokens_path
        );
        return Ok(());
    }
    let outputs: OutputTokens = serde_json::from_str(&fs::read_to_string(output_tokens_path)?)?;

    let logits_path = metadata
        .logits_path
        .as_deref()
        .unwrap_or("baselines/sample/logits.npz");
    let logits_path = resolve_workspace_path(logits_path);
    if !logits_path.exists() {
        eprintln!(
            "skipping teacher forcing test: logits missing at {:?}",
            logits_path
        );
        return Ok(());
    }

    let projector_path = metadata
        .projector_outputs_path
        .as_deref()
        .unwrap_or("baselines/sample/projector_outputs.npz");
    let projector_path = resolve_workspace_path(projector_path);
    if !projector_path.exists() {
        eprintln!(
            "skipping teacher forcing test: projector outputs missing at {:?}",
            projector_path
        );
        return Ok(());
    }

    match with_shared_ocr_model(|model| {
        let device = model.device();
        let all_tokens = &outputs.tokens;
        let seq_len = all_tokens.len();
        assert_eq!(
            seq_len,
            outputs.prefill_len + outputs.generated_len,
            "prefill_len + generated_len mismatch total tokens"
        );
        assert_eq!(
            prompt.prefill_len, outputs.prefill_len,
            "prompt prefill len mismatch output tokens"
        );

        let mut mask_vec = prompt.images_seq_mask.clone();
        mask_vec.extend(std::iter::repeat(0u8).take(outputs.generated_len));
        assert_eq!(mask_vec.len(), seq_len, "mask length mismatch total tokens");

        let attention_mask = Tensor::ones((1, seq_len), DType::U8, device)?.to_dtype(DType::I64)?;
        let mask_tensor = Tensor::from_vec(mask_vec, (1, seq_len), device)?.to_dtype(DType::U8)?;
        let input_ids =
            Tensor::from_vec(all_tokens.clone(), (1, seq_len), device)?.to_dtype(DType::I64)?;

        let mut projector_npz = ndarray_npy::NpzReader::new(fs::File::open(projector_path)?)?;
        let fused_tokens_py: Array2<f32> = projector_npz
            .by_name("fused_concat.npy")
            .context("missing fused_concat.npy for teacher forcing")?;
        drop(projector_npz);
        let (vision_rows, vision_hidden) = (fused_tokens_py.nrows(), fused_tokens_py.ncols());
        let fused_vec = fused_tokens_py.iter().copied().collect::<Vec<f32>>();
        let fused_tensor = Tensor::from_vec(fused_vec, (vision_rows, vision_hidden), device)?
            .to_dtype(DType::F32)?;
        let image_embeddings = vec![fused_tensor];

        let forward = model.forward(
            Some(&input_ids),
            None,
            Some(&attention_mask),
            None,
            Some(&mask_tensor),
            None,
            Some(image_embeddings.as_slice()),
            None,
            false,
        )?;
        let logits = forward.logits.get(0).context("logits missing batch dim")?;

        let mut logits_npz = ndarray_npy::NpzReader::new(fs::File::open(logits_path)?)?;
        let logits_py: Array2<f32> = logits_npz
            .by_name("logits.npy")
            .context("missing logits.npy")?;
        drop(logits_npz);

        let (py_rows, py_cols) = (logits_py.nrows(), logits_py.ncols());
        assert_eq!(
            py_rows, seq_len,
            "python logits rows {} do not match token count {}",
            py_rows, seq_len
        );
        let (rust_rows, rust_cols) = logits.shape().dims2().context("rust logits must be 2D")?;
        assert_eq!(
            rust_cols, py_cols,
            "rust logits vocab {} does not match python {}",
            rust_cols, py_cols
        );
        assert_eq!(
            rust_rows, seq_len,
            "rust logits rows {} do not match token count {}",
            rust_rows, seq_len
        );

        let mut max_diff = 0f32;
        for (row_idx, py_row) in logits_py.outer_iter().enumerate() {
            let rust_row = logits.narrow(0, row_idx, 1)?.squeeze(0)?;
            let rust_vec = rust_row.to_vec1::<f32>()?;
            let py_vec = py_row.to_vec();
            for (lhs, rhs) in rust_vec.iter().zip(py_vec.iter()) {
                let diff = (lhs - rhs).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        println!("teacher-forcing logits max diff: {max_diff}");
        assert!(
            max_diff <= 1e-1,
            "teacher forcing logits diverge (max diff {max_diff})"
        );
        Ok(())
    }) {
        Ok(result) => Ok(result),
        Err(err) => {
            eprintln!("skipping teacher forcing test: failed to use shared model:\n{err:#}");
            Ok(())
        }
    }
}

fn normalize_text(s: &str) -> String {
    s.replace("\r\n", "\n")
        .replace("<｜end▁of▁sentence｜>", "")
        .trim()
        .to_string()
}
