mod common;

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Tensor};
use common::test_utils::{with_shared_ocr_model, workspace_path};
use deepseek_ocr_core::inference::{DecodeParameters, OcrEngine, VisionSettings};
use deepseek_ocr_infer_deepseek::vision::dynamic_preprocess;
use image::open;
use serde::Deserialize;

fn top2(values: &[f32]) -> (usize, f32, usize, f32) {
    let mut best1 = (0usize, f32::NEG_INFINITY);
    let mut best2 = (0usize, f32::NEG_INFINITY);
    for (idx, &v) in values.iter().enumerate() {
        if v > best1.1 {
            best2 = best1;
            best1 = (idx, v);
        } else if v > best2.1 {
            best2 = (idx, v);
        }
    }
    (best1.0, best1.1, best2.0, best2.1)
}

#[derive(Debug, Deserialize)]
struct PromptRange {
    #[allow(dead_code)]
    start: usize,
    #[allow(dead_code)]
    length: usize,
}

#[derive(Debug, Deserialize)]
struct PromptAssets {
    #[serde(default)]
    rendered_prompt: Option<String>,
    input_ids: Vec<i64>,
    images_seq_mask: Vec<u8>,
    #[allow(dead_code)]
    image_token_ranges: Vec<PromptRange>,
    #[allow(dead_code)]
    image_token_counts: Vec<usize>,
    #[allow(dead_code)]
    vision_token_counts: Vec<usize>,
    vision_token_total: usize,
    #[allow(dead_code)]
    bos_token_id: i64,
    #[allow(dead_code)]
    image_token_id: i64,
    prefill_len: usize,
}

#[derive(Debug, Deserialize)]
struct OutputTokens {
    tokens: Vec<i64>,
    prefill_len: usize,
    generated_len: usize,
    #[serde(default)]
    eos_token_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct BaselineMetadata {
    variant: String,
    #[allow(dead_code)]
    prompt: String,
    image: String,
    #[serde(default)]
    base_size: Option<u32>,
    #[serde(default)]
    image_size: Option<u32>,
    #[serde(default)]
    crop_mode: Option<bool>,
    #[serde(default)]
    _max_new_tokens: Option<usize>,
    #[serde(default)]
    prompt_assets_path: Option<String>,
    #[serde(default)]
    output_tokens_path: Option<String>,
}

fn resolve_workspace_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_path(path)
    }
}

fn first_mismatch(lhs: &[i64], rhs: &[i64]) -> Option<usize> {
    lhs.iter()
        .zip(rhs.iter())
        .position(|(a, b)| a != b)
        .or_else(|| {
            if lhs.len() == rhs.len() {
                None
            } else {
                Some(lhs.len().min(rhs.len()))
            }
        })
}

fn load_baseline(baseline_dir: &Path) -> Result<(BaselineMetadata, PromptAssets, OutputTokens)> {
    let meta_path = baseline_dir.join("baseline.json");
    let meta: BaselineMetadata = serde_json::from_str(
        &fs::read_to_string(&meta_path).with_context(|| format!("read {}", meta_path.display()))?,
    )
    .with_context(|| format!("parse {}", meta_path.display()))?;

    let prompt_path = meta.prompt_assets_path.as_deref().unwrap_or("prompt.json");
    let prompt_path = if prompt_path.contains('/') {
        resolve_workspace_path(prompt_path)
    } else {
        baseline_dir.join(prompt_path)
    };
    let prompt: PromptAssets = serde_json::from_str(
        &fs::read_to_string(&prompt_path)
            .with_context(|| format!("read {}", prompt_path.display()))?,
    )
    .with_context(|| format!("parse {}", prompt_path.display()))?;

    let out_path = meta
        .output_tokens_path
        .as_deref()
        .unwrap_or("output_tokens.json");
    let out_path = if out_path.contains('/') {
        resolve_workspace_path(out_path)
    } else {
        baseline_dir.join(out_path)
    };
    let out: OutputTokens = serde_json::from_str(
        &fs::read_to_string(&out_path).with_context(|| format!("read {}", out_path.display()))?,
    )
    .with_context(|| format!("parse {}", out_path.display()))?;

    Ok((meta, prompt, out))
}

fn expected_generated_tokens(out: &OutputTokens) -> Vec<i64> {
    let mut generated = out.tokens[out.prefill_len..].to_vec();
    if let Some(eos) = out.eos_token_id
        && generated.last().copied() == Some(eos)
    {
        generated.pop();
    }
    generated
}

fn run_one_baseline(baseline_dir: &Path) -> Result<()> {
    let (meta, prompt, out) = load_baseline(baseline_dir)?;
    let image_path = resolve_workspace_path(&meta.image);
    let image = open(&image_path)
        .with_context(|| format!("open baseline image {}", image_path.display()))?;

    let images_dir = baseline_dir.join("images");
    if images_dir.exists() {
        let image_size = meta.image_size.unwrap_or(640);
        let preprocess = dynamic_preprocess(&image, 2, 9, image_size, false);
        if !preprocess.tiles.is_empty() {
            for (idx, tile) in preprocess.tiles.iter().enumerate() {
                let rust_rgb = tile.to_rgb8();
                let python_path = images_dir.join(format!("local_crop_image0_{idx}.png"));
                if python_path.exists() {
                    let python_rgb = open(&python_path)?.to_rgb8();
                    anyhow::ensure!(
                        rust_rgb.dimensions() == python_rgb.dimensions(),
                        "crop {idx} dimension mismatch: rust {:?} vs python {:?}",
                        rust_rgb.dimensions(),
                        python_rgb.dimensions()
                    );
                    let max_diff = rust_rgb
                        .as_raw()
                        .iter()
                        .zip(python_rgb.as_raw().iter())
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap_or(0);
                    anyhow::ensure!(
                        max_diff <= 1,
                        "crop {idx} pixel mismatch detected (max diff {max_diff})"
                    );
                }
            }
        }
    }

    let seq_len = prompt.input_ids.len();
    anyhow::ensure!(
        prompt.prefill_len == seq_len,
        "prefill_len {} != input_ids len {}",
        prompt.prefill_len,
        seq_len
    );
    anyhow::ensure!(
        out.prefill_len == seq_len,
        "output prefill_len {} != prompt len {}",
        out.prefill_len,
        seq_len
    );
    anyhow::ensure!(
        prompt.images_seq_mask.len() == seq_len,
        "images_seq_mask len {} != prompt len {}",
        prompt.images_seq_mask.len(),
        seq_len
    );
    anyhow::ensure!(
        !prompt.image_token_ranges.is_empty(),
        "expected at least one image token range"
    );
    anyhow::ensure!(
        prompt.vision_token_total > 0,
        "expected non-zero vision_token_total"
    );

    let expected = expected_generated_tokens(&out);
    let requested_tokens = out.generated_len;
    let repetition_penalty = 1.0;

    let tokenizer_path = workspace_path("DeepSeek-OCR/tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        anyhow!(
            "failed to load tokenizer {}: {err}",
            tokenizer_path.display()
        )
    })?;

    let variant = meta.variant.as_str();
    if variant != "ocr1" {
        return Ok(());
    }

    with_shared_ocr_model(|model| {
        let base_size = meta.base_size.unwrap_or(1024);
        let image_size = meta.image_size.unwrap_or(640);
        let crop_mode = meta.crop_mode.unwrap_or(true);

        let rendered_prompt = prompt
            .rendered_prompt
            .clone()
            .unwrap_or_else(|| meta.prompt.clone());
        let mut params = DecodeParameters::with_sampling_defaults(requested_tokens);
        params.no_repeat_ngram_size = Some(20);
        params.repetition_penalty = repetition_penalty;
        params.use_cache = true;

        let vision = VisionSettings {
            base_size,
            image_size,
            crop_mode,
        };

        let outcome = model.decode(
            &tokenizer,
            &rendered_prompt,
            std::slice::from_ref(&image),
            vision,
            &params,
            None,
        )?;
        let generated_vec = outcome.generated_tokens;

        let mut debug_logits = None;
        if generated_vec != expected
            && let Some(idx) = first_mismatch(&generated_vec, &expected)
        {
            let prefix = &expected[..idx];
            let mut all_tokens = prompt.input_ids.clone();
            all_tokens.extend_from_slice(prefix);
            let seq_len_dbg = all_tokens.len();
            let input_ids_dbg = Tensor::from_vec(all_tokens, (1, seq_len_dbg), model.device())?
                .to_dtype(DType::I64)?;
            let mut mask_dbg = prompt.images_seq_mask.clone();
            mask_dbg.extend(std::iter::repeat_n(0u8, prefix.len()));
            let mask_dbg = Tensor::from_vec(mask_dbg, (1, seq_len_dbg), model.device())?
                .to_dtype(DType::U8)?;

            let owned_input =
                model.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
            let vision_input = owned_input.as_ref();
            let image_embeddings = model.compute_image_embeddings(&[Some(vision_input)])?;

            let forward = model.forward(
                Some(&input_ids_dbg),
                None,
                None,
                None,
                Some(&mask_dbg),
                None,
                Some(image_embeddings.as_slice()),
                None,
                false,
            )?;
            let logits_last = forward
                .logits
                .get(0)
                .context("missing batch 0")?
                .get(seq_len_dbg - 1)
                .context("missing last timestep")?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let vec = logits_last.to_vec1::<f32>()?;
            let (top1, top1v, top2, top2v) = top2(&vec);
            debug_logits = Some((idx, top1, top1v, top2, top2v));
        }

        let mut generated_vec_no_cache = None;
        if generated_vec != expected
            && std::env::var("DEEPSEEK_OCR_LONG_SKIP_NO_CACHE")
                .ok()
                .as_deref()
                != Some("1")
        {
            let mut params_nc = params.clone();
            params_nc.use_cache = false;
            let outcome_nc = model.decode(
                &tokenizer,
                &rendered_prompt,
                std::slice::from_ref(&image),
                vision,
                &params_nc,
                None,
            )?;
            generated_vec_no_cache = Some(outcome_nc.generated_tokens);
        }

        let got_prefix_ids = generated_vec
            .iter()
            .take(32)
            .map(|&v| v as u32)
            .collect::<Vec<_>>();
        let exp_prefix_ids = expected
            .iter()
            .take(32)
            .map(|&v| v as u32)
            .collect::<Vec<_>>();
        let got_prefix = tokenizer
            .decode(&got_prefix_ids, true)
            .unwrap_or_else(|_| "<decode failed>".to_string());
        let exp_prefix = tokenizer
            .decode(&exp_prefix_ids, true)
            .unwrap_or_else(|_| "<decode failed>".to_string());

        if generated_vec != expected {
            if let Some(idx) = first_mismatch(&generated_vec, &expected) {
                let got = generated_vec.get(idx).copied();
                let exp = expected.get(idx).copied();
                eprintln!(
                    "token mismatch at {idx}: got={got:?} expected={exp:?} (baseline dir {})",
                    baseline_dir.display()
                );
                let ctx_start = idx.saturating_sub(8);
                let ctx_end = (idx + 8).min(generated_vec.len()).min(expected.len());
                eprintln!("context got: {:?}", &generated_vec[ctx_start..ctx_end]);
                eprintln!("context exp: {:?}", &expected[ctx_start..ctx_end]);
                eprintln!("decoded prefix got: {got_prefix}");
                eprintln!("decoded prefix exp: {exp_prefix}");
                if let Some((step, t1, v1, t2, v2)) = debug_logits {
                    eprintln!("logits debug @ step {step}: top1 {t1}={v1}, top2 {t2}={v2}");
                }
            }

            if let Some(vec_nc) = generated_vec_no_cache.as_ref() {
                if let Some(idx_nc) = first_mismatch(vec_nc, &expected) {
                    let got_nc = vec_nc.get(idx_nc).copied();
                    let exp_nc = expected.get(idx_nc).copied();
                    eprintln!("no-cache mismatch at {idx_nc}: got={got_nc:?} expected={exp_nc:?}");
                } else {
                    eprintln!("no-cache generation matches expected tokens");
                }
            }

            anyhow::bail!(
                "generated token sequence diverges for {} ({})",
                variant,
                baseline_dir.display()
            );
        }

        Ok(())
    })
}

fn discover_baseline_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    if !root.exists() {
        return Ok(dirs);
    }

    for entry in fs::read_dir(root).with_context(|| format!("read_dir {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if path.join("baseline.json").exists() {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

#[test]
fn long_generation_baseline() -> Result<()> {
    let baseline_root = workspace_path("baselines/long");
    let dirs = discover_baseline_dirs(&baseline_root)?;
    if dirs.is_empty() {
        eprintln!("no long baselines found under {}", baseline_root.display());
        return Ok(());
    }

    let selected = std::env::var("DEEPSEEK_OCR_LONG_BASELINES").ok();
    let variants = std::env::var("DEEPSEEK_OCR_LONG_VARIANTS").ok();
    let selected: Option<Vec<String>> =
        selected.map(|raw| raw.split(',').map(|s| s.trim().to_string()).collect());
    let variants: Option<Vec<String>> =
        variants.map(|raw| raw.split(',').map(|s| s.trim().to_string()).collect());

    for dir in dirs {
        let name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unknown>")
            .to_string();
        if let Some(list) = &selected
            && !list.iter().any(|sel| sel == &name)
        {
            continue;
        }
        if let Some(list) = &variants {
            let meta: BaselineMetadata =
                serde_json::from_str(&fs::read_to_string(dir.join("baseline.json"))?)?;
            if !list.iter().any(|sel| sel == &meta.variant) {
                continue;
            }
        }
        run_one_baseline(&dir)?;
    }
    Ok(())
}
