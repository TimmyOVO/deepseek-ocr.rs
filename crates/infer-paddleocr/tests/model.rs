mod model_internal_tests {
    use anyhow::{Context, Result, ensure};
    use ahash::AHashMap;
    use candle_core::{DType, Device, Tensor};
    use deepseek_ocr_core::{ModelKind, ModelLoadArgs, OcrEngine};
    use deepseek_ocr_core::tensor::gather_token_embeddings;
    use deepseek_ocr_infer_paddleocr::{
        PaddleOcrModel,
        config::load_config,
        model::{
            build_prompt_tokens, compute_position_ids, inject_image_embeddings,
            projector_token_count, resolve_eos_token_id,
        },
        vision::{SiglipPreprocessConfig, preprocess_image},
    };
    use ndarray::{Array2, Array3, Array5, Axis, s};
    use ndarray_npy::NpzReader;
    use serde::Deserialize;
    use std::{
        fs::File,
        path::{Path, PathBuf},
    };
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };

    fn asset_path(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate has parent")
            .parent()
            .expect("workspace root")
            .join(relative)
    }

    const SAMPLE_DOC_IMAGES: [&str; 1] = ["baselines/fixtures/paddleocr_vl/fixture_image.png"];
    const MULTI_IMAGE_DOC_IMAGES: [&str; 2] = [
        "baselines/fixtures/paddleocr_vl/fixture_image.png",
        "baselines/fixtures/paddleocr_vl/fixture_receipt.png",
    ];
    const LONG_PROMPT_IMAGES: [&str; 1] = ["baselines/fixtures/paddleocr_vl/fixture_image.png"];

    struct FixtureCase {
        name: &'static str,
        npz: &'static str,
        images: &'static [&'static str],
    }

    const PADDLE_FIXTURES: &[FixtureCase] = &[
        FixtureCase {
            name: "sample_doc",
            npz: "baselines/fixtures/paddleocr_vl/sample_doc.npz",
            images: &SAMPLE_DOC_IMAGES,
        },
        FixtureCase {
            name: "multi_image_doc",
            npz: "baselines/fixtures/paddleocr_vl/multi_image_doc.npz",
            images: &MULTI_IMAGE_DOC_IMAGES,
        },
        FixtureCase {
            name: "long_prompt_doc",
            npz: "baselines/fixtures/paddleocr_vl/long_prompt_doc.npz",
            images: &LONG_PROMPT_IMAGES,
        },
    ];

    fn build_test_tokenizer() -> Tokenizer {
        let mut vocab = AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("<|IMAGE_START|>".to_string(), 1);
        vocab.insert("<|IMAGE_END|>".to_string(), 2);
        vocab.insert("Question:".to_string(), 3);
        vocab.insert("Describe.".to_string(), 4);
        vocab.insert("User:".to_string(), 5);
        vocab.insert("end.".to_string(), 6);
        vocab.insert("</s>".to_string(), 7);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .expect("wordlevel model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
    }

    #[test]
    fn prompt_builder_matches_placeholder_tokens() -> Result<()> {
        let tokenizer = build_test_tokenizer();
        let loaded = load_config(Some(&asset_path("PaddleOCR-VL/config.json")))?;
        let cfg = loaded.value;
        let grid = (1usize, 16usize, 16usize);
        let (tokens, mask) =
            build_prompt_tokens(&tokenizer, "Question: <image> Describe.", &[grid], &cfg)?;
        let placeholders = mask.iter().filter(|&&flag| flag != 0).count();
        assert_eq!(
            placeholders,
            projector_token_count(grid, cfg.vision_config.spatial_merge_size)?
        );
        assert_eq!(tokens.len(), mask.len());
        Ok(())
    }

    #[test]
    fn position_ids_cover_image_span() -> Result<()> {
        let tokenizer = build_test_tokenizer();
        let loaded = load_config(Some(&asset_path("PaddleOCR-VL/config.json")))?;
        let cfg = loaded.value;
        let grid = (1usize, 16usize, 16usize);
        let (tokens, mask) = build_prompt_tokens(&tokenizer, "User: <image> end.", &[grid], &cfg)?;
        let seq_len = tokens.len();
        let ids =
            Tensor::from_vec(tokens.clone(), (1, seq_len), &Device::Cpu)?.to_dtype(DType::I64)?;
        let mask_tensor = Tensor::from_vec(mask.clone(), (1, seq_len), &Device::Cpu)?;
        let (position_ids, deltas) =
            compute_position_ids(&cfg, &ids, Some(&mask_tensor), &[vec![grid]])?;
        assert_eq!(position_ids.shape().dims(), [3, 1, seq_len]);
        assert_eq!(deltas.shape().dims(), [1, 1]);
        let time_axis = position_ids.get(0)?.get(0)?.to_vec1::<i64>()?;
        let image_token_id = cfg.image_token_id.unwrap();
        let first_image_index = tokens
            .iter()
            .position(|&id| id == image_token_id)
            .expect("image token present");
        assert_eq!(
            time_axis[first_image_index],
            time_axis[first_image_index + 1]
        );
        Ok(())
    }

    #[test]
    fn injection_replaces_masked_rows() -> Result<()> {
        let embeddings = Tensor::from_vec(
            (0..12).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 4, 3),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(vec![0u8, 1, 1, 0], (1, 4), &Device::Cpu)?;
        let replacements = Tensor::from_vec(
            vec![100f32, 101., 102., 200., 201., 202.],
            (2, 3),
            &Device::Cpu,
        )?;
        let result = inject_image_embeddings(&embeddings, &mask, &[replacements])?;
        let rows = result.to_vec3::<f32>()?;
        assert_eq!(rows[0][1][0], 100.0);
        assert_eq!(rows[0][2][2], 202.0);
        Ok(())
    }

    #[derive(Deserialize)]
    struct FixtureMetadata {
        prompt: String,
        images: Option<Vec<String>>,
    }

    #[test]
    fn tokenizer_fallback_supplies_eos_token() -> Result<()> {
        let tokenizer = build_test_tokenizer();
        assert_eq!(tokenizer.token_to_id("</s>"), Some(7));
        let mut loaded = load_config(Some(&asset_path("PaddleOCR-VL/config.json")))?;
        loaded.value.eos_token_id = None;
        let eos = resolve_eos_token_id(&loaded.value, &tokenizer);
        assert_eq!(eos, Some(7));
        Ok(())
    }

    #[test]
    fn paddle_fixture_matches_python_reference() -> Result<()> {
        let weights_path = asset_path("PaddleOCR-VL/model.safetensors");
        if !weights_path.exists() {
            eprintln!(
                "skipping PaddleOCR parity test: weights missing at {:?}",
                weights_path
            );
            return Ok(());
        }
        let tokenizer_path = asset_path("PaddleOCR-VL/tokenizer.json");
        if !tokenizer_path.exists() {
            eprintln!(
                "skipping PaddleOCR parity test: tokenizer missing at {:?}",
                tokenizer_path
            );
            return Ok(());
        }

        let device = Device::Cpu;
        let config_path = asset_path("PaddleOCR-VL/config.json");
        let args = ModelLoadArgs {
            kind: ModelKind::PaddleOcrVl,
            config_path: Some(config_path.as_path()),
            weights_path: Some(weights_path.as_path()),
            snapshot_path: None,
            device: device.clone(),
            dtype: DType::F32,
        };
        let model = PaddleOcrModel::load(&args)?;
        let prep_cfg = SiglipPreprocessConfig::from_vision_config(&model.config().vision_config);

        let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
            Ok(tok) => tok,
            Err(err) => {
                eprintln!("skipping PaddleOCR parity test: tokenizer load failed ({err})");
                return Ok(());
            }
        };

        let mut executed = 0usize;
        for case in PADDLE_FIXTURES {
            let fixture_npz = asset_path(case.npz);
            if !fixture_npz.exists() {
                eprintln!(
                    "skipping fixture {}: npz missing at {:?}",
                    case.name, fixture_npz
                );
                continue;
            }
            if let Some(missing_image) = case
                .images
                .iter()
                .map(|relative| asset_path(relative))
                .find(|path| !path.exists())
            {
                eprintln!(
                    "skipping fixture {}: image missing at {:?}",
                    case.name, missing_image
                );
                continue;
            }
            run_fixture_case(case, &fixture_npz, &model, &tokenizer, &prep_cfg)?;
            executed += 1;
        }

        if executed == 0 {
            eprintln!("skipping PaddleOCR parity test: no fixtures present");
        }
        Ok(())
    }

    fn run_fixture_case(
        case: &FixtureCase,
        fixture_npz: &Path,
        model: &PaddleOcrModel,
        tokenizer: &Tokenizer,
        prep_cfg: &SiglipPreprocessConfig,
    ) -> Result<()> {
        let fixture_file = File::open(fixture_npz).context("failed to open fixture npz")?;
        let mut reader = NpzReader::new(fixture_file).context("failed to parse fixture npz")?;
        let input_ids_np: Array2<i64> = reader
            .by_name("input_ids")
            .context("fixture missing input_ids")?;
        let attention_mask_np: Array2<i64> = reader
            .by_name("attention_mask")
            .context("fixture missing attention_mask")?;
        let position_ids_np: Array3<i64> = reader
            .by_name("position_ids")
            .context("fixture missing position_ids")?;
        let rope_deltas_np: Array2<i64> = reader
            .by_name("rope_deltas")
            .context("fixture missing rope_deltas")?;
        let image_grid_np: Array2<i64> = reader
            .by_name("image_grid_thw")
            .context("fixture missing image_grid_thw")?;
        let siglip_hidden_np: Array2<f32> = reader
            .by_name("siglip_hidden")
            .context("fixture missing siglip_hidden")?;
        let siglip_hidden_states_np: Option<Array3<f32>> =
            reader.by_name("siglip_hidden_states").ok();
        if siglip_hidden_states_np.is_some() {
            eprintln!("fixture {} includes siglip_hidden_states", case.name);
        }
        let pixel_values_np: Option<Array5<f32>> = reader.by_name("pixel_values_for_encoder").ok();
        let projector_np: Array2<f32> = reader
            .by_name("projector_embeddings")
            .context("fixture missing projector_embeddings")?;
        let fused_np: Array3<f32> = reader
            .by_name("fused_embeddings")
            .context("fixture missing fused_embeddings")?;
        let logits_np: Array2<f32> = reader
            .by_name("next_token_logits")
            .context("fixture missing next_token_logits")?;
        drop(reader);

        let fixture_json = fixture_npz.with_extension("json");
        let metadata: FixtureMetadata = serde_json::from_reader(
            File::open(&fixture_json).context("failed to read fixture metadata")?,
        )
        .context("failed to parse fixture metadata")?;
        let allow_state_drift = std::env::var_os("SIGLIP_ALLOW_STATE_DRIFT").is_some();

        let mut fixture_grids = Vec::new();
        for row in image_grid_np.axis_iter(Axis(0)) {
            ensure!(row.len() == 3, "grid row must have 3 entries");
            fixture_grids.push((row[0] as usize, row[1] as usize, row[2] as usize));
        }
        ensure!(
            fixture_grids.len() == case.images.len(),
            "fixture {} grid count ({}) mismatches image list ({})",
            case.name,
            fixture_grids.len(),
            case.images.len()
        );
        if let Some(images) = metadata.images {
            ensure!(
                images.len() == case.images.len(),
                "metadata lists {} images but fixture {} expects {}",
                images.len(),
                case.name,
                case.images.len()
            );
        }

        let mut patches = Vec::new();
        let mut pixel_offset = 0usize;
        for (idx, image_rel) in case.images.iter().enumerate() {
            let image_path = asset_path(image_rel);
            let image = image::open(&image_path).with_context(|| {
                format!("failed to open image {:?} for {}", image_path, case.name)
            })?;
            let processed = preprocess_image(&image, model.device(), prep_cfg)
                .with_context(|| format!("failed to preprocess {}", case.name))?;
            ensure!(
                processed.grid_thw == fixture_grids[idx],
                "preprocess grid {:?} mismatches fixture {:?} for {} (image {idx})",
                processed.grid_thw,
                fixture_grids[idx],
                case.name
            );
            if let Some(pixel_np) = pixel_values_np.as_ref() {
                let token_count =
                    processed.grid_thw.0 * processed.grid_thw.1 * processed.grid_thw.2;
                let end = pixel_offset + token_count;
                ensure!(
                    end <= pixel_np.shape()[1],
                    "pixel_values_for_encoder slice exceeds fixture bounds"
                );
                let slice = pixel_np.slice(s![0, pixel_offset..end, .., .., ..]);
                let expected: Vec<f32> = slice.iter().copied().collect();
                let (_, channels, patch_h, patch_w) = processed.patches.shape().dims4()?;
                let per_token = channels * patch_h * patch_w;
                let actual = processed
                    .patches
                    .to_dtype(DType::F32)?
                    .reshape((token_count * per_token,))?
                    .to_vec1::<f32>()?;
                ensure!(
                    actual.len() == expected.len(),
                    "pixel patch length mismatch ({} vs {})",
                    actual.len(),
                    expected.len()
                );
                let mut max_diff = 0f32;
                for (a, b) in actual.iter().zip(expected.iter()) {
                    max_diff = max_diff.max((a - b).abs());
                }
                eprintln!(
                    "{} pixel patch diff for image {} max abs {}",
                    case.name, idx, max_diff
                );
                ensure!(
                    max_diff <= 5e-3,
                    "{} pixel patch mismatch exceeds tolerance (image {idx})",
                    case.name
                );
                pixel_offset = end;
            }
            patches.push(processed);
        }

        let mut siglip_chunks = Vec::new();
        let mut projector_chunks = Vec::new();
        let mut per_layer_state_chunks: Option<Vec<Vec<Tensor>>> = None;
        let capture_states = siglip_hidden_states_np.is_some();
        let merge = model.config().vision_config.spatial_merge_size;
        for (patch, expected_grid) in patches.iter().zip(fixture_grids.iter()) {
            let (vision_hidden, state_list) = if capture_states {
                let (hidden, states) = model.vision_model().forward_with_states(
                    patch,
                    model.config().use_3d_rope,
                    true,
                    model.device(),
                )?;
                (hidden, Some(states))
            } else {
                (
                    model.vision_model().forward(
                        patch,
                        model.config().use_3d_rope,
                        true,
                        model.device(),
                    )?,
                    None,
                )
            };
            let (batch, tokens, hidden) = vision_hidden.shape().dims3()?;
            ensure!(batch == 1, "vision batch must be 1");
            let siglip_flat = vision_hidden
                .reshape((tokens, hidden))?
                .to_dtype(DType::F32)?
                .contiguous()?;
            siglip_chunks.push(siglip_flat.clone());

            if let Some(states) = state_list {
                if per_layer_state_chunks.is_none() {
                    per_layer_state_chunks = Some(vec![Vec::new(); states.len()]);
                }
                let buckets = per_layer_state_chunks
                    .as_mut()
                    .expect("state bucket initialized");
                ensure!(
                    buckets.len() == states.len(),
                    "state bucket count mismatch (expected {}, got {})",
                    buckets.len(),
                    states.len()
                );
                for (bucket, state) in buckets.iter_mut().zip(states.into_iter()) {
                    bucket.push(state);
                }
            }

            let projected = model
                .projector()
                .project_single(&siglip_flat, patch.grid_thw)?;
            ensure!(
                expected_grid.1 % merge == 0 && expected_grid.2 % merge == 0,
                "fixture grid {:?} not divisible by merge size {}",
                expected_grid,
                merge
            );
            let expected_projector_grid = (
                expected_grid.0,
                expected_grid.1 / merge,
                expected_grid.2 / merge,
            );
            ensure!(
                projected.grid == expected_projector_grid,
                "projector grid {:?} mismatches fixture {:?} for {}",
                projected.grid,
                expected_projector_grid,
                case.name
            );
            projector_chunks.push(projected.tokens().to_dtype(DType::F32)?.contiguous()?);
        }

        let siglip_tensor = concat_tensors(siglip_chunks)?.contiguous()?;
        if let (Some(state_chunks), Some(expected_states)) =
            (per_layer_state_chunks, siglip_hidden_states_np.as_ref())
        {
            let tol = 5e-3;
            ensure!(
                state_chunks.len() == expected_states.shape()[0],
                "fixture {} state count mismatch: expected {}, got {}",
                case.name,
                expected_states.shape()[0],
                state_chunks.len()
            );
            for (idx, (bucket, expected_slice)) in state_chunks
                .into_iter()
                .zip(expected_states.axis_iter(Axis(0)))
                .enumerate()
            {
                let tensor = concat_tensors(bucket)?.to_dtype(DType::F32)?.contiguous()?;
                let elem = tensor.shape().elem_count();
                let actual = tensor.reshape((elem,))?.to_vec1::<f32>()?;
                let expected = expected_slice
                    .as_slice()
                    .expect("siglip hidden state slice contiguous");
                ensure!(
                    actual.len() == expected.len(),
                    "{} siglip_hidden_states[{idx}] length mismatch ({} vs {})",
                    case.name,
                    actual.len(),
                    expected.len()
                );
                let mut max_diff = 0f32;
                let mut max_idx = 0usize;
                let mut max_vals = (0f32, 0f32);
                for (pos, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
                    let diff = (a - b).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = pos;
                        max_vals = (*a, *b);
                    }
                }
                let hidden = tensor.shape().dims2()?.1;
                let token_idx = max_idx / hidden;
                let feature_idx = max_idx % hidden;
                eprintln!(
                    "{} siglip_hidden_states[{idx}] max abs diff {max_diff} (token {}, feature {}) rust={} python={}",
                    case.name, token_idx, feature_idx, max_vals.0, max_vals.1
                );
                if allow_state_drift {
                    eprintln!(
                        "{} siglip_hidden_states[{idx}] exceeds tol {} (max diff {})",
                        case.name, tol, max_diff
                    );
                } else {
                    ensure!(
                        max_diff <= tol,
                        "{} siglip_hidden_states[{idx}] max abs diff {} exceeds tolerance {}",
                        case.name,
                        max_diff,
                        tol
                    );
                }
            }
        }

        if let Some(expected_states) = siglip_hidden_states_np.as_ref() {
            let hidden_dim = expected_states.shape()[2];
            let mut offset = 0usize;
            for (img_idx, patch) in patches.iter().enumerate() {
                let token_count = patch.grid_thw.0 * patch.grid_thw.1 * patch.grid_thw.2;
                let end = offset + token_count;
                let embedding_slice = expected_states.slice(s![0, offset..end, ..]);
                let pyro_embed: Vec<f32> = embedding_slice.iter().copied().collect();
                let pyro_tensor =
                    Tensor::from_vec(pyro_embed, (1, token_count, hidden_dim), model.device())?;
                let pyro_states = model.vision_model().encode_hidden_with_states(
                    &pyro_tensor,
                    patch,
                    model.config().use_3d_rope,
                    model.device(),
                )?;
                for (idx, state_tensor) in pyro_states.into_iter().enumerate() {
                    let expected_slice = expected_states.slice(s![idx, offset..end, ..]);
                    let label = format!(
                        "{} encoder_from_python_image{}_layer{}",
                        case.name, img_idx, idx
                    );
                    if allow_state_drift {
                        let stage = state_tensor.to_dtype(DType::F32)?.contiguous()?;
                        let elem = stage.shape().elem_count();
                        let actual = stage.reshape((elem,))?.to_vec1::<f32>()?;
                        let expected_flat = expected_slice
                            .as_slice()
                            .expect("py input state slice contiguous");
                        let mut max_diff = 0f32;
                        for (a, b) in actual.iter().zip(expected_flat.iter()) {
                            max_diff = max_diff.max((a - b).abs());
                        }
                        eprintln!("{label} max abs diff {}", max_diff);
                    } else {
                        assert_close(
                            &state_tensor,
                            expected_slice
                                .as_slice()
                                .expect("py input state slice contiguous"),
                            5e-3,
                            &label,
                        )?;
                    }
                }
                offset = end;
            }

            if case.name == "sample_doc" && patches.len() == 1 {
                let debug_path = asset_path("outputs/sample_doc_layer17_debug.npz");
                if debug_path.exists() {
                    eprintln!(
                        "sample_doc layer17 debug instrumentation active at {}",
                        debug_path.display()
                    );
                    let mut debug_reader = NpzReader::new(File::open(&debug_path)?)
                        .context("failed to read layer17 debug npz")?;
                    let debug_norm1: Array2<f32> = debug_reader
                        .by_name("norm1")
                        .context("layer17 debug missing norm1")?;
                    let debug_attn: Array2<f32> = debug_reader
                        .by_name("attn_out")
                        .context("layer17 debug missing attn_out")?;
                    let debug_after_attn: Array2<f32> = debug_reader
                        .by_name("after_attn")
                        .context("layer17 debug missing after_attn")?;
                    let debug_norm2: Array2<f32> = debug_reader
                        .by_name("norm2")
                        .context("layer17 debug missing norm2")?;
                    let debug_mlp: Array2<f32> = debug_reader
                        .by_name("mlp_out")
                        .context("layer17 debug missing mlp_out")?;
                    let debug_output: Array2<f32> = debug_reader
                        .by_name("output")
                        .context("layer17 debug missing output")?;

                    let layer_index = 17usize;
                    let patch = &patches[0];
                    let token_count = patch.grid_thw.0 * patch.grid_thw.1 * patch.grid_thw.2;
                    let layer_input = expected_states.slice(s![layer_index, 0..token_count, ..]);
                    let pyro_vec: Vec<f32> = layer_input.iter().copied().collect();
                    let pyro_tensor =
                        Tensor::from_vec(pyro_vec, (1, token_count, hidden_dim), model.device())?;
                    let debug = model.vision_model().debug_layer_outputs(
                        &pyro_tensor,
                        patch,
                        layer_index,
                        model.config().use_3d_rope,
                        model.device(),
                    )?;
                    let report_stage =
                        |tensor: &Tensor, expected: &Array2<f32>, label: &str| -> Result<()> {
                            let stage = tensor
                                .to_dtype(DType::F32)?
                                .reshape((token_count, hidden_dim))?
                                .contiguous()?;
                            let elem = stage.shape().elem_count();
                            let actual = stage.reshape((elem,))?.to_vec1::<f32>()?;
                            let expected_slice = expected
                                .as_slice()
                                .context("layer17 debug slice not contiguous")?;
                            let mut max_diff = 0f32;
                            for (a, b) in actual.iter().zip(expected_slice.iter()) {
                                max_diff = max_diff.max((a - b).abs());
                            }
                            eprintln!("sample_doc debug {} max diff {}", label, max_diff);
                            Ok(())
                        };
                    report_stage(&debug.norm1, &debug_norm1, "layer17 norm1")?;
                    report_stage(&debug.attn_out, &debug_attn, "layer17 attn_out")?;
                    report_stage(&debug.after_attn, &debug_after_attn, "layer17 after_attn")?;
                    report_stage(&debug.norm2, &debug_norm2, "layer17 norm2")?;
                    report_stage(&debug.mlp_out, &debug_mlp, "layer17 mlp_out")?;
                    report_stage(&debug.output, &debug_output, "layer17 output")?;
                } else {
                    eprintln!(
                        "sample_doc layer17 debug npz missing at {}",
                        debug_path.display()
                    );
                }
            }
        }

        if case.name == "sample_doc" {
            let py_pos_path = Path::new("outputs/py_pos_embed.npy");
            if py_pos_path.exists() && !patches.is_empty() {
                let py_pos: Array2<f32> = ndarray_npy::read_npy(py_pos_path)
                    .context("failed to read python positional embedding dump")?;
                let candle_pos = model
                    .vision_model()
                    .debug_positional_encoding(patches[0].grid_thw, model.device())?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                assert_close(
                    &candle_pos,
                    py_pos
                        .as_slice()
                        .expect("python positional embed contiguous"),
                    5e-3,
                    "siglip positional embedding (sample_doc)",
                )?;
            }
        }

        assert_close(
            &siglip_tensor,
            siglip_hidden_np
                .as_slice()
                .expect("siglip array contiguous"),
            5e-3,
            &format!("{} siglip_hidden", case.name),
        )?;

        let projector_tensor = concat_tensors(projector_chunks)?.contiguous()?;
        assert_close(
            &projector_tensor,
            projector_np.as_slice().expect("projector array contiguous"),
            5e-3,
            &format!("{} projector_embeddings", case.name),
        )?;

        let grids = fixture_grids.clone();
        let (prompt_tokens, image_mask_vec) =
            build_prompt_tokens(tokenizer, &metadata.prompt, &grids, model.config())?;
        let fixture_tokens: Vec<i64> = input_ids_np.iter().copied().collect();
        assert_eq!(
            prompt_tokens, fixture_tokens,
            "{} prompt tokens diverged",
            case.name
        );

        let seq_len = fixture_tokens.len();
        let vocab = logits_np.shape()[1];
        let input_ids_tensor =
            Tensor::from_vec(fixture_tokens.clone(), (1, seq_len), model.device())?
                .to_dtype(DType::I64)?;
        let attention_mask_vec: Vec<u8> = attention_mask_np
            .iter()
            .map(|&v| u8::try_from(v).unwrap_or(0))
            .collect();
        let attention_mask_tensor =
            Tensor::from_vec(attention_mask_vec.clone(), (1, seq_len), model.device())?
                .to_dtype(DType::U8)?;

        let base_embeddings =
            gather_token_embeddings(model.decoder().embed_tokens(), &input_ids_tensor)?;
        let image_mask_tensor =
            Tensor::from_vec(image_mask_vec.clone(), (1, seq_len), model.device())?
                .to_dtype(DType::U8)?;
        let fused = inject_image_embeddings(
            &base_embeddings,
            &image_mask_tensor,
            std::slice::from_ref(&projector_tensor),
        )?
        .to_dtype(DType::F32)?;
        assert_close(
            &fused,
            fused_np.as_slice().expect("fused array contiguous"),
            5e-3,
            &format!("{} fused_embeddings", case.name),
        )?;

        let image_grid_metadata = vec![grids.clone()];
        let (position_ids, rope_deltas) = compute_position_ids(
            model.config(),
            &input_ids_tensor,
            Some(&attention_mask_tensor),
            &image_grid_metadata,
        )?;
        assert_int_match(
            &position_ids,
            position_ids_np.as_slice().expect("position ids contiguous"),
            &format!("{} position_ids", case.name),
        )?;
        assert_int_match(
            &rope_deltas,
            rope_deltas_np.as_slice().expect("rope deltas contiguous"),
            &format!("{} rope_deltas", case.name),
        )?;

        let decoder = model.decoder();
        let prefill = decoder.forward(
            None,
            Some(&fused),
            Some(&attention_mask_tensor),
            Some(&position_ids),
            None,
            false,
        )?;
        let logits = prefill
            .logits
            .get(0)?
            .get(seq_len - 1)?
            .to_dtype(DType::F32)?;
        assert_eq!(
            logits.shape().dims1()?,
            vocab,
            "logit dimension mismatch for {}",
            case.name
        );
        assert_close(
            &logits,
            logits_np.as_slice().expect("logits contiguous"),
            1e-2,
            &format!("{} next_token_logits", case.name),
        )?;

        Ok(())
    }

    fn concat_tensors(mut tensors: Vec<Tensor>) -> Result<Tensor> {
        ensure!(!tensors.is_empty(), "cannot concatenate empty tensor list");
        if tensors.len() == 1 {
            Ok(tensors.pop().expect("length checked above"))
        } else {
            let refs: Vec<&Tensor> = tensors.iter().collect();
            Ok(Tensor::cat(&refs, 0)?)
        }
    }

    fn assert_close(tensor: &Tensor, expected: &[f32], tol: f32, label: &str) -> Result<()> {
        let elem = tensor.shape().elem_count();
        let actual = tensor.reshape((elem,))?.to_vec1::<f32>()?;
        ensure!(
            actual.len() == expected.len(),
            "{} length mismatch ({} vs {})",
            label,
            actual.len(),
            expected.len()
        );
        let mut max_diff = 0f32;
        for (a, b) in actual.iter().zip(expected.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff <= tol,
            "{} max abs diff {} exceeds tolerance {}",
            label,
            max_diff,
            tol
        );
        Ok(())
    }

    fn assert_int_match(tensor: &Tensor, expected: &[i64], label: &str) -> Result<()> {
        let elem = tensor.shape().elem_count();
        let actual = tensor.reshape((elem,))?.to_vec1::<i64>()?;
        ensure!(
            actual == expected,
            "{} mismatch between tensor ({:?}...) and fixture ({:?}...)",
            label,
            &actual[..actual.len().min(4)],
            &expected[..expected.len().min(4)]
        );
        Ok(())
    }

}
