# Baseline Asset Layout

The `baselines/` directory stores reference outputs captured from the official
Python DeepSeek-OCR pipeline. Candle tests load these artifacts to check tensor
parity, so the layout and data types are fixed. The most recent capture lives in
`baselines/sample/` (last refreshed 2025-10-23) and was generated with
`scripts/capture_baseline.py` using the sample prompt and
`baselines/sample/images/test.png`. The capture script overwrites that folder so
take a copy if you need to keep older assets.

Each run of `scripts/capture_baseline.py` produces the following files inside
the target baseline folder (for example `baselines/sample/`). Unless otherwise
stated, JSON integer arrays correspond to int64 tensors during capture.

## Quick Reference (Rust tests)
- `prompt.json`: use `input_ids`, `images_seq_mask`, `image_token_ranges`, and
  `images_spatial_crop` when constructing prompt-side fixtures. Integer lists
  are serialized as JSON `int`s (load as `i64`/`usize`), while `images_seq_mask`
  is stored as `0/1` and should be treated as `bool`. `vision_token_total`
  equals the sum of all `<image>` spans. `schema_version` (currently `1`)
  should be asserted by the loader.
- `vision_embeddings.npz`: load `global_pre_image{N}` / `local_pre_image{N}` and the
  separated `*_clip_tokens` / `*_sam_tokens` arrays to compare intermediate features.
- `projector_outputs.npz`: `fused_tokens_image{N}` rows align with each `<image>`
  placeholder; `fused_concat` is the concatenation expected by projector parity tests.
- `logits.npz`: contains the teacher-forcing logits matrix plus 1-element arrays for
  `prefill_len` and `generated_len` (both int32).
- `clip_trace.npz`: layerwise CLIP embeddings/pre-norm outputs for global/local views to
  help pinpoint where the Rust transformer diverges from Python.
- `sam_trace.npz`: SAM ViT backbone traces (patch embed, per-block outputs, neck/downsample)
  for global and local crops; useful to isolate feature mismatches before the projector.

## `result.mmd`
- Final markdown emitted by the Python decoder (UTF-8 text). Rust tests can
  load this file directly when checking end-to-end generation parity.

## `images/`
- Drawn overlays used by the Python reference (`result_with_boxes.jpg` and
  supporting assets). These are informational only; current Rust tests do not
  inspect them but we keep them for manual debugging.

## `baseline.json`
- High level metadata recorded when the baseline was captured.  
- Important fields:
  - `model`, `device`, `dtype`: Python capture configuration hints. Candle
    tests can ignore them, but they are useful for debugging drifts.
  - `prompt`: original prompt string (still contains `<image>` placeholders).
  - `image`: path to the source image used by Python.
  - `markdown_path`, `images_dir`: where to find the rendered markdown and the
    draw-overlays produced during capture.
  - `markdown`: sanitized markdown string; duplicated in `output_tokens.json`
    to avoid opening multiple files during quick checks.
  - `prompt_assets_path`, `output_tokens_path`, `image_tensors_path`,
    `vision_embeddings_path`, `projector_outputs_path`, `logits_path`: relative paths to the detailed
    assets described below.
  - `vision_token_total`: number of `<image>` positions in the prompt. This
    equals the count of `True` entries in `prompt.json["images_seq_mask"]`.

## `prompt.json`
- Scalar metadata plus the prompt tensorization results needed for tests.
- Fields (types in parentheses):
  - `schema_version` (`int`): bumped when the JSON layout changes. Tests should
    assert the expected version.
  - `prompt` (`str`): original message block passed into the Python pipeline.
  - `rendered_prompt` (`str`): tokenizer-ready prompt including system/user
    tags exactly as sent to the model.
  - `input_ids` (`List[int]`): full prefill token ids including the leading BOS.
  - `images_seq_mask` (`List[int]`): mask aligned with `input_ids` (`1` marks
    `<image>` positions where projector features were injected; conceptually
    boolean).
  - `image_token_ranges` (`List[{"start": int, "length": int}]`): contiguous
    `<image>` spans describing where vision tokens slot into `input_ids`.
  - `image_token_counts` (`List[int]`): total `<image>` tokens contributed by
    each input image; matches `vision_token_counts`.
  - `images_spatial_crop` (`List[List[int]]`): `[width_tiles, height_tiles]`
    pairs for dynamic cropping.
  - `per_image_patch_counts` (`List[int]`): number of local tiles per image (0
    when only the global view is used).
  - `vision_token_counts` (`List[int]`), `vision_token_total` (`int`): mirror
    the fused projector token counts saved in `projector_outputs.npz`.
  - `bos_token_id`, `image_token_id`, `prefill_len` (`int`), `image_paths`
    (`List[str]`).

## `output_tokens.json`
- Teacher-forcing targets for the language model.
- Fields:
  - `tokens` (`List[int]`): entire sequence (prefill + generated tokens).
  - `prefill_len` (`int`): matches `prompt.json["prefill_len"]`.
  - `generated_len` (`int`): equals `len(tokens) - prefill_len`.
  - `eos_token_id` (`int`): copied from the tokenizer for parity checks.
  - `decoded_markdown` (`str`): final Markdown after Python post-processing.

## `image_tensors.npz`
- Global and local image tensors after Python preprocessing.
- Arrays:
  - `global_views_stack`: `(num_images, 3, base_size, base_size)` float32 tensor
    containing every global view after normalization/padding.
  - `global_view_image{N}`: individual global views for convenience.
  - `local_crops_image{N}`: `(crop_count, 3, image_size, image_size)` float32
    tensor for dynamic crops associated with image `N` (empty arrays when no
    crops are generated).

## `vision_embeddings.npz`
- Float32 tensors captured before the projector.
- Arrays share the naming pattern `*_image{N}` where `N` is the image index.
  - `global_pre_image{N}`: `(global_hw, 2048)` — concatenated CLIP/SAM projector
    inputs for the global 1024×1024 view.
  - `global_clip_tokens_image{N}` / `global_sam_tokens_image{N}`: `(global_hw, 1024)`
    arrays containing the CLIP tokens (class token removed) and flattened SAM
    tokens separately.
  - `local_pre_image{N}`: `(patch_count * patch_hw, 2048)` — concatenated
    CLIP/SAM projector inputs for any cropped tiles (empty when no local crops).
  - `local_clip_tokens_image{N}` / `local_sam_tokens_image{N}`: local CLIP/SAM
    tokens matching the above (empty when no crops are generated).

## `projector_outputs.npz`
- Projector outputs and the fused embeddings inserted into the text stream.
- Arrays (all float32 unless noted):
  - `global_post_image{N}`: `(global_hw, 1280)` projector output per global
    token.
  - `local_post_image{N}`: `(patch_count * patch_hw, 1280)` projector output
    per local token (empty when no crops).
  - `global_tokens_image{N}`: `(global_hw + h, 1280)` — global projector output
    after appending image-newline tokens for each row.
  - `local_tokens_image{N}`: `(patch_token_count, 1280)` — local projector
    output after re-ordering tiles, appending image-newlines, and flattening.
  - `fused_tokens_image{N}`: `(image_token_counts[N], 1280)` — concatenation of
    local tokens, global tokens, and the final view-separator token. These rows
    align with every `<image>` position in `input_ids`.
  - `fused_concat`: concatenation of all `fused_tokens_image{N}` arrays. Length
    equals `vision_token_total`.
  - `image_newline`, `view_separator`: learned projector parameters inserted
    while stitching vision features (shape `(1280,)`).

## `clip_trace.npz`
- Layer-by-layer CLIP transformer dumps for debugging vision parity.
- Arrays (float32, flattened to `[rows, hidden]`):
  - `global_embeddings_image{N}` / `local_embeddings_image{N}`: patch embeddings + class
    token after positional encodings, before the pre-layernorm.
  - `global_pre_layernorm_image{N}` / `local_pre_layernorm_image{N}`: tensors fed into the
    first transformer block.
  - `global_block{L}_image{N}` / `local_block{L}_image{N}`: output of block `L` (0-indexed).
    Rows collapse batch/sequence dimensions (`global`: 1×(seq+1); `local`: patches×(seq+1)).

## `sam_trace.npz`
- Intermediate SAM ViT features saved with original tensor shapes:
  - `global_patch_embed_image{N}` / `local_patch_embed_image{N}`: patch embeddings after
    the initial conv (NHWC order).
  - `global_pos_added_image{N}` / `local_pos_added_image{N}`: embeddings after adding the
    absolute position grid.
  - `global_block{L}_image{N}` / `local_block{L}_image{N}`: transformer block outputs in
    NHWC layout for block `L`.
  - `global_neck_conv{K}_image{N}` / `local_neck_conv{K}_image{N}` (`K` ∈ {1,2}) and
    corresponding `neck_norm{K}` arrays: activations inside the neck sequential conv+LN.
  - `global_net2_image{N}`, `global_net3_image{N}` (and local counterparts): outputs of the
    stride-2 downsample convs that feed the projector (NCHW layout).

## `logits.npz`
- Float32 teacher-forcing logits covering the entire sequence.
- Arrays:
  - `logits`: `(prefill_len + generated_len, vocab_size)` float32 matrix.
  - `prefill_len`, `generated_len`: 1-element int32 arrays mirroring the JSON
    metadata, included for convenience.

All tensors stored in `.npz` files use row-major layout (`C` order) and can be
loaded with `numpy.load(path)["key"]`. Candle tests should map those buffers
into `Tensor`s using the recorded shapes and dtypes. When verifying parity,
remember that both projector outputs and logits are serialized as float32.
