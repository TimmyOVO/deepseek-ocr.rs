#!/usr/bin/env python3
"""
Capture structured DeepSeek-OCR baseline artifacts for Rust parity tests.

This script reproduces the official Python inference path while saving the
tokenized prompt, image masks, vision backbone features, projector outputs,
decoder logits, and final markdown into `baselines/` so the Rust tests can load
the same tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import types

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = REPO_ROOT / "DeepSeek-OCR"
if not LOCAL_MODEL_DIR.exists():
    raise RuntimeError(
        "Local DeepSeek-OCR weights are required (expected at DeepSeek-OCR/). "
        "Run `git submodule update --init --recursive` or download the model first."
    )
if str(LOCAL_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_MODEL_DIR))

PACKAGE_NAME = "deepseek_ocr"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(LOCAL_MODEL_DIR)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE_NAME] = package

from deepseek_ocr.modeling_deepseekocr import (  # type: ignore  # noqa: E402
    BasicImageTransform,
    dynamic_preprocess,
    format_messages,
    load_pil_images,
    process_image_with_refs,
    re_match,
    text_encode,
)
from deepseek_ocr.deepencoder import get_abs_pos_sam  # type: ignore  # noqa: E402

DEFAULT_MODEL_NAME = str(LOCAL_MODEL_DIR)


@dataclass
class PromptArtifacts:
    prompt_rendered: str
    input_ids: torch.Tensor
    images_seq_mask: torch.Tensor
    images_seq_mask_list: List[bool]
    images_spatial_crop: torch.Tensor
    global_views_tensor: torch.Tensor
    global_views_list: List[torch.Tensor]
    crop_tensor: torch.Tensor
    per_image_crops: List[List[torch.Tensor]]
    image_token_ranges: List[Tuple[int, int]]
    image_token_counts: List[int]
    bos_token_id: int
    image_token_id: int
    image_draw: Image.Image
    image_paths: List[str]

    @property
    def prefill_len(self) -> int:
        return int(self.input_ids.shape[0])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture DeepSeek-OCR baseline tensors for Rust parity tests."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model identifier or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt to feed into the model. Include `<image>` tokens if required.",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to an input image.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to store captured artifacts.",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Global view image size passed to the model (default: %(default)s).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Local crop size passed to the model (default: %(default)s).",
    )
    parser.add_argument(
        "--crop-mode",
        action="store_true",
        default=True,
        help="Enable dynamic crop mode (default: enabled).",
    )
    parser.add_argument(
        "--no-crop-mode",
        action="store_false",
        dest="crop_mode",
        help="Disable dynamic crop mode.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Preferred torch dtype (GPU uses bf16/fp16 when available; CPU falls back to fp32).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device override (default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate during baseline capture (default: %(default)s).",
    )
    return parser.parse_args(argv)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def build_prompt_artifacts(
    tokenizer: AutoTokenizer,
    prompt: str,
    image_path: Path,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    dtype: torch.dtype,
    device: torch.device,
    images_dir: Path,
) -> PromptArtifacts:
    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {image_path}")

    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [str(image_path)],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    rendered_prompt = format_messages(
        conversations=conversation, sft_format="plain", system_prompt=""
    )
    pil_images = load_pil_images(conversation)
    if not pil_images:
        raise ValueError("prompt must reference at least one image")

    image_draw = pil_images[0].copy()
    image_transform = BasicImageTransform(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
    )
    patch_size = 16
    downsample_ratio = 4

    image_token = "<image>"
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    if image_token_id is None:
        raise ValueError("tokenizer missing <image> token")
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = 0

    text_splits = rendered_prompt.split(image_token)
    if len(text_splits) != len(pil_images) + 1:
        raise ValueError(
            "rendered prompt must include `<image>` placeholders matching the attached images"
        )

    tokenized: List[int] = []
    mask_flags: List[bool] = []
    global_tensors: List[torch.Tensor] = []
    per_image_crops: List[List[torch.Tensor]] = []
    all_crops: List[torch.Tensor] = []
    spatial_crops: List[List[int]] = []
    image_token_counts: List[int] = []

    images_dir.mkdir(parents=True, exist_ok=True)

    for img_idx, (segment, pil_img) in enumerate(zip(text_splits[:-1], pil_images)):
        segment_ids = text_encode(tokenizer, segment, bos=False, eos=False)
        tokenized.extend(segment_ids)
        mask_flags.extend([False] * len(segment_ids))

        if crop_mode:
            if pil_img.width <= image_size and pil_img.height <= image_size:
                crop_tiles: Sequence[Image.Image] = []
                crop_ratio = [1, 1]
            else:
                crop_tiles, crop_ratio = dynamic_preprocess(
                    pil_img, image_size=image_size
                )
        else:
            crop_tiles = []
            crop_ratio = [1, 1]

        if crop_mode:
            global_target = (base_size, base_size)
            image_for_global = ImageOps.pad(
                pil_img,
                global_target,
                color=tuple(int(x * 255) for x in image_transform.mean),
            )
        else:
            resized = pil_img
            if image_size <= 640:
                resized = pil_img.resize((image_size, image_size))
            image_for_global = ImageOps.pad(
                resized,
                (image_size, image_size),
                color=tuple(int(x * 255) for x in image_transform.mean),
            )

        image_for_global.save(images_dir / f"global_view_image{img_idx}.png")
        global_tensor = image_transform(image_for_global).to(device=device, dtype=dtype)
        global_tensors.append(global_tensor)

        width_crop_num, height_crop_num = int(crop_ratio[0]), int(crop_ratio[1])
        spatial_crops.append([width_crop_num, height_crop_num])

        current_crops: List[torch.Tensor] = []
        if crop_mode and (width_crop_num > 1 or height_crop_num > 1):
            for tile_idx, tile in enumerate(crop_tiles):
                crop_tensor = image_transform(tile).to(device=device, dtype=dtype)
                current_crops.append(crop_tensor)
                all_crops.append(crop_tensor)
                tile.save(images_dir / f"local_crop_image{img_idx}_{tile_idx}.png")
        per_image_crops.append(current_crops)

        if crop_mode:
            num_queries_global = math.ceil((base_size // patch_size) / downsample_ratio)
            num_queries_local = math.ceil((image_size // patch_size) / downsample_ratio)
            tokenized_image = (
                ([image_token_id] * num_queries_global + [image_token_id])
                * num_queries_global
            )
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    ([image_token_id] * (num_queries_local * width_crop_num) + [image_token_id])
                    * (num_queries_local * height_crop_num)
                )
        else:
            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
            tokenized_image += [image_token_id]

        image_token_counts.append(len(tokenized_image))
        tokenized.extend(tokenized_image)
        mask_flags.extend([True] * len(tokenized_image))

    final_segment_ids = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized.extend(final_segment_ids)
    mask_flags.extend([False] * len(final_segment_ids))

    tokenized = [int(bos_id)] + tokenized
    mask_flags = [False] + mask_flags

    input_ids = torch.tensor(tokenized, dtype=torch.long)
    images_seq_mask = torch.tensor(mask_flags, dtype=torch.bool)

    ranges: List[Tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask_flags):
        if flag:
            if start is None:
                start = idx
        elif start is not None:
            ranges.append((start, idx - start))
            start = None
    if start is not None:
        ranges.append((start, len(mask_flags) - start))

    if len(ranges) != len(image_token_counts):
        raise ValueError("image token range count mismatch")

    spatial_tensor = torch.tensor(spatial_crops, dtype=torch.long)
    if global_tensors:
        global_stack = torch.stack(global_tensors, dim=0)
    else:
        global_stack = torch.zeros(
            (1, 3, base_size if crop_mode else image_size, base_size if crop_mode else image_size),
            dtype=dtype,
            device=device,
        )

    if all_crops:
        crop_stack = torch.stack(all_crops, dim=0)
    else:
        crop_stack = torch.zeros(
            (1, 3, base_size, base_size),
            dtype=dtype,
            device=device,
        )

    return PromptArtifacts(
        prompt_rendered=rendered_prompt,
        input_ids=input_ids,
        images_seq_mask=images_seq_mask,
        images_seq_mask_list=list(mask_flags),
        images_spatial_crop=spatial_tensor,
        global_views_tensor=global_stack,
        global_views_list=list(global_tensors),
        crop_tensor=crop_stack,
        per_image_crops=per_image_crops,
        image_token_ranges=ranges,
        image_token_counts=image_token_counts,
        bos_token_id=int(bos_id),
        image_token_id=int(image_token_id),
        image_draw=image_draw,
        image_paths=[str(image_path)],
    )


def compute_clip_trace(
    vision_model,
    pixel_values: torch.Tensor,
    patch_embeds: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    embeddings = vision_model.embeddings(pixel_values, patch_embeds)
    pre_layernorm = vision_model.pre_layrnorm(embeddings)
    hidden = pre_layernorm
    layers: List[torch.Tensor] = []
    for layer in vision_model.transformer.layers:
        hidden = layer(hidden)
        layers.append(hidden)

    trace = {
        "embeddings": embeddings.detach().cpu().to(torch.float32),
        "pre_layernorm": pre_layernorm.detach().cpu().to(torch.float32),
        "layers": [tensor.detach().cpu().to(torch.float32) for tensor in layers],
    }
    return hidden, trace


def compute_sam_trace(
    sam_model,
    pixel_values: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    trace: Dict[str, torch.Tensor] = {}
    x = sam_model.patch_embed(pixel_values)
    trace["patch_embed"] = x.detach().cpu().to(torch.float32)

    if getattr(sam_model, "pos_embed", None) is not None:
        pos = get_abs_pos_sam(sam_model.pos_embed, x.size(1))  # type: ignore[attr-defined]
        if pos.dtype != x.dtype:
            pos = pos.to(x.dtype)
        x = x + pos
    trace["pos_added"] = x.detach().cpu().to(torch.float32)

    for layer_idx, block in enumerate(sam_model.blocks):
        x = block(x)
        trace[f"block{layer_idx}"] = x.detach().cpu().to(torch.float32)

    x = x.permute(0, 3, 1, 2)
    conv1 = sam_model.neck[0](x)
    trace["neck_conv1"] = conv1.detach().cpu().to(torch.float32)
    norm1 = sam_model.neck[1](conv1)
    trace["neck_norm1"] = norm1.detach().cpu().to(torch.float32)
    conv2 = sam_model.neck[2](norm1)
    trace["neck_conv2"] = conv2.detach().cpu().to(torch.float32)
    norm2 = sam_model.neck[3](conv2)
    trace["neck_norm2"] = norm2.detach().cpu().to(torch.float32)

    net2 = sam_model.net_2(norm2)
    trace["net2"] = net2.detach().cpu().to(torch.float32)
    net3 = sam_model.net_3(net2.clone())
    trace["net3"] = net3.detach().cpu().to(torch.float32)

    return net3, trace


def compute_vision_embeddings(
    model: AutoModel,
    artifacts: PromptArtifacts,
    device: torch.device,
) -> Tuple[
    Dict[str, List[torch.Tensor]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    vision_model = model.model.vision_model  # type: ignore[attr-defined]
    sam_model = model.model.sam_model  # type: ignore[attr-defined]
    projector = model.model.projector  # type: ignore[attr-defined]
    image_newline = model.model.image_newline  # type: ignore[attr-defined]
    view_separator = model.model.view_seperator  # type: ignore[attr-defined]

    global_pre_list: List[torch.Tensor] = []
    global_clip_list: List[torch.Tensor] = []
    global_sam_list: List[torch.Tensor] = []
    local_pre_list: List[torch.Tensor] = []
    local_clip_list: List[torch.Tensor] = []
    local_sam_list: List[torch.Tensor] = []
    global_post_list: List[torch.Tensor] = []
    local_post_list: List[torch.Tensor] = []
    global_tokens_list: List[torch.Tensor] = []
    local_tokens_list: List[torch.Tensor] = []
    fused_tokens_list: List[torch.Tensor] = []
    vision_token_counts: List[int] = []
    clip_trace_np: Dict[str, np.ndarray] = {}
    sam_trace_np: Dict[str, np.ndarray] = {}

    newline_param = image_newline.to(device=device)
    view_param_cpu = view_separator.detach().cpu().to(torch.float32)

    with torch.no_grad():
        spatial_crops = artifacts.images_spatial_crop.tolist()
        for idx, (global_view, crops, crop_shape) in enumerate(
            zip(artifacts.global_views_list, artifacts.per_image_crops, spatial_crops)
        ):
            global_tensor = global_view.unsqueeze(0)
            global_sam_trace_tensor: Dict[str, torch.Tensor]
            global_features_1, global_sam_trace_tensor = compute_sam_trace(
                sam_model, global_tensor
            )
            global_trace_tensor: Dict[str, torch.Tensor]
            global_features_2, global_trace_tensor = compute_clip_trace(
                vision_model, global_tensor, global_features_1
            )
            global_pre = torch.cat(
                (global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1
            )
            global_post = projector(global_pre)

            global_pre_flat = (
                global_pre.reshape(-1, global_pre.shape[-1]).detach().cpu().to(torch.float32)
            )
            global_clip_flat = (
                global_features_2[:, 1:]
                .reshape(-1, global_features_2.shape[-1])
                .detach()
                .cpu()
                .to(torch.float32)
            )
            global_sam_flat = (
                global_features_1.flatten(2)
                .permute(0, 2, 1)
                .reshape(-1, global_features_1.shape[1])
                .detach()
                .cpu()
                .to(torch.float32)
            )
            global_post_flat = (
                global_post.reshape(-1, global_post.shape[-1]).detach().cpu().to(torch.float32)
            )

            _, hw, hidden = global_post.shape
            h = w = int(math.isqrt(hw))
            if h * w != hw:
                raise ValueError(f"global feature map is not square: {hw}")
            global_tokens = (
                torch.cat(
                    [
                        global_post.view(h, w, hidden),
                        newline_param[None, None, :].expand(h, 1, hidden),
                    ],
                    dim=1,
                )
                .reshape(-1, hidden)
                .detach()
                .cpu()
                .to(torch.float32)
            )

            width_crop_num = int(crop_shape[0]) if crop_shape else 1
            height_crop_num = int(crop_shape[1]) if crop_shape else 1

            hidden_dim = int(global_trace_tensor["embeddings"].shape[-1])
            clip_trace_np[f"global_embeddings_image{idx}"] = (
                global_trace_tensor["embeddings"].reshape(-1, hidden_dim).numpy()
            )
            clip_trace_np[f"global_pre_layernorm_image{idx}"] = (
                global_trace_tensor["pre_layernorm"].reshape(-1, hidden_dim).numpy()
            )
            for layer_idx, layer_tensor in enumerate(global_trace_tensor["layers"]):
                reshaped = layer_tensor.reshape(
                    layer_tensor.shape[0] * layer_tensor.shape[1], layer_tensor.shape[2]
                )
                clip_trace_np[f"global_block{layer_idx}_image{idx}"] = reshaped.numpy()

            for name, tensor in global_sam_trace_tensor.items():
                sam_trace_np[f"global_{name}_image{idx}"] = tensor.numpy()

            local_pre_flat = torch.zeros((0, global_pre.shape[-1]), dtype=torch.float32)
            local_clip_flat = torch.zeros((0, global_features_2.shape[-1]), dtype=torch.float32)
            local_sam_flat = torch.zeros((0, global_features_1.shape[1]), dtype=torch.float32)
            local_post_flat = torch.zeros((0, global_post.shape[-1]), dtype=torch.float32)
            local_tokens = torch.zeros((0, hidden), dtype=torch.float32)

            if crops:
                patches_tensor = torch.stack(crops, dim=0)
                local_sam_trace_tensor: Dict[str, torch.Tensor]
                local_features_1, local_sam_trace_tensor = compute_sam_trace(
                    sam_model, patches_tensor
                )
                local_trace_tensor: Dict[str, torch.Tensor]
                local_features_2, local_trace_tensor = compute_clip_trace(
                    vision_model, patches_tensor, local_features_1
                )
                local_pre = torch.cat(
                    (local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1
                )
                local_post = projector(local_pre)

                _, hw_local, hidden_local = local_post.shape
                h2 = w2 = int(math.isqrt(hw_local))
                if h2 * w2 != hw_local:
                    raise ValueError(f"local feature map is not square: {hw_local}")
                expected_patches = max(width_crop_num * height_crop_num, 1)
                if local_post.shape[0] != expected_patches:
                    raise ValueError(
                        f"local patch count mismatch for image {idx}: "
                        f"{local_post.shape[0]} vs {expected_patches}"
                    )

                local_tokens_tensor = local_post.view(
                    height_crop_num,
                    width_crop_num,
                    h2,
                    w2,
                    hidden_local,
                )
                local_tokens_tensor = local_tokens_tensor.permute(0, 2, 1, 3, 4).reshape(
                    height_crop_num * h2, width_crop_num * w2, hidden_local
                )
                local_tokens_tensor = torch.cat(
                    [
                        local_tokens_tensor,
                        newline_param[None, None, :].expand(height_crop_num * h2, 1, hidden_local),
                    ],
                    dim=1,
                )
                local_tokens = (
                    local_tokens_tensor.view(-1, hidden_local).detach().cpu().to(torch.float32)
                )
                local_pre_flat = (
                    local_pre.reshape(-1, local_pre.shape[-1]).detach().cpu().to(torch.float32)
                )
                local_clip_flat = (
                    local_features_2[:, 1:]
                    .reshape(-1, local_features_2.shape[-1])
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )
                local_sam_flat = (
                    local_features_1.flatten(2)
                    .permute(0, 2, 1)
                    .reshape(-1, local_features_1.shape[1])
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )
                local_post_flat = (
                    local_post.reshape(-1, local_post.shape[-1]).detach().cpu().to(torch.float32)
                )

                local_hidden_dim = int(local_trace_tensor["embeddings"].shape[-1])
                clip_trace_np[f"local_embeddings_image{idx}"] = (
                    local_trace_tensor["embeddings"].reshape(-1, local_hidden_dim).numpy()
                )
                clip_trace_np[f"local_pre_layernorm_image{idx}"] = (
                    local_trace_tensor["pre_layernorm"].reshape(-1, local_hidden_dim).numpy()
                )
                for layer_idx, layer_tensor in enumerate(local_trace_tensor["layers"]):
                    reshaped = layer_tensor.reshape(
                        layer_tensor.shape[0] * layer_tensor.shape[1], layer_tensor.shape[2]
                    )
                    clip_trace_np[f"local_block{layer_idx}_image{idx}"] = reshaped.numpy()

                for name, tensor in local_sam_trace_tensor.items():
                    sam_trace_np[f"local_{name}_image{idx}"] = tensor.numpy()

            fused = torch.cat(
                [local_tokens, global_tokens, view_param_cpu.unsqueeze(0)],
                dim=0,
            )

            if fused.shape[0] != artifacts.image_token_counts[idx]:
                raise ValueError(
                    f"vision token count mismatch for image {idx}: "
                    f"{fused.shape[0]} vs {artifacts.image_token_counts[idx]}"
                )

            global_pre_list.append(global_pre_flat)
            global_clip_list.append(global_clip_flat)
            global_sam_list.append(global_sam_flat)
            local_pre_list.append(local_pre_flat)
            local_clip_list.append(local_clip_flat)
            local_sam_list.append(local_sam_flat)
            global_post_list.append(global_post_flat)
            local_post_list.append(local_post_flat)
            global_tokens_list.append(global_tokens)
            local_tokens_list.append(local_tokens)
            fused_tokens_list.append(fused)
            vision_token_counts.append(fused.shape[0])

    if fused_tokens_list:
        fused_concat_tensor = torch.cat(fused_tokens_list, dim=0)
    else:
        fused_concat_tensor = torch.zeros(
            (0, view_param_cpu.shape[-1]), dtype=torch.float32, device=device
        )
    fused_concat = fused_concat_tensor.detach().cpu().to(torch.float32)

    vision_data = {
        "global_pre": global_pre_list,
        "global_clip_tokens": global_clip_list,
        "global_sam_tokens": global_sam_list,
        "local_pre": local_pre_list,
        "local_clip_tokens": local_clip_list,
        "local_sam_tokens": local_sam_list,
        "global_post": global_post_list,
        "local_post": local_post_list,
        "global_tokens": global_tokens_list,
        "local_tokens": local_tokens_list,
        "fused_tokens": fused_tokens_list,
        "fused_concat": fused_concat,
        "vision_token_counts": vision_token_counts,
        "image_newline": image_newline.detach().cpu().to(torch.float32),
        "view_separator": view_separator.detach().cpu().to(torch.float32),
    }

    return vision_data, clip_trace_np, sam_trace_np


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)


def capture_baseline(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type != "cuda":
        dtype = torch.float32
        attn_impl = "eager"
    else:
        attn_impl = "flash_attention_2"

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation=attn_impl,
    )
    model = model.eval()
    model = model.to(device=device, dtype=dtype)
    model.disable_torch_init()  # type: ignore[attr-defined]

    artifacts = build_prompt_artifacts(
        tokenizer=tokenizer,
        prompt=args.prompt,
        image_path=args.image,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        dtype=dtype,
        device=device,
        images_dir=output_dir / "images",
    )

    input_ids_batch = artifacts.input_ids.unsqueeze(0).to(device=device)
    attention_mask_batch = torch.ones_like(input_ids_batch, dtype=torch.long, device=device)
    images_seq_mask_batch = artifacts.images_seq_mask.unsqueeze(0).to(device=device)
    images_pair = [(artifacts.crop_tensor, artifacts.global_views_tensor)]
    max_new_tokens = args.max_new_tokens

    autocast_ctx = (
        torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            vision_data, clip_trace_np, sam_trace_np = compute_vision_embeddings(
                model, artifacts, device
            )

            generation = model.generate(  # type: ignore[arg-type]
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                images=images_pair,
                images_seq_mask=images_seq_mask_batch,
                images_spatial_crop=artifacts.images_spatial_crop,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=20,
                use_cache=True,
                return_dict_in_generate=True,
            )

    output_sequences = generation.sequences
    prefill_len = artifacts.prefill_len
    output_tokens = output_sequences[0].tolist()
    generated_len = len(output_tokens) - prefill_len

    decoded_suffix = tokenizer.decode(
        output_sequences[0][prefill_len:], skip_special_tokens=False
    )
    stop_str = "<｜end▁of▁sentence｜>"
    if decoded_suffix.endswith(stop_str):
        decoded_suffix = decoded_suffix[: -len(stop_str)]
    decoded_suffix = decoded_suffix.strip()

    matches_ref, matches_images, matches_other = re_match(decoded_suffix)
    result_image = process_image_with_refs(
        artifacts.image_draw, matches_ref, str(output_dir)
    )

    for idx, match in enumerate(matches_images):
        decoded_suffix = decoded_suffix.replace(
            match, f"![](images/{idx}.jpg)\n"
        )
    for match in matches_other:
        decoded_suffix = (
            decoded_suffix.replace(match, "")
            .replace("\\coloneqq", ":=")
            .replace("\\eqqcolon", "=:")
        )

    markdown_path = output_dir / "result.mmd"
    markdown_path.write_text(decoded_suffix, encoding="utf-8")
    result_image.save(output_dir / "result_with_boxes.jpg")

    attention_mask = torch.ones_like(output_sequences, dtype=torch.long, device=device)
    zeros = torch.zeros((1, max(generated_len, 0)), dtype=torch.bool, device=device)
    full_mask = torch.cat([images_seq_mask_batch, zeros], dim=1)

    with torch.no_grad():
        teacher_out = model(
            input_ids=output_sequences.to(device=device),
            attention_mask=attention_mask,
            images=images_pair,
            images_seq_mask=full_mask,
            images_spatial_crop=artifacts.images_spatial_crop,
            use_cache=False,
            return_dict=True,
        )
    logits = teacher_out.logits.squeeze(0).detach().cpu().to(torch.float32).numpy()

    vision_token_total = int(artifacts.images_seq_mask.sum().item())
    if vision_data["fused_concat"].shape[0] != vision_token_total:
        raise ValueError(
            f"vision fused token count mismatch: "
            f"{vision_data['fused_concat'].shape[0]} vs {vision_token_total}"
        )

    prompt_path = output_dir / "prompt.json"
    output_tokens_path = output_dir / "output_tokens.json"
    vision_embeddings_path = output_dir / "vision_embeddings.npz"
    projector_outputs_path = output_dir / "projector_outputs.npz"
    logits_path = output_dir / "logits.npz"
    image_tensors_path = output_dir / "image_tensors.npz"
    clip_trace_path = output_dir / "clip_trace.npz"
    sam_trace_path = output_dir / "sam_trace.npz"
    metadata_path = output_dir / "baseline.json"

    prompt_data = {
        "schema_version": 1,
        "prompt": args.prompt,
        "rendered_prompt": artifacts.prompt_rendered,
        "input_ids": artifacts.input_ids.tolist(),
        "bos_token_id": artifacts.bos_token_id,
        "image_token_id": artifacts.image_token_id,
        "prefill_len": prefill_len,
        "images_seq_mask": [int(flag) for flag in artifacts.images_seq_mask_list],
        "image_token_ranges": [
            {"start": int(start), "length": int(length)}
            for start, length in artifacts.image_token_ranges
        ],
        "image_token_counts": [int(x) for x in artifacts.image_token_counts],
        "images_spatial_crop": artifacts.images_spatial_crop.tolist(),
        "vision_token_counts": [int(x) for x in vision_data["vision_token_counts"]],
        "vision_token_total": vision_token_total,
        "image_paths": artifacts.image_paths,
        "per_image_patch_counts": [len(crops) for crops in artifacts.per_image_crops],
    }
    prompt_path.write_text(json.dumps(prompt_data, indent=2), encoding="utf-8")

    output_tokens_payload = {
        "tokens": output_tokens,
        "prefill_len": prefill_len,
        "generated_len": generated_len,
        "eos_token_id": tokenizer.eos_token_id,
        "decoded_markdown": decoded_suffix,
    }
    output_tokens_path.write_text(
        json.dumps(output_tokens_payload, indent=2), encoding="utf-8"
    )

    vision_embeddings_np = {
        f"global_pre_image{idx}": tensor.numpy()
        for idx, tensor in enumerate(vision_data["global_pre"])
    }
    for idx, tensor in enumerate(vision_data["global_clip_tokens"]):
        vision_embeddings_np[f"global_clip_tokens_image{idx}"] = tensor.numpy()
    for idx, tensor in enumerate(vision_data["global_sam_tokens"]):
        vision_embeddings_np[f"global_sam_tokens_image{idx}"] = tensor.numpy()
    for idx, tensor in enumerate(vision_data["local_pre"]):
        vision_embeddings_np[f"local_pre_image{idx}"] = tensor.numpy()
    for idx, tensor in enumerate(vision_data["local_clip_tokens"]):
        vision_embeddings_np[f"local_clip_tokens_image{idx}"] = tensor.numpy()
    for idx, tensor in enumerate(vision_data["local_sam_tokens"]):
        vision_embeddings_np[f"local_sam_tokens_image{idx}"] = tensor.numpy()
    save_npz(vision_embeddings_path, vision_embeddings_np)

    projector_outputs_np = {
        f"global_post_image{idx}": tensor.numpy()
        for idx, tensor in enumerate(vision_data["global_post"])
    }
    for idx, tensor in enumerate(vision_data["local_post"]):
        projector_outputs_np[f"local_post_image{idx}"] = tensor.numpy()
    for idx, tensor in enumerate(vision_data["global_tokens"]):
        projector_outputs_np[f"global_tokens_image{idx}"] = tensor.numpy()
    for idx, tensor in enumerate(vision_data["local_tokens"]):
        projector_outputs_np[f"local_tokens_image{idx}"] = tensor.numpy()
    for idx, tensor in enumerate(vision_data["fused_tokens"]):
        projector_outputs_np[f"fused_tokens_image{idx}"] = tensor.numpy()
    projector_outputs_np["fused_concat"] = vision_data["fused_concat"].numpy()
    projector_outputs_np["image_newline"] = vision_data["image_newline"].numpy()
    projector_outputs_np["view_separator"] = vision_data["view_separator"].numpy()
    save_npz(projector_outputs_path, projector_outputs_np)

    if clip_trace_np:
        save_npz(clip_trace_path, clip_trace_np)
    if sam_trace_np:
        save_npz(sam_trace_path, sam_trace_np)

    image_tensors_np: Dict[str, np.ndarray] = {}
    if artifacts.global_views_list:
        stacked_globals = torch.stack(
            [tensor.detach().cpu().to(torch.float32) for tensor in artifacts.global_views_list],
            dim=0,
        )
        image_tensors_np["global_views_stack"] = stacked_globals.numpy()
    else:
        image_tensors_np["global_views_stack"] = np.zeros(
            (0, 3, args.base_size, args.base_size), dtype=np.float32
        )

    for idx, global_view in enumerate(artifacts.global_views_list):
        image_tensors_np[f"global_view_image{idx}"] = (
            global_view.detach().cpu().to(torch.float32).numpy()
        )

    for idx, crops in enumerate(artifacts.per_image_crops):
        if crops:
            stacked = torch.stack(
                [crop.detach().cpu().to(torch.float32) for crop in crops],
                dim=0,
            )
            array = stacked.numpy()
        else:
            array = np.zeros(
                (0, 3, args.image_size, args.image_size), dtype=np.float32
            )
        image_tensors_np[f"local_crops_image{idx}"] = array

    save_npz(image_tensors_path, image_tensors_np)

    save_npz(
        logits_path,
        {
            "logits": logits,
            "prefill_len": np.array([prefill_len], dtype=np.int32),
            "generated_len": np.array([generated_len], dtype=np.int32),
        },
    )

    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "image": str(args.image),
        "base_size": args.base_size,
        "image_size": args.image_size,
        "crop_mode": args.crop_mode,
        "dtype": "fp32" if dtype == torch.float32 else args.dtype,
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "torch_version": torch.__version__,
        "transformers_version": AutoModel.__module__.split(".")[0],
        "markdown_path": str(markdown_path),
        "images_dir": str(output_dir / "images"),
        "prompt_assets_path": str(prompt_path),
        "output_tokens_path": str(output_tokens_path),
        "vision_embeddings_path": str(vision_embeddings_path),
        "projector_outputs_path": str(projector_outputs_path),
        "clip_trace_path": str(clip_trace_path) if clip_trace_np else None,
        "sam_trace_path": str(sam_trace_path) if sam_trace_np else None,
        "logits_path": str(logits_path),
        "image_tensors_path": str(image_tensors_path),
        "vision_token_total": vision_token_total,
        "markdown": decoded_suffix,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[baseline] prompt assets saved to {prompt_path}")
    print(f"[baseline] vision embeddings saved to {vision_embeddings_path}")
    print(f"[baseline] projector outputs saved to {projector_outputs_path}")
    print(f"[baseline] image tensors saved to {image_tensors_path}")
    if clip_trace_np:
        print(f"[baseline] clip trace saved to {clip_trace_path}")
    if sam_trace_np:
        print(f"[baseline] sam trace saved to {sam_trace_path}")
    print(f"[baseline] logits saved to {logits_path}")
    print(f"[baseline] markdown saved to {markdown_path}")
    print(f"[baseline] metadata saved to {metadata_path}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.image.exists():
        sys.exit(f"input image not found: {args.image}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    capture_baseline(args)


if __name__ == "__main__":
    main()
