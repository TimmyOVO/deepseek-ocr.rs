#!/usr/bin/env python3
"""
Dump selected DeepSeek-OCR vision weights from the safetensors shard so they can be
compared against the Candle-loaded tensors in Rust.

By default the script exports all CLIP vision parameters under
`model.vision_model.*` into an NPZ bundle with float32 buffers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from safetensors.torch import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARD = REPO_ROOT / "DeepSeek-OCR" / "model-00001-of-000001.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump DeepSeek-OCR vision weights.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_SHARD,
        help=f"safetensors shard to inspect (default: {DEFAULT_SHARD})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output .npz file.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=["model.vision_model."],
        help=(
            "Weight prefix to export (can be specified multiple times). "
            "Default dumps CLIP vision weights."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Floating-point dtype for the exported arrays (default: fp32).",
    )
    return parser.parse_args()


def select_keys(all_keys: Iterable[str], prefixes: List[str]) -> List[str]:
    selected: List[str] = []
    for key in all_keys:
        if any(key.startswith(prefix) for prefix in prefixes):
            selected.append(key)
    return selected


def to_dtype(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "fp32":
        return tensor.to(torch.float32)
    if dtype == "fp16":
        return tensor.to(torch.float16)
    if dtype == "bf16":
        return tensor.to(torch.bfloat16)
    raise ValueError(f"unsupported dtype: {dtype}")


def k_to_np_key(key: str) -> str:
    return key.replace(".", "__")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with safe_open(args.checkpoint, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        selected = select_keys(all_keys, args.prefix)
        if not selected:
            raise RuntimeError(
                f"no tensors matched prefixes {args.prefix} "
                f"in checkpoint {args.checkpoint}"
            )
        tensors = {}
        for key in selected:
            tensor = f.get_tensor(key)
            converted = to_dtype(tensor, args.dtype).cpu().numpy()
            tensors[k_to_np_key(key)] = converted

    np.savez(args.output, **tensors)
    print(f"[dump] saved {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
