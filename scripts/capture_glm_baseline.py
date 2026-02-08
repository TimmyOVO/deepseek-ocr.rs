#!/usr/bin/env python
"""Capture GLM-OCR deterministic baseline artifacts (offline, local-files-only)."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.glm_ocr import GlmOcrConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture GLM-OCR token baseline")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def set_offline_env(repo_root: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(repo_root / ".hf-cache")
    os.environ["TRANSFORMERS_CACHE"] = str(repo_root / ".hf-cache")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_glm_model(model_dir: Path) -> AutoModelForImageTextToText:
    cfg_path = model_dir / "config.json"
    cfg_raw = json.loads(cfg_path.read_text())
    if cfg_raw.get("model_type") != "glm_ocr":
        cfg_raw["model_type"] = "glm_ocr"
        cfg_raw["architectures"] = ["GlmOcrForConditionalGeneration"]
        if "text_config" in cfg_raw:
            cfg_raw["text_config"]["model_type"] = "glm_ocr_text"
        if "vision_config" in cfg_raw:
            cfg_raw["vision_config"]["model_type"] = "glm_ocr_vision"

    config = GlmOcrConfig.from_dict(cfg_raw)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    set_offline_env(repo_root)
    set_seed(0)

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=False,
    )
    model = load_glm_model(args.model_dir)

    image = Image.open(args.image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": args.image.as_posix()},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(images=[image], text=[rendered], return_tensors="pt")
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    prompt_len = int(inputs["input_ids"].shape[1])
    output_ids = generated[0][prompt_len:]
    token_ids = [int(v) for v in output_ids.tolist()]
    decoded = processor.decode(output_ids, skip_special_tokens=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_dir": str(args.model_dir),
                "image": str(args.image),
                "prompt": args.prompt,
                "rendered_prompt": rendered,
                "prompt_tokens": prompt_len,
                "generated_tokens": token_ids,
                "decoded": decoded,
                "seed": 0,
                "do_sample": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(baseline_path)


if __name__ == "__main__":
    main()
