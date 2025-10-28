#!/usr/bin/env python3
"""Run DeepSeek-OCR Python benchmark with instrumentation output."""

import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer


def _ensure_package(model_dir: Optional[str]) -> str:
    root = Path(__file__).resolve().parents[1]
    target = Path(model_dir) if model_dir else root / "DeepSeek-OCR"
    target = target.resolve()

    package_name = "deepseek_ocr_pkg"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(target)]  # type: ignore[attr-defined]
        sys.modules[package_name] = pkg

    modules = [
        "conversation",
        "deepencoder",
        "benchmark",
        "modeling_deepseekv2",
        "modeling_deepseekocr",
    ]
    for module in modules:
        module_name = f"{package_name}.{module}"
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, target / f"{module}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module {module_name} from {target}")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        spec.loader.exec_module(loaded)
    return package_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Python benchmark harness")
    parser.add_argument("--model-dir", type=str, help="Path to model directory (default: DeepSeek-OCR)")
    parser.add_argument("--prompt", type=str, help="Inline prompt text")
    parser.add_argument("--prompt-file", type=str, help="Read prompt text from file", default=None)
    parser.add_argument("--image", type=str, required=True, help="Image file used for <image> slots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["auto", "f32", "bf16", "fp16"], default="auto")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--bench-output", type=str, help="Write benchmark JSON output")
    parser.add_argument("--no-bench", action="store_true", help="Disable instrumentation")
    parser.add_argument("--results-dir", type=str, default="outputs")
    parser.add_argument("--stream", action="store_true", help="Stream tokens during decode")
    return parser.parse_args()


def _read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt:
        return prompt
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    raise ValueError("Either --prompt or --prompt-file must be provided.")


def _resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    if spec == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {"f32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    return mapping[spec]


def main() -> None:
    args = parse_args()
    package_name = _ensure_package(args.model_dir)

    benchmark_mod = importlib.import_module(f"{package_name}.benchmark")
    BenchmarkSession = getattr(benchmark_mod, "BenchmarkSession")
    print_summary = getattr(benchmark_mod, "print_summary")
    modeling_mod = importlib.import_module(f"{package_name}.modeling_deepseekocr")
    DeepseekOCRForCausalLM = getattr(modeling_mod, "DeepseekOCRForCausalLM")

    model_dir = Path(args.model_dir) if args.model_dir else Path(__file__).resolve().parents[1] / "DeepSeek-OCR"

    prompt = _read_prompt(args.prompt, args.prompt_file)
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    try:
        importlib.import_module("accelerate")
        low_cpu = True
    except ImportError:
        low_cpu = False
    model = DeepseekOCRForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu,
    )
    model = model.to(device)
    model.eval()

    session = BenchmarkSession(enabled=not args.no_bench, output_path=args.bench_output)
    with session:
        output = model.infer(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=args.image,
            output_path=args.results_dir,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=not args.no_crop,
            eval_mode=not args.stream,
            device=device,
            dtype=dtype,
        )
        if isinstance(output, str) and output:
            print(output)

    print_summary(session if not args.no_bench else None)


if __name__ == "__main__":
    main()
