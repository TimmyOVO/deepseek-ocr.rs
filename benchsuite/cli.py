#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchsuite.common import repo_root, write_json
from benchsuite.orchestrator import BenchOrchestrator
from benchsuite.registry import get_adapter

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


_ORCHESTRATOR = BenchOrchestrator()


def _single_job_progress(desc: str):
    if tqdm is None:
        return None
    return tqdm(total=1, desc=desc, unit="job")


def _run_gate(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.model)
    report = adapter.compare_tokens(args.baseline, args.rust)
    out = args.output if args.output else args.rust.parent / "compare.json"
    write_json(out, report)
    print(out)
    return 0 if report["match"] else 1


def _run_bench_python(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.model)
    pbar = _single_job_progress("bench-python")
    try:
        payload = adapter.run_python_bench(
            model_dir=args.model_dir,
            image=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            py_device=args.device,
            py_dtype=args.dtype,
            output=args.output,
            repo_root=repo_root(),
        )
        if pbar is not None:
            pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
    print(args.output)
    if args.print_json:
        import json

        print(json.dumps(payload, ensure_ascii=False))
    return 0


def _run_bench_rust(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.model)
    pbar = _single_job_progress("bench-rust")
    try:
        payload = adapter.run_rust_bench(
            cli=args.cli,
            image=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            rs_device=args.device,
            rs_dtype=args.dtype,
            output=args.output,
            repo_root=repo_root(),
        )
        if pbar is not None:
            pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
    print(args.output)
    if args.print_json:
        import json

        print(json.dumps(payload, ensure_ascii=False))
    return 0


def _run_perf(args: argparse.Namespace) -> int:
    return _ORCHESTRATOR.run_perf(args)


def _run_matrix_gate(args: argparse.Namespace) -> int:
    return _ORCHESTRATOR.run_matrix_gate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchsuite.cli",
        description="Unified benchmark + gate CLI with model adapters",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gate", help="strict token gate: baseline vs rust output")
    p.add_argument("--model", default="glm-ocr")
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--rust", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=_run_gate)

    p = sub.add_parser("bench-python", help="run python benchmark for one model case")
    p.add_argument("--model", default="glm-ocr")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--device", required=True, choices=["cpu", "mps"])
    p.add_argument("--dtype", required=True, choices=["f32", "f16"])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--print-json", action="store_true")
    p.set_defaults(func=_run_bench_python)

    p = sub.add_parser("bench-rust", help="run rust benchmark for one model case")
    p.add_argument("--model", default="glm-ocr")
    p.add_argument("--cli", default=Path("target/release/deepseek-ocr-cli"), type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--device", required=True, choices=["cpu", "metal"])
    p.add_argument("--dtype", required=True, choices=["f32", "f16"])
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--print-json", action="store_true")
    p.set_defaults(func=_run_bench_rust)

    p = sub.add_parser("perf", help="one-command run: py+rust compare with history")
    p.add_argument("--run", help="run id used under baselines/*/runs/<run>")
    p.add_argument("--tag", default="latest")
    p.add_argument("--include-models", nargs="*", default=[])
    p.add_argument("--include-devices", nargs="*", default=[])
    p.add_argument("--include-precision", nargs="*", default=[])
    p.add_argument("--cli", default=Path("target/release/deepseek-ocr-cli"), type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--case-name")
    p.add_argument("--baseline-json", type=Path)
    p.add_argument("--matrix-source", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--prompt")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--cases", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--output-root", type=Path)
    p.set_defaults(func=_run_perf)

    p = sub.add_parser("matrix-gate", help="one-command strict matrix gate run")
    p.add_argument("--run", help="run id used under baselines/*/runs/<run>")
    p.add_argument("--tag", default="latest")
    p.add_argument("--include-models", nargs="*", default=[])
    p.add_argument("--include-devices", nargs="*", default=[])
    p.add_argument("--include-precision", nargs="*", default=[])
    p.add_argument("--cli", default=Path("target/release/deepseek-ocr-cli"), type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--source-matrix", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--case-name", default="adhoc")
    p.add_argument("--image", type=Path)
    p.add_argument("--prompt")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--cases", nargs="*")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=_run_matrix_gate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
