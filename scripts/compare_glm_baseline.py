#!/usr/bin/env python
"""Compare Rust GLM output json against Python baseline with strict token equality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GLM token baseline")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    return parser.parse_args()


def earliest_divergence(a: list[int], b: list[int]) -> tuple[int, int | None, int | None] | None:
    upto = min(len(a), len(b))
    for idx in range(upto):
        if a[idx] != b[idx]:
            return idx, a[idx], b[idx]
    if len(a) != len(b):
        idx = upto
        return idx, a[idx] if idx < len(a) else None, b[idx] if idx < len(b) else None
    return None


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    rust = json.loads(args.rust.read_text())

    bt = baseline.get("generated_tokens")
    rt = rust.get("tokens")
    if not isinstance(bt, list) or not isinstance(rt, list):
        raise SystemExit("baseline/rust json missing token arrays")

    diff = earliest_divergence(bt, rt)
    report = {
        "match": diff is None,
        "baseline_tokens": len(bt),
        "rust_tokens": len(rt),
    }
    if diff is not None:
        idx, b, r = diff
        report["earliest_divergence"] = {
            "index": idx,
            "baseline": b,
            "rust": r,
        }

    out = args.rust.parent / "compare.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(out)
    if diff is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
