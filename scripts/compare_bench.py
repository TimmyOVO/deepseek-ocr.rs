#!/usr/bin/env python3
"""Compare benchmark JSON outputs from Rust/Python runs."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_stage_totals(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    totals = data.get("stage_totals", [])
    mapping: Dict[str, Dict[str, float]] = {}
    for entry in totals:
        stage = entry.get("stage")
        if not stage:
            continue
        mapping[stage] = {
            "count": float(entry.get("count", 0.0)),
            "total_ms": float(entry.get("total_ms", 0.0)),
            "avg_ms": float(entry.get("avg_ms", 0.0)),
            "min_ms": float(entry.get("min_ms", 0.0)),
            "max_ms": float(entry.get("max_ms", 0.0)),
        }
    return mapping


FRIENDLY_STAGE_NAMES: Dict[str, str] = {
    "decode.generate": "Decode – Overall",
    "decode.iterative": "Decode – Token Loop",
    "decode.prefill": "Decode – Prompt Prefill",
    "decode.generate_no_cache": "Decode – No Cache",
    "prompt.build_tokens": "Prompt – Build Tokens",
    "prompt.render": "Prompt – Render Template",
    "vision.compute_embeddings": "Vision – Embed Images",
    "vision.prepare_inputs": "Vision – Prepare Inputs",
    "vision.compute_projection": "Vision – Project Features",
    "vision.prepare_masks": "Vision – Prepare Masks",
}


def friendly_stage_name(stage: str) -> str:
    if stage in FRIENDLY_STAGE_NAMES:
        return f"{FRIENDLY_STAGE_NAMES[stage]} ({stage})"
    cleaned = stage.replace(".", " ").replace("_", " ")
    pretty = " ".join(part.capitalize() for part in cleaned.split())
    if not pretty:
        return stage
    return f"{pretty} ({stage})"


def render_table(header: List[str], rows: List[List[str]]) -> None:
    table = [header] + rows
    widths = [max(len(row[idx]) for row in table) for idx in range(len(header))]

    def fmt(row: Iterable[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    print(fmt(header))
    separator = "-+-".join("-" * width for width in widths)
    print(separator)
    for row in rows:
        print(fmt(row))


def compare(reference: Dict[str, Dict[str, float]], targets: List[Tuple[str, Dict[str, Dict[str, float]]]]) -> None:
    header = ["stage", "ref total (ms)", "ref avg (ms)"]
    for label, _ in targets:
        header.extend([f"{label} total", f"{label}/ref"])

    stages = sorted(reference.keys() | {stage for _, data in targets for stage in data.keys()})
    rows: List[List[str]] = []
    for stage in stages:
        ref = reference.get(stage)
        row = [friendly_stage_name(stage)]
        if ref:
            row.append(f"{ref['total_ms']:.3f}")
            row.append(f"{ref['avg_ms']:.3f}")
        else:
            row.extend(["-", "-"])
        for _, data in targets:
            target = data.get(stage)
            if target:
                row.append(f"{target['total_ms']:.3f}")
                if ref and ref["total_ms"] > 0:
                    ratio = target["total_ms"] / ref["total_ms"]
                    row.append(f"{ratio:.2f}x")
                else:
                    row.append("-")
            else:
                row.extend(["-", "-"])
        rows.append(row)

    render_table(header, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark JSON outputs")
    parser.add_argument("reference", type=Path, help="Reference benchmark JSON (e.g., Rust)")
    parser.add_argument("targets", nargs="+", type=Path, help="Benchmark JSON files to compare")
    parser.add_argument("--labels", nargs="+", help="Optional labels for targets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = load_stage_totals(args.reference)
    labels = args.labels if args.labels else [path.stem for path in args.targets]
    if len(labels) != len(args.targets):
        raise ValueError("Number of labels must match number of target files")
    targets = [(label, load_stage_totals(path)) for label, path in zip(labels, args.targets)]
    compare(reference, targets)


if __name__ == "__main__":
    main()
