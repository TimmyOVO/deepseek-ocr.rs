from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def offline_env(root: Path | None = None) -> dict[str, str]:
    root = root or repo_root()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HOME"] = str(root / ".hf-cache")
    env["TRANSFORMERS_CACHE"] = str(root / ".hf-cache")
    env["DEEPSEEK_OCR_CONFIG_DIR"] = str(root / ".cli-config")
    env["DEEPSEEK_OCR_CACHE_DIR"] = str(root / ".cli-cache")
    return env


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def earliest_divergence(a: list[int], b: list[int]) -> tuple[int, int | None, int | None] | None:
    upto = min(len(a), len(b))
    for idx in range(upto):
        if a[idx] != b[idx]:
            return idx, a[idx], b[idx]
    if len(a) != len(b):
        idx = upto
        return idx, a[idx] if idx < len(a) else None, b[idx] if idx < len(b) else None
    return None


def has_mps() -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False
    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())

