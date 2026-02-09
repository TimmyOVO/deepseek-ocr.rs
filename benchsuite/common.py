from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME_DIR_NAME = "deepseek-ocr-benchsuite"
RUNTIME_ROOT_ENV_KEYS = ("BENCHSUITE_RUNTIME_ROOT", "DEEPSEEK_OCR_RUNTIME_ROOT")


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    hf_home: Path
    hf_transformers_cache: Path
    hf_hub_cache: Path
    cli_config_dir: Path
    cli_cache_dir: Path
    cli_models_dir: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_runtime_root(override: Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser()

    for key in RUNTIME_ROOT_ENV_KEYS:
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser()

    return Path("/tmp") / DEFAULT_RUNTIME_DIR_NAME


def runtime_paths(*, runtime_root: Path | None = None, create_dirs: bool = True) -> RuntimePaths:
    root = resolve_runtime_root(runtime_root)
    hf_home = root / "huggingface"
    hf_transformers_cache = hf_home / "transformers"
    hf_hub_cache = hf_home / "hub"
    cli_config_dir = root / "deepseek-ocr-config"
    cli_cache_dir = root / "deepseek-ocr-cache"
    cli_models_dir = cli_cache_dir / "models"

    if create_dirs:
        for path in [
            root,
            hf_home,
            hf_transformers_cache,
            hf_hub_cache,
            cli_config_dir,
            cli_cache_dir,
            cli_models_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    return RuntimePaths(
        root=root,
        hf_home=hf_home,
        hf_transformers_cache=hf_transformers_cache,
        hf_hub_cache=hf_hub_cache,
        cli_config_dir=cli_config_dir,
        cli_cache_dir=cli_cache_dir,
        cli_models_dir=cli_models_dir,
    )


def runtime_env(*, runtime_root: Path | None = None) -> dict[str, str]:
    paths = runtime_paths(runtime_root=runtime_root, create_dirs=True)
    env = os.environ.copy()
    env["HF_HOME"] = str(paths.hf_home)
    env["TRANSFORMERS_CACHE"] = str(paths.hf_transformers_cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(paths.hf_hub_cache)
    env["DEEPSEEK_OCR_CONFIG_DIR"] = str(paths.cli_config_dir)
    env["DEEPSEEK_OCR_CACHE_DIR"] = str(paths.cli_cache_dir)
    return env


def offline_env(root: Path | None = None, runtime_root: Path | None = None) -> dict[str, str]:
    _ = root
    return runtime_env(runtime_root=runtime_root)


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
