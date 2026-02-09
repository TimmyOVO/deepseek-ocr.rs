from __future__ import annotations

from benchsuite.models.deepseek import DeepseekOcr2Adapter, DeepseekOcrAdapter
from benchsuite.models.dots import DotsOcrAdapter
from benchsuite.models.glm import GlmAdapter
from benchsuite.models.paddle import PaddleOcrVlAdapter


_REGISTRY = {
    "deepseek-ocr": DeepseekOcrAdapter(),
    "deepseek-ocr-2": DeepseekOcr2Adapter(),
    "paddleocr-vl": PaddleOcrVlAdapter(),
    "dots-ocr": DotsOcrAdapter(),
    "glm-ocr": GlmAdapter(),
}

_ALIASES = {
    "deepseek-ocr2": "deepseek-ocr-2",
}


def list_registered_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def list_default_models() -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for adapter in _REGISTRY.values():
        model_id = str(getattr(adapter, "model_id", "")).strip().lower()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(model_id)
    return models


def get_adapter(name: str):
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _REGISTRY:
        supported = ", ".join(sorted(list(_REGISTRY.keys()) + list(_ALIASES.keys())))
        raise SystemExit(f"unsupported model adapter: {name}. supported: {supported}")
    return _REGISTRY[key]
