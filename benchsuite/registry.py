from __future__ import annotations

from benchsuite.models.glm import GlmAdapter


_REGISTRY = {
    "glm-ocr": GlmAdapter(),
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
    if key not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise SystemExit(f"unsupported model adapter: {name}. supported: {supported}")
    return _REGISTRY[key]
