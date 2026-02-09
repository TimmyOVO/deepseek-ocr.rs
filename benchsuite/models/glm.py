from __future__ import annotations

from pathlib import Path
from typing import Any

from benchsuite.models.base import BaseAdapter


class GlmAdapter(BaseAdapter):
    model_id = "glm-ocr"
    suite_name = "glm"
    default_model_dir = Path(".cli-cache/models/glm-ocr")
    default_matrix_dir = Path("baselines/glm/matrix_v20")
    default_prompts = {
        "text": "Text Recognition:",
        "formula": "Formula Recognition:",
        "table": "Table Recognition:",
    }
    default_images = {
        "image": "baselines/sample/images/image.png",
        "test": "baselines/sample/images/test.png",
        "test2": "baselines/sample/images/test2.png",
        "test3": "baselines/sample/images/test3.png",
    }
    default_max_new_tokens = (8, 64)

    def default_case_matrix(self, *, root: Path | None = None) -> list[dict[str, Any]]:
        root = root or Path(".")
        rows: list[dict[str, Any]] = []
        for prompt_key, prompt in self.default_prompts.items():
            for image_key, image_rel in self.default_images.items():
                image_path = Path(image_rel)
                abs_image = image_path if image_path.is_absolute() else root / image_path
                for max_new in self.default_max_new_tokens:
                    rows.append(
                        {
                            "case": f"{prompt_key}__{image_key}__n{max_new}",
                            "image": str(abs_image),
                            "prompt": prompt,
                            "max_new_tokens": int(max_new),
                        }
                    )
        return rows

    @staticmethod
    def infer_max_new_from_case(case_name: str, fallback: int = 64) -> int:
        if "__n" in case_name:
            tail = case_name.rsplit("__n", 1)[-1]
            if tail.isdigit():
                return int(tail)
        return fallback

    def python_build_model_config(self, cfg_raw: dict[str, Any]) -> Any:
        from transformers.models.glm_ocr import GlmOcrConfig

        return GlmOcrConfig.from_dict(cfg_raw)

    def python_load_processor(self, model_dir: Path) -> Any:
        from transformers import AutoProcessor

        try:
            from transformers.models.glm46v.processing_glm46v import Glm46VProcessor

            return Glm46VProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        except Exception:
            return AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
