from __future__ import annotations

from pathlib import Path
from typing import Any

from benchsuite.models.base import AdapterCapabilities, BaseAdapter


class GlmAdapter(BaseAdapter):
    python_interpreter_env_key = "BENCHSUITE_PY_GLM"
    python_runtime_env_name = "glm"
    python_runtime_extras = ("bench", "bench-glm")
    model_id = "glm-ocr"
    suite_name = "glm"
    hf_repo_id = "zai-org/GLM-OCR"
    default_model_dir = None
    default_matrix_dir = Path("baselines/glm/matrix_v20")
    capabilities = AdapterCapabilities(
        python_baseline=True,
        strict_compare=True,
        rust_infer=True,
    )

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
        return self.build_static_case_matrix(
            root=root,
            prompts=self.default_prompts,
            images=self.default_images,
            max_new_tokens=self.default_max_new_tokens,
        )

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

        _ = model_dir
        return AutoProcessor.from_pretrained(self.hf_repo_id, use_fast=False)

    def python_prepare_inputs(self, processor: Any, *, image: Path, prompt: str, device: Any) -> tuple[str, dict[str, Any]]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": str(image),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        inputs.pop("token_type_ids", None)
        rendered = self.normalize_prompt(prompt)
        return rendered, dict(inputs)
