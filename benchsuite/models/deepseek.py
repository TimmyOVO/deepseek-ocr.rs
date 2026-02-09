from __future__ import annotations

from pathlib import Path
from typing import Any

from benchsuite.models.base import AdapterCapabilities, BaseAdapter


class _DeepseekFamilyAdapter(BaseAdapter):
    suite_name = "deepseek"
    default_matrix_dir = Path("baselines/long")
    capabilities = AdapterCapabilities(
        python_baseline=False,
        strict_compare=False,
        rust_infer=True,
        python_skip_reason="deepseek python baseline requires the upstream DeepSeek-OCR Python package layout",
        strict_skip_reason="strict compare requires same-precision python baseline output",
    )

    default_images = {
        "image": "baselines/sample/images/image.png",
        "test": "baselines/sample/images/test.png",
        "test2": "baselines/sample/images/test2.png",
        "test3": "baselines/sample/images/test3.png",
    }
    default_max_new_tokens = (64, 256)

    def default_case_matrix(self, *, root: Path | None = None) -> list[dict[str, Any]]:
        return self.build_static_case_matrix(
            root=root,
            prompts=self.default_prompts,
            images=self.default_images,
            max_new_tokens=self.default_max_new_tokens,
        )


class DeepseekOcrAdapter(_DeepseekFamilyAdapter):
    model_id = "deepseek-ocr"
    default_model_dir = None
    default_prompts = {
        "grounding_md": "<|grounding|>Convert the document to markdown.",
        "describe": "Describe this image in detail.",
    }


class DeepseekOcr2Adapter(_DeepseekFamilyAdapter):
    model_id = "deepseek-ocr-2"
    default_model_dir = None
    default_prompts = {
        "describe": "Describe this image in detail.",
        "grounding_md": "<|grounding|>Convert the document to markdown.",
    }
