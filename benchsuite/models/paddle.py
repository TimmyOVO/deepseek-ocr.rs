from __future__ import annotations

from pathlib import Path
from typing import Any

from benchsuite.models.base import AdapterCapabilities, BaseAdapter


class PaddleOcrVlAdapter(BaseAdapter):
    model_id = "paddleocr-vl"
    suite_name = "paddle"
    default_model_dir = None
    default_matrix_dir = Path("baselines/sample")
    capabilities = AdapterCapabilities(
        python_baseline=False,
        strict_compare=False,
        rust_infer=True,
        python_skip_reason="paddleocr-vl python baseline is not yet implemented in benchsuite",
        strict_skip_reason="strict compare requires same-precision python baseline output",
    )

    default_prompts = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }
    default_images = {
        "image": "baselines/sample/images/image.png",
        "test": "baselines/sample/images/test.png",
        "test2": "baselines/sample/images/test2.png",
    }
    default_max_new_tokens = (64, 256)

    def default_case_matrix(self, *, root: Path | None = None) -> list[dict[str, Any]]:
        return self.build_static_case_matrix(
            root=root,
            prompts=self.default_prompts,
            images=self.default_images,
            max_new_tokens=self.default_max_new_tokens,
        )
