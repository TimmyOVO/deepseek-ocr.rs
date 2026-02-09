from __future__ import annotations

from pathlib import Path
from typing import Any

from benchsuite.models.base import AdapterCapabilities, BaseAdapter


class DotsOcrAdapter(BaseAdapter):
    model_id = "dots-ocr"
    suite_name = "dots"
    default_model_dir = None
    default_matrix_dir = Path("baselines/sample")
    capabilities = AdapterCapabilities(
        python_baseline=False,
        strict_compare=False,
        rust_infer=True,
        python_skip_reason="dots-ocr python baseline is not yet implemented in benchsuite",
        strict_skip_reason="strict compare requires same-precision python baseline output",
    )

    default_prompts = {
        "prompt_layout_all_en": (
            "Please output the layout information from the PDF image, including each layout element's bbox, "
            "its category, and the corresponding text content within the bbox.\n\n"
            "1. Bbox format: [x1, y1, x2, y2]\n\n"
            "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', "
            "'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
            "3. Text Extraction & Formatting Rules:\n"
            "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
            "    - Formula: Format its text as LaTeX.\n"
            "    - Table: Format its text as HTML.\n"
            "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
            "4. Constraints:\n"
            "    - The output text must be the original text from the image, with no translation.\n"
            "    - All layout elements must be sorted according to human reading order.\n\n"
            "5. Final Output: The entire output must be a single JSON object."
        ),
        "prompt_layout_only_en": (
            "Please output the layout information from this PDF image, including each layout's bbox and its "
            "category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF "
            "document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', "
            "'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. "
            "The layout result should be in JSON format."
        ),
        "prompt_ocr": "Extract the text content from this image.",
        "prompt_grounding_ocr": (
            "Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\n"
            "Bounding Box:\n"
        ),
    }
    default_images = {
        "image": "baselines/sample/images/image.png",
        "test": "baselines/sample/images/test.png",
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
