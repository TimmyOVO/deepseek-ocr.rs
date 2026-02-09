"""Model adapters for benchmark and correctness gates."""

from benchsuite.models.base import BaseAdapter
from benchsuite.models.deepseek import DeepseekOcr2Adapter, DeepseekOcrAdapter
from benchsuite.models.dots import DotsOcrAdapter
from benchsuite.models.glm import GlmAdapter
from benchsuite.models.paddle import PaddleOcrVlAdapter

__all__ = [
    "BaseAdapter",
    "DeepseekOcrAdapter",
    "DeepseekOcr2Adapter",
    "PaddleOcrVlAdapter",
    "DotsOcrAdapter",
    "GlmAdapter",
]
