from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenDiff:
    index: int
    left: int | None
    right: int | None

    def to_dict(self, *, left_key: str, right_key: str) -> dict[str, Any]:
        return {
            "index": self.index,
            left_key: self.left,
            right_key: self.right,
        }


@dataclass(frozen=True)
class BaselineTokens:
    generated_tokens: list[int]
    rendered_prompt: str | None

    @staticmethod
    def from_payload(payload: dict[str, Any], *, token_field: str) -> "BaselineTokens":
        raw = payload.get(token_field)
        if not isinstance(raw, list):
            raise SystemExit(f"baseline json missing token array `{token_field}`")
        tokens = [int(v) for v in raw]
        rendered_prompt = payload.get("rendered_prompt")
        return BaselineTokens(
            generated_tokens=tokens,
            rendered_prompt=rendered_prompt if isinstance(rendered_prompt, str) else None,
        )


@dataclass(frozen=True)
class RustDecodeOutput:
    tokens: list[int]
    prompt_tokens: int
    generated_len: int
    rendered_prompt: str | None

    @staticmethod
    def from_payload(payload: dict[str, Any], *, token_field: str) -> "RustDecodeOutput":
        raw = payload.get(token_field)
        if not isinstance(raw, list):
            raise SystemExit(f"rust output json missing token array `{token_field}`")
        tokens = [int(v) for v in raw]
        prompt_tokens = int(payload.get("prompt_tokens", 0))
        generated_len = int(payload.get("generated_len", len(tokens)))
        rendered_prompt = payload.get("rendered_prompt")
        return RustDecodeOutput(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            generated_len=generated_len,
            rendered_prompt=rendered_prompt if isinstance(rendered_prompt, str) else None,
        )


@dataclass(frozen=True)
class StageTotals:
    by_stage: dict[str, dict[str, Any]]

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "StageTotals":
        by_stage: dict[str, dict[str, Any]] = {}
        for entry in payload.get("stage_totals", []):
            if not isinstance(entry, dict):
                continue
            stage = entry.get("stage")
            if not isinstance(stage, str):
                continue
            by_stage[stage] = entry
        return StageTotals(by_stage=by_stage)

    def stage_ms(self, stage: str) -> float:
        item = self.by_stage.get(stage)
        if item is None:
            return 0.0
        return float(item.get("total_ms", 0.0))

