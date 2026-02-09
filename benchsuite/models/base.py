from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import random
import subprocess
import time
from typing import Any

from benchsuite.common import earliest_divergence, offline_env, read_json, write_json
from benchsuite.schemas import BaselineTokens, RustDecodeOutput, StageTotals, TokenDiff


class BaseAdapter(ABC):
    model_id: str
    suite_name: str
    default_model_dir: Path
    default_matrix_dir: Path

    supported_devices: tuple[str, ...] = ("cpu", "mps")
    supported_precisions: tuple[str, ...] = ("f32", "f16")
    disallowed_device_precision: set[tuple[str, str]] = {("cpu", "f16")}
    rs_device_map: dict[str, str] = {"cpu": "cpu", "mps": "metal"}

    baseline_token_field: str = "generated_tokens"
    rust_token_field: str = "tokens"
    rust_prompt_tokens_field: str = "prompt_tokens"
    rust_generated_len_field: str = "generated_len"
    rust_rendered_prompt_field: str = "rendered_prompt"

    bench_stage_load: str = "model.load"
    bench_stage_prefill: tuple[str, ...] = ("prompt.render", "vision.prepare_inputs", "decode.prefill")
    bench_stage_decode: str = "decode.iterative"
    bench_stage_total: str = "decode.generate"

    def build_device_precision_matrix(
        self,
        *,
        devices: list[str],
        precisions: list[str],
        mps_available: bool,
    ) -> list[dict[str, str]]:
        pairs: list[dict[str, str]] = []
        for device in devices:
            if device not in self.supported_devices:
                supported = ",".join(self.supported_devices)
                raise SystemExit(f"unsupported device for {self.model_id}: {device}; supported: {supported}")
            if device == "mps" and not mps_available:
                continue

            for dtype in precisions:
                if dtype not in self.supported_precisions:
                    supported = ",".join(self.supported_precisions)
                    raise SystemExit(f"unsupported precision for {self.model_id}: {dtype}; supported: {supported}")
                if (device, dtype) in self.disallowed_device_precision:
                    continue

                rs_device = self.rs_device_map.get(device)
                if rs_device is None:
                    raise SystemExit(f"no rust device mapping for {self.model_id}: {device}")

                pairs.append(
                    {
                        "py_device": device,
                        "py_dtype": dtype,
                        "rs_device": rs_device,
                        "rs_dtype": dtype,
                    }
                )

        dedup: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pair in pairs:
            key = (pair["py_device"], pair["py_dtype"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(pair)
        return dedup

    @abstractmethod
    def default_case_matrix(self, *, root: Path | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def normalize_prompt(self, prompt: str) -> str:
        return prompt if "<image>" in prompt else f"<image>{prompt}"

    def compare_tokens(self, baseline_path: Path, rust_output_path: Path) -> dict[str, Any]:
        baseline_payload = read_json(baseline_path)
        rust_payload = read_json(rust_output_path)
        baseline = BaselineTokens.from_payload(baseline_payload, token_field=self.baseline_token_field)
        rust = RustDecodeOutput.from_payload(rust_payload, token_field=self.rust_token_field)

        raw_diff = earliest_divergence(baseline.generated_tokens, rust.tokens)
        diff = None if raw_diff is None else TokenDiff(index=raw_diff[0], left=raw_diff[1], right=raw_diff[2])

        report: dict[str, Any] = {
            "match": diff is None,
            "baseline_tokens": len(baseline.generated_tokens),
            "rust_tokens": len(rust.tokens),
        }
        if diff is not None:
            report["earliest_divergence"] = diff.to_dict(left_key="baseline", right_key="rust")
        return report

    def run_rust_infer(
        self,
        *,
        cli: Path,
        image: Path,
        prompt: str,
        max_new_tokens: int,
        rs_device: str,
        rs_dtype: str,
        output: Path,
        repo_root: Path,
        rendered_prompt: str | None = None,
    ) -> dict[str, Any]:
        env = offline_env(repo_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        _ = rendered_prompt

        cmd = [
            str(cli),
            "--model",
            self.model_id,
            "--image",
            str(image),
            "--device",
            rs_device,
            "--dtype",
            rs_dtype,
            "--max-new-tokens",
            str(max_new_tokens),
            "--output-json",
            str(output),
            "--quiet",
            "--prompt",
            self.normalize_prompt(prompt),
        ]
        subprocess.run(cmd, check=True, env=env)
        return read_json(output)

    def run_rust_bench(
        self,
        *,
        cli: Path,
        image: Path,
        prompt: str,
        max_new_tokens: int,
        rs_device: str,
        rs_dtype: str,
        output: Path,
        repo_root: Path,
        rendered_prompt: str | None = None,
    ) -> dict[str, Any]:
        env = offline_env(repo_root)
        raw_dir = output.parent
        raw_dir.mkdir(parents=True, exist_ok=True)
        bench_raw = raw_dir / "bench_raw.json"
        rust_output = raw_dir / "rust_output.json"
        _ = rendered_prompt

        cmd = [
            str(cli),
            "--model",
            self.model_id,
            "--image",
            str(image),
            "--device",
            rs_device,
            "--dtype",
            rs_dtype,
            "--max-new-tokens",
            str(max_new_tokens),
            "--bench",
            "--bench-output",
            str(bench_raw),
            "--output-json",
            str(rust_output),
            "--quiet",
            "--prompt",
            self.normalize_prompt(prompt),
        ]
        subprocess.run(cmd, check=True, env=env)

        bench_payload = read_json(bench_raw)
        rust_payload = read_json(rust_output)
        stage_totals = StageTotals.from_payload(bench_payload)
        rust = RustDecodeOutput.from_payload(rust_payload, token_field=self.rust_token_field)

        load_time_s = stage_totals.stage_ms(self.bench_stage_load) / 1e3
        prefill_time_s = sum(stage_totals.stage_ms(stage) for stage in self.bench_stage_prefill) / 1e3
        decode_time_s = stage_totals.stage_ms(self.bench_stage_decode) / 1e3
        total_time_s = stage_totals.stage_ms(self.bench_stage_total) / 1e3

        payload = {
            "schema_version": 1,
            "seed": 0,
            "do_sample": False,
            "model_id": self.model_id,
            "image": str(image),
            "prompt": prompt,
            "rendered_prompt": rust.rendered_prompt,
            "device": rs_device,
            "dtype": rs_dtype,
            "max_new_tokens": max_new_tokens,
            "load_time_s": load_time_s,
            "prefill_time_s": prefill_time_s,
            "decode_time_s": decode_time_s,
            "total_time_s": total_time_s,
            "prompt_tokens": rust.prompt_tokens,
            "generated_tokens": rust.generated_len,
            "generated_token_ids": rust.tokens,
            "tok_per_s": {
                "prefill": rust.prompt_tokens / prefill_time_s if prefill_time_s > 0 else 0.0,
                "decode": rust.generated_len / decode_time_s if decode_time_s > 0 else 0.0,
            },
            "raw": {
                "bench_output": str(bench_raw),
                "rust_output": str(rust_output),
                "stage_totals": stage_totals.by_stage,
            },
        }
        write_json(output, payload)
        return payload

    def _resolve_python_device(self, py_device: str, torch: Any) -> Any:
        if py_device == "mps":
            if not torch.backends.mps.is_built():
                raise RuntimeError("torch mps backend is not built")
            if not torch.backends.mps.is_available():
                raise RuntimeError("torch mps backend is not available")
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_python_dtype(self, py_dtype: str, torch: Any) -> Any:
        return torch.float16 if py_dtype == "f16" else torch.float32

    def python_patch_config(self, cfg_raw: dict[str, Any]) -> dict[str, Any]:
        return cfg_raw

    def python_build_model_config(self, cfg_raw: dict[str, Any]) -> Any:
        _ = cfg_raw
        return None

    def python_load_processor(self, model_dir: Path) -> Any:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)

    def python_load_model(self, model_dir: Path, *, dtype: Any, cfg_raw: dict[str, Any]) -> Any:
        from transformers import AutoModelForImageTextToText

        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": dtype,
        }
        model_config = self.python_build_model_config(cfg_raw)
        if model_config is not None:
            kwargs["config"] = model_config
        return AutoModelForImageTextToText.from_pretrained(model_dir, **kwargs)

    def python_build_messages(self, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "inline://image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def python_prepare_inputs(self, processor: Any, *, image: Path, prompt: str, device: Any) -> tuple[str, dict[str, Any]]:
        from PIL import Image

        messages = self.python_build_messages(prompt)
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pil_image = Image.open(image).convert("RGB")
        inputs = processor(images=[pil_image], text=[rendered], return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        return rendered, inputs

    def python_generate(self, model: Any, *, inputs: dict[str, Any], max_new_tokens: int, torch: Any) -> Any:
        with torch.no_grad():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

    def run_python_bench(
        self,
        *,
        model_dir: Path,
        image: Path,
        prompt: str,
        max_new_tokens: int,
        py_device: str,
        py_dtype: str,
        output: Path,
        repo_root: Path,
    ) -> dict[str, Any]:
        import numpy as np
        import torch
        import os

        env = offline_env(repo_root)
        for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "TRANSFORMERS_CACHE"]:
            os.environ[key] = env[key]

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        device = self._resolve_python_device(py_device, torch)
        dtype = self._resolve_python_dtype(py_dtype, torch)

        cfg_raw = read_json(model_dir / "config.json")
        cfg_raw = self.python_patch_config(cfg_raw)

        load_start = time.perf_counter()
        processor = self.python_load_processor(model_dir)
        model = self.python_load_model(model_dir, dtype=dtype, cfg_raw=cfg_raw)
        model = model.to(device)
        model.eval()
        if device.type == "mps":
            torch.mps.synchronize()
        load_time_s = time.perf_counter() - load_start

        rendered, inputs = self.python_prepare_inputs(processor, image=image, prompt=prompt, device=device)
        prompt_len = int(inputs["input_ids"].shape[1])

        call_times: list[tuple[int, float]] = []
        call_starts: list[tuple[float, int]] = []

        def infer_seq_len(call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> int:
            input_ids = call_kwargs.get("input_ids")
            if input_ids is not None:
                return int(input_ids.shape[1])
            inputs_embeds = call_kwargs.get("inputs_embeds")
            if inputs_embeds is not None:
                return int(inputs_embeds.shape[1])
            if call_args:
                first = call_args[0]
                if isinstance(first, torch.Tensor) and first.ndim >= 2:
                    return int(first.shape[1])
            return 0

        def pre_hook(_module: Any, call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> None:
            if device.type == "mps":
                torch.mps.synchronize()
            call_starts.append((time.perf_counter(), infer_seq_len(call_args, call_kwargs)))

        def post_hook(_module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], _output: Any) -> None:
            if device.type == "mps":
                torch.mps.synchronize()
            start, seq_len = call_starts.pop()
            call_times.append((seq_len, time.perf_counter() - start))

        h1 = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        h2 = model.register_forward_hook(post_hook, with_kwargs=True)

        total_start = time.perf_counter()
        try:
            generated = self.python_generate(model, inputs=inputs, max_new_tokens=max_new_tokens, torch=torch)
        finally:
            h1.remove()
            h2.remove()

        if device.type == "mps":
            torch.mps.synchronize()
        total_time_s = time.perf_counter() - total_start

        output_ids = generated[0][prompt_len:]
        token_ids = [int(v) for v in output_ids.tolist()]

        prefill_time_s = 0.0
        decode_time_s = 0.0
        for idx, (seq_len, elapsed) in enumerate(call_times):
            if idx == 0 or seq_len > 1:
                prefill_time_s += elapsed
            else:
                decode_time_s += elapsed

        canonical_prompt = self.normalize_prompt(prompt)

        payload = {
            "schema_version": 1,
            "seed": 0,
            "do_sample": False,
            "model_dir": str(model_dir),
            "image": str(image),
            "prompt": prompt,
            "rendered_prompt": canonical_prompt,
            "rendered_prompt_hf": rendered,
            "device": py_device,
            "dtype": py_dtype,
            "max_new_tokens": max_new_tokens,
            "load_time_s": load_time_s,
            "prefill_time_s": prefill_time_s,
            "decode_time_s": decode_time_s,
            "total_time_s": total_time_s,
            "prompt_tokens": prompt_len,
            "generated_tokens": len(token_ids),
            "generated_token_ids": token_ids,
            "tok_per_s": {
                "prefill": prompt_len / prefill_time_s if prefill_time_s > 0 else 0.0,
                "decode": len(token_ids) / decode_time_s if decode_time_s > 0 else 0.0,
            },
        }
        write_json(output, payload)
        return payload
