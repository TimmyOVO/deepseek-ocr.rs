from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import random
import subprocess
import sys
import time
from typing import Any

from benchsuite.common import earliest_divergence, offline_env, read_json, runtime_paths, write_json
from benchsuite.schemas import BaselineTokens, RustDecodeOutput, StageTotals, TokenDiff


@dataclass(frozen=True)
class AdapterCapabilities:
    python_baseline: bool = True
    strict_compare: bool = True
    rust_infer: bool = True
    python_skip_reason: str | None = None
    strict_skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_baseline": self.python_baseline,
            "strict_compare": self.strict_compare,
            "rust_infer": self.rust_infer,
            "python_skip_reason": self.python_skip_reason,
            "strict_skip_reason": self.strict_skip_reason,
        }


class BaseAdapter(ABC):
    model_id: str
    suite_name: str
    default_model_dir: Path | None = None
    default_matrix_dir: Path
    capabilities: AdapterCapabilities = AdapterCapabilities()
    requires_snapshot: bool = False
    python_interpreter_env_key: str | None = None
    python_interpreter_required: bool = False
    python_runtime_env_name: str | None = None
    python_runtime_extras: tuple[str, ...] = ()
    required_rust_files: tuple[str, ...] = ("config.json", "tokenizer.json", "model.safetensors")

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

    def describe_capabilities(self) -> dict[str, Any]:
        payload = self.capabilities.to_dict()
        payload["strict_compare"] = self.strict_compare_enabled()
        return payload

    def python_baseline_enabled(self) -> bool:
        return bool(self.capabilities.python_baseline)

    def strict_compare_enabled(self) -> bool:
        return bool(self.capabilities.strict_compare and self.python_baseline_enabled())

    def python_skip_reason(self) -> str | None:
        return self.capabilities.python_skip_reason

    def strict_skip_reason(self) -> str | None:
        if not self.python_baseline_enabled():
            return self.python_skip_reason() or "python baseline is disabled"
        if not self.capabilities.strict_compare:
            return self.capabilities.strict_skip_reason or "strict compare is disabled"
        return None

    def python_support_status(self, *, model_dir: Path) -> tuple[bool, str | None]:
        if not self.python_baseline_enabled():
            return False, self.python_skip_reason() or "python baseline is disabled"
        return True, None

    def resolve_rust_model_dir(self, *, root: Path, runtime_root: Path | None = None) -> Path:
        _ = root
        return runtime_paths(runtime_root=runtime_root, create_dirs=True).cli_models_dir / self.model_id

    def rust_support_status(self, *, root: Path) -> tuple[bool, str | None]:
        if not self.capabilities.rust_infer:
            return False, "rust infer is disabled"
        return True, None

    def resolve_model_dir(
        self,
        *,
        root: Path,
        override: Path | None = None,
        runtime_root: Path | None = None,
    ) -> Path:
        if override is not None:
            raw = Path(override)
            return raw if raw.is_absolute() else root / raw

        if self.default_model_dir is not None:
            raw = Path(self.default_model_dir)
            return raw if raw.is_absolute() else root / raw

        raw = runtime_paths(runtime_root=runtime_root, create_dirs=True).cli_models_dir / self.model_id
        return raw if raw.is_absolute() else root / raw

    def build_static_case_matrix(
        self,
        *,
        root: Path | None,
        prompts: dict[str, str],
        images: dict[str, str],
        max_new_tokens: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        base = root or Path(".")
        rows: list[dict[str, Any]] = []
        for prompt_key, prompt in prompts.items():
            for image_key, image_rel in images.items():
                image_path = Path(image_rel)
                abs_image = image_path if image_path.is_absolute() else base / image_path
                for max_new in max_new_tokens:
                    rows.append(
                        {
                            "case": f"{prompt_key}__{image_key}__n{int(max_new)}",
                            "image": str(abs_image),
                            "prompt": prompt,
                            "max_new_tokens": int(max_new),
                        }
                    )
        return rows

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
        runtime_root: Path | None = None,
        rendered_prompt: str | None = None,
    ) -> dict[str, Any]:
        env = offline_env(repo_root, runtime_root=runtime_root)
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
            "--prompt",
            self.normalize_prompt(prompt),
        ]
        print(
            f"[rust-infer] model={self.model_id} device={rs_device} dtype={rs_dtype} max_new={max_new_tokens}",
            flush=True,
        )
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
        runtime_root: Path | None = None,
        rendered_prompt: str | None = None,
    ) -> dict[str, Any]:
        env = offline_env(repo_root, runtime_root=runtime_root)
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
            "--prompt",
            self.normalize_prompt(prompt),
        ]
        print(
            f"[rust-bench] model={self.model_id} device={rs_device} dtype={rs_dtype} max_new={max_new_tokens}",
            flush=True,
        )
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

        return AutoProcessor.from_pretrained(model_dir, use_fast=False)

    def python_load_model(self, model_dir: Path, *, dtype: Any, cfg_raw: dict[str, Any]) -> Any:
        from transformers import AutoModelForImageTextToText

        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
        }
        model_config = self.python_build_model_config(cfg_raw)
        if model_config is not None:
            kwargs["config"] = model_config
        return AutoModelForImageTextToText.from_pretrained(model_dir, **kwargs)

    def python_baseline_status(self, *, model_dir: Path) -> tuple[bool, str | None]:
        try:
            ok, reason = self.python_support_status(model_dir=model_dir)
            if not ok:
                return ok, reason
            return True, None
        except Exception as exc:
            return False, str(exc)

    def resolve_python_executable(self, *, runtime_root: Path | None = None) -> tuple[str | None, str | None]:
        env_key = self.python_interpreter_env_key
        if env_key:
            raw = os.environ.get(env_key)
            if raw:
                return raw, None
            if self.python_interpreter_required:
                return None, f"missing python interpreter env: {env_key}"

        global_python = os.environ.get("BENCHSUITE_PYTHON")
        if global_python:
            return global_python, None

        if self.python_runtime_env_name:
            runtime = runtime_paths(runtime_root=runtime_root, create_dirs=True)
            env_dir = runtime.root / "python-envs" / self.python_runtime_env_name
            python_exec = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            return str(python_exec), None

        return sys.executable, None

    def _python_interpreter_overridden(self) -> bool:
        if self.python_interpreter_env_key and os.environ.get(self.python_interpreter_env_key):
            return True
        if os.environ.get("BENCHSUITE_PYTHON"):
            return True
        return False

    def _runtime_env_fingerprint(self, *, repo_root: Path) -> str:
        if not self.python_runtime_extras:
            return ""
        payload = "|".join(self.python_runtime_extras)
        pyproject = repo_root / "pyproject.toml"
        if pyproject.exists():
            payload += "|" + pyproject.read_text(encoding="utf-8")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _bootstrap_runtime_python_env(
        self,
        *,
        target_exec: Path,
        repo_root: Path,
    ) -> None:
        if not self.python_runtime_extras:
            raise RuntimeError(f"python interpreter not found: {target_exec}")

        parent_name = target_exec.parent.name
        if parent_name in {"bin", "Scripts"}:
            env_dir = target_exec.parent.parent
        else:
            env_dir = target_exec.parent

        print(
            f"[python-env] bootstrap model={self.model_id} env={env_dir} extras={','.join(self.python_runtime_extras)}",
            flush=True,
        )
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)

        if not target_exec.exists():
            raise RuntimeError(f"failed to create python env executable: {target_exec}")

        subprocess.run(
            [str(target_exec), "-m", "pip", "install", "-e", f".[{','.join(self.python_runtime_extras)}]"],
            check=True,
            cwd=str(repo_root),
        )

    def _ensure_runtime_python_env(
        self,
        *,
        target_exec: Path,
        repo_root: Path,
    ) -> None:
        if not self.python_runtime_extras:
            return

        if not target_exec.exists():
            self._bootstrap_runtime_python_env(target_exec=target_exec, repo_root=repo_root)

        parent_name = target_exec.parent.name
        if parent_name in {"bin", "Scripts"}:
            env_dir = target_exec.parent.parent
        else:
            env_dir = target_exec.parent

        stamp_dir = env_dir / ".benchsuite"
        stamp_dir.mkdir(parents=True, exist_ok=True)
        stamp_file = stamp_dir / f"{self.model_id}.fingerprint"

        current_fp = self._runtime_env_fingerprint(repo_root=repo_root)
        existing_fp = stamp_file.read_text(encoding="utf-8").strip() if stamp_file.exists() else ""

        if existing_fp == current_fp:
            return

        print(
            f"[python-env] sync model={self.model_id} env={env_dir} extras={','.join(self.python_runtime_extras)}",
            flush=True,
        )
        subprocess.run(
            [str(target_exec), "-m", "pip", "install", "-e", f".[{','.join(self.python_runtime_extras)}]"],
            check=True,
            cwd=str(repo_root),
        )
        stamp_file.write_text(current_fp + "\n", encoding="utf-8")

    def _maybe_delegate_python_bench(
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
        runtime_root: Path | None,
    ) -> dict[str, Any] | None:
        if os.environ.get("BENCHSUITE_INTERNAL_PY_BENCH") == "1":
            return None

        python_exec, python_reason = self.resolve_python_executable(runtime_root=runtime_root)
        if python_exec is None:
            raise RuntimeError(python_reason or f"python interpreter unavailable for {self.model_id}")

        try:
            current_exec = Path(sys.executable).expanduser().absolute()
            target_exec = Path(python_exec).expanduser().absolute()
        except Exception:
            current_exec = Path(sys.executable)
            target_exec = Path(python_exec)

        if self.python_runtime_env_name and self.python_runtime_extras and not self._python_interpreter_overridden():
            self._ensure_runtime_python_env(target_exec=target_exec, repo_root=repo_root)
        elif not target_exec.exists():
            raise RuntimeError(f"python interpreter not found: {target_exec}")

        if target_exec == current_exec:
            return None

        cmd = [
            str(target_exec),
            "-m",
            "benchsuite.cli",
            "bench-python",
            "--model",
            self.model_id,
            "--model-dir",
            str(model_dir),
            "--image",
            str(image),
            "--prompt",
            prompt,
            "--device",
            py_device,
            "--dtype",
            py_dtype,
            "--max-new-tokens",
            str(max_new_tokens),
            "--output",
            str(output),
        ]

        env = os.environ.copy()
        env["BENCHSUITE_INTERNAL_PY_BENCH"] = "1"
        if runtime_root is not None:
            env["BENCHSUITE_RUNTIME_ROOT"] = str(runtime_root)

        py_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(repo_root) if not py_path else f"{repo_root}{os.pathsep}{py_path}"

        print(
            f"[python-bench] model={self.model_id} py={target_exec} device={py_device} dtype={py_dtype} max_new={max_new_tokens}",
            flush=True,
        )
        subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))
        return read_json(output)

    def python_pair_status(
        self,
        *,
        model_dir: Path,
        py_device: str,
        py_dtype: str,
        runtime_root: Path | None = None,
    ) -> tuple[bool, str | None]:
        _ = py_device
        _ = py_dtype

        py_ok, py_reason = self.python_support_status(model_dir=model_dir)
        if not py_ok:
            return py_ok, py_reason

        python_exec, python_reason = self.resolve_python_executable(runtime_root=runtime_root)
        if python_exec is None:
            return False, python_reason or "python interpreter unavailable"

        python_path = Path(python_exec).expanduser()
        if not python_path.exists():
            if self._python_interpreter_overridden():
                return False, f"python interpreter not found: {python_exec}"
            if self.python_runtime_env_name and self.python_runtime_extras:
                return True, None
            return False, f"python interpreter not found: {python_exec}"

        return True, None

    def strict_status(self, *, model_dir: Path) -> tuple[bool, str | None]:
        py_ok, py_reason = self.python_baseline_status(model_dir=model_dir)
        if not py_ok:
            return False, py_reason or self.strict_skip_reason() or "python baseline unavailable"
        if not self.strict_compare_enabled():
            return False, self.strict_skip_reason() or "strict compare is disabled"
        return True, None

    def strict_pair_status(
        self,
        *,
        model_dir: Path,
        py_device: str,
        py_dtype: str,
        runtime_root: Path | None = None,
    ) -> tuple[bool, str | None]:
        py_ok, py_reason = self.python_pair_status(
            model_dir=model_dir,
            py_device=py_device,
            py_dtype=py_dtype,
            runtime_root=runtime_root,
        )
        if not py_ok:
            return False, py_reason or self.strict_skip_reason() or "python baseline unavailable"
        if not self.strict_compare_enabled():
            return False, self.strict_skip_reason() or "strict compare is disabled"
        return True, None

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
        runtime_root: Path | None = None,
    ) -> dict[str, Any]:
        if not self.python_baseline_enabled():
            reason = self.python_skip_reason() or f"python baseline is disabled for model {self.model_id}"
            raise RuntimeError(reason)

        delegated = self._maybe_delegate_python_bench(
            model_dir=model_dir,
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            py_device=py_device,
            py_dtype=py_dtype,
            output=output,
            repo_root=repo_root,
            runtime_root=runtime_root,
        )
        if delegated is not None:
            return delegated

        import numpy as np
        import torch
        import os

        env = offline_env(repo_root, runtime_root=runtime_root)
        for key in [
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "DEEPSEEK_OCR_CONFIG_DIR",
            "DEEPSEEK_OCR_CACHE_DIR",
        ]:
            if key in env:
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
