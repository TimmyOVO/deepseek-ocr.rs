from __future__ import annotations

import contextlib
from importlib import metadata as importlib_metadata
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from benchsuite.models.base import AdapterCapabilities, BaseAdapter


class _DeepseekFamilyAdapter(BaseAdapter):
    python_interpreter_env_key = "BENCHSUITE_PY_DEEPSEEK"
    python_runtime_env_name = "deepseek"
    python_runtime_extras = ("bench", "bench-deepseek")
    suite_name = "deepseek"
    default_matrix_dir = Path("baselines/long")
    capabilities = AdapterCapabilities(
        python_baseline=True,
        strict_compare=True,
        rust_infer=True,
    )

    default_images = {
        "image": "baselines/sample/images/image.png",
        "test": "baselines/sample/images/test.png",
        "test2": "baselines/sample/images/test2.png",
        "test3": "baselines/sample/images/test3.png",
    }
    default_max_new_tokens = (64, 256)

    hf_repo_id: str
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True

    def python_pair_status(
        self,
        *,
        model_dir: Path,
        py_device: str,
        py_dtype: str,
        runtime_root: Path | None = None,
    ) -> tuple[bool, str | None]:
        _ = py_dtype

        if py_device != "cpu":
            return False, "deepseek python baseline currently supports cpu only"

        return super().python_pair_status(
            model_dir=model_dir,
            py_device=py_device,
            py_dtype=py_dtype,
            runtime_root=runtime_root,
        )

    def python_baseline_status(self, *, model_dir: Path) -> tuple[bool, str | None]:
        _ = model_dir
        ok, reason = self.python_support_status(model_dir=model_dir)
        if not ok:
            return ok, reason

        if os.environ.get("BENCHSUITE_INTERNAL_PY_BENCH") != "1":
            return True, None

        try:
            version = importlib_metadata.version("transformers")
        except Exception as exc:
            return False, f"transformers missing: {exc}"

        if version != "4.46.3":
            return (
                False,
                f"deepseek python baseline requires transformers==4.46.3, got {version}; use BENCHSUITE_PY_DEEPSEEK",
            )

        return True, None

    def normalize_prompt(self, prompt: str) -> str:
        if "<image>" in prompt:
            return prompt
        canonical = prompt if prompt.startswith("\n") else f"\n{prompt}"
        return f"<image>{canonical}"

    def python_load_processor(self, model_dir: Path) -> Any:
        from transformers import AutoTokenizer
        import os

        _ = model_dir
        cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
        return AutoTokenizer.from_pretrained(
            self.hf_repo_id,
            trust_remote_code=True,
            cache_dir=cache_root,
        )

    def python_load_model(self, model_dir: Path, *, dtype: Any, cfg_raw: dict[str, Any]) -> Any:
        import os

        _ = model_dir
        _ = dtype
        _ = cfg_raw
        cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
        try:
            from transformers import AutoModel
        except Exception as exc:
            raise RuntimeError(f"failed to import transformers AutoModel: {exc}") from exc

        try:
            return AutoModel.from_pretrained(
                self.hf_repo_id,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=cache_root,
            )
        except Exception as exc:
            message = str(exc)
            if "LlamaFlashAttention2" in message:
                raise RuntimeError(
                    "deepseek python baseline needs transformers==4.46.3 in BENCHSUITE_PY_DEEPSEEK"
                ) from exc
            raise

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
        ok, reason = self.python_pair_status(
            model_dir=model_dir,
            py_device=py_device,
            py_dtype=py_dtype,
            runtime_root=runtime_root,
        )
        if not ok:
            raise RuntimeError(reason or "deepseek python baseline unavailable")

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
        import os
        import torch

        from benchsuite.common import offline_env, write_json

        env = offline_env(repo_root, runtime_root=runtime_root)
        for key in ["HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]:
            if key in env:
                os.environ[key] = env[key]

        random_seed = 0
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        load_start = time.perf_counter()
        tokenizer = self.python_load_processor(model_dir)
        model = self.python_load_model(model_dir, dtype=torch.float32, cfg_raw={})
        model = model.to(torch.float32)
        try:
            vision_model = getattr(model, "vision_model", None)
            if vision_model is not None:
                vision_model.to(torch.bfloat16)
        except Exception:
            pass
        model = model.eval().to(torch.device("cpu"))
        load_time_s = time.perf_counter() - load_start

        normalized_prompt = self.normalize_prompt(prompt)

        output_dir = Path(tempfile.mkdtemp(prefix="benchsuite_deepseek_py_"))

        capture: dict[str, Any] = {}
        origin_generate = model.generate

        def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
            kwargs["max_new_tokens"] = int(max_new_tokens)
            kwargs["do_sample"] = False
            kwargs["temperature"] = 0.0
            kwargs["use_cache"] = True
            kwargs["no_repeat_ngram_size"] = 20
            if args and hasattr(args[0], "shape"):
                capture["input_len"] = int(args[0].shape[1])
            output_ids = origin_generate(*args, **kwargs)
            capture["output_ids"] = output_ids.detach().cpu() if hasattr(output_ids, "detach") else output_ids
            return output_ids

        model.generate = wrapped_generate  # type: ignore[assignment]

        original_tensor_cuda = torch.Tensor.cuda
        original_module_cuda = torch.nn.Module.cuda
        original_autocast = torch.autocast
        original_bfloat16 = torch.bfloat16

        def _tensor_cuda_noop(self: Any, *args: Any, **kwargs: Any) -> Any:
            _ = args
            _ = kwargs
            return self

        def _module_cuda_noop(self: Any, *args: Any, **kwargs: Any) -> Any:
            _ = args
            _ = kwargs
            return self

        @contextlib.contextmanager
        def _autocast_noop(*args: Any, **kwargs: Any):
            _ = args
            _ = kwargs
            yield

        torch.Tensor.cuda = _tensor_cuda_noop  # type: ignore[assignment]
        torch.nn.Module.cuda = _module_cuda_noop  # type: ignore[assignment]
        torch.autocast = _autocast_noop  # type: ignore[assignment]
        torch.bfloat16 = torch.float32  # type: ignore[assignment]

        infer_start = time.perf_counter()
        try:
            generated_text = model.infer(
                tokenizer,
                prompt=normalized_prompt,
                image_file=str(image),
                output_path=str(output_dir),
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                save_results=False,
                eval_mode=True,
            )
            total_time_s = time.perf_counter() - infer_start
        finally:
            torch.Tensor.cuda = original_tensor_cuda  # type: ignore[assignment]
            torch.nn.Module.cuda = original_module_cuda  # type: ignore[assignment]
            torch.autocast = original_autocast  # type: ignore[assignment]
            torch.bfloat16 = original_bfloat16  # type: ignore[assignment]
            model.generate = origin_generate  # type: ignore[assignment]

        output_ids = capture.get("output_ids")
        input_len = capture.get("input_len")
        if (
            output_ids is not None
            and isinstance(input_len, int)
            and hasattr(output_ids, "shape")
            and len(output_ids.shape) >= 2
        ):
            token_ids = [int(v) for v in output_ids[0, input_len:].tolist()]
        else:
            encoded = tokenizer.encode(str(generated_text), add_special_tokens=False)
            token_ids = [int(v) for v in encoded]

        payload = {
            "schema_version": 1,
            "seed": random_seed,
            "do_sample": False,
            "model_dir": str(model_dir),
            "image": str(image),
            "prompt": prompt,
            "rendered_prompt": normalized_prompt,
            "device": py_device,
            "dtype": py_dtype,
            "max_new_tokens": int(max_new_tokens),
            "load_time_s": load_time_s,
            "prefill_time_s": 0.0,
            "decode_time_s": total_time_s,
            "total_time_s": total_time_s,
            "prompt_tokens": int(input_len) if isinstance(input_len, int) else 0,
            "generated_tokens": len(token_ids),
            "generated_token_ids": token_ids,
            "generated_text": str(generated_text),
            "tok_per_s": {
                "prefill": 0.0,
                "decode": (len(token_ids) / total_time_s) if total_time_s > 0 else 0.0,
            },
        }
        write_json(output, payload)
        return payload

    def default_case_matrix(self, *, root: Path | None = None) -> list[dict[str, Any]]:
        return self.build_static_case_matrix(
            root=root,
            prompts=self.default_prompts,
            images=self.default_images,
            max_new_tokens=self.default_max_new_tokens,
        )


class DeepseekOcrAdapter(_DeepseekFamilyAdapter):
    model_id = "deepseek-ocr"
    hf_repo_id = "deepseek-ai/DeepSeek-OCR"
    default_model_dir = None
    image_size = 640
    default_prompts = {
        "grounding_md": "<|grounding|>Convert the document to markdown.",
        "describe": "Describe this image in detail.",
    }


class DeepseekOcr2Adapter(_DeepseekFamilyAdapter):
    model_id = "deepseek-ocr-2"
    hf_repo_id = "deepseek-ai/DeepSeek-OCR-2"
    default_model_dir = None
    image_size = 768
    default_prompts = {
        "describe": "Describe this image in detail.",
        "grounding_md": "<|grounding|>Convert the document to markdown.",
    }
