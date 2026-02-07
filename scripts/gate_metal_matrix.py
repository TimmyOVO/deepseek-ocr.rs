#!/usr/bin/env python3
"""Offline, reproducible gate for DeepSeek-OCR long baseline matrix.

Hard requirements enforced:
- One command runs the full 16/16 matrix (ocr1+ocr2 x 4 images x 2 prompts).
- Progress visualization via tqdm (fallback to plain counters if unavailable).
- Strict token-id equality vs baselines/long/<case>/output_tokens.json.
- Per-case structured JSON result + one summary JSON.
- Forced offline + repo-local caches/config (no global cache pollution).
- CLI parameters are derived from baseline.json/prompt.json assets.

This script is the canonical gate carrier for this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_ROOT = REPO_ROOT / "baselines" / "long"


@dataclass
class CaseSpec:
    name: str
    variant: str  # ocr1|ocr2
    baseline_dir: Path
    baseline_json: Path
    prompt_json: Path
    output_tokens_json: Path
    image_paths: List[Path]
    base_size: int
    image_size: int
    crop_mode: bool
    seed: Optional[int]


def sha256_int_list(xs: List[int]) -> str:
    h = hashlib.sha256()
    for x in xs:
        h.update(str(int(x)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_baseline_path(path_str: str, baseline_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    candidate = baseline_dir / p
    if candidate.exists():
        return candidate
    return REPO_ROOT / p


def expected_generated_tokens(output_tokens: Dict[str, Any]) -> List[int]:
    tokens = [int(x) for x in output_tokens["tokens"]]
    prefill_len = int(output_tokens["prefill_len"])
    gen = tokens[prefill_len:]
    eos = output_tokens.get("eos_token_id")
    if eos is not None and gen and gen[-1] == int(eos):
        gen = gen[:-1]
    return gen


def first_mismatch(a: List[int], b: List[int]) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def window(xs: List[int], idx: int, radius: int = 8) -> List[int]:
    start = max(0, idx - radius)
    end = min(len(xs), idx + radius + 1)
    return xs[start:end]


def parse_no_repeat_ngram(name: str, baseline_meta: Dict[str, Any]) -> Optional[int]:
    if "no_repeat_ngram_size" in baseline_meta:
        try:
            return int(baseline_meta["no_repeat_ngram_size"])
        except Exception:
            return None
    m = re.search(r"__ng(\d+)$", name)
    if m:
        return int(m.group(1))
    return None


def discover_cases(only_variant: Optional[str]) -> List[CaseSpec]:
    if not BASELINES_ROOT.exists():
        raise SystemExit(f"missing baselines root: {BASELINES_ROOT}")

    # Canonical 16-case definition: all baselines ending with __8192_ng20.
    case_dirs = sorted(
        [p for p in BASELINES_ROOT.iterdir() if p.is_dir() and p.name.endswith("__8192_ng20")],
        key=lambda p: p.name,
    )
    if only_variant is None and len(case_dirs) != 16:
        raise SystemExit(
            f"expected 16 baseline case dirs under {BASELINES_ROOT}, found {len(case_dirs)}"
        )

    specs: List[CaseSpec] = []
    for d in case_dirs:
        baseline_json = d / "baseline.json"
        meta = read_json(baseline_json)
        variant = meta.get("variant")
        if variant not in ("ocr1", "ocr2"):
            raise SystemExit(f"unexpected variant={variant!r} in {d}")
        if only_variant is not None and variant != only_variant:
            continue

        prompt_assets = meta.get("prompt_assets_path") or str(d / "prompt.json")
        prompt_json = resolve_baseline_path(prompt_assets, d)
        output_assets = meta.get("output_tokens_path") or str(d / "output_tokens.json")
        output_tokens_json = resolve_baseline_path(output_assets, d)

        prompt_data = read_json(prompt_json)
        image_paths_raw = prompt_data.get("image_paths")
        if image_paths_raw:
            image_paths = [resolve_baseline_path(p, d) for p in image_paths_raw]
        else:
            image_field = meta.get("image")
            image_paths = [resolve_baseline_path(image_field, d)] if image_field else []

        base_size = int(meta.get("base_size", 0))
        image_size = int(meta.get("image_size", 0))
        crop_mode = bool(meta.get("crop_mode", False))
        seed = meta.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except Exception:
                seed = None

        specs.append(
            CaseSpec(
                name=d.name,
                variant=variant,
                baseline_dir=d,
                baseline_json=baseline_json,
                prompt_json=prompt_json,
                output_tokens_json=output_tokens_json,
                image_paths=image_paths,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                seed=seed,
            )
        )

    return specs


def ensure_offline_env(env: Dict[str, str], *, strict: bool = True) -> Dict[str, str]:
    env = dict(env)

    # Hard offline requirements.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    # Repo-local caches/config.
    env["HF_HOME"] = str(REPO_ROOT / ".hf-cache")
    env["DEEPSEEK_OCR_CACHE_DIR"] = str(REPO_ROOT / ".cli-cache")
    env["DEEPSEEK_OCR_CONFIG_DIR"] = str(REPO_ROOT / ".cli-config")

    if strict:
        # Try to avoid accidental fallbacks to user-global caches.
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(REPO_ROOT / ".hf-cache" / "hub"))
        env.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".hf-cache" / "transformers"))

    return env


def require_local_model_files(case: CaseSpec) -> None:
    """Hard offline guard: ensure we will not trigger any network downloads.

    The CLI asset helper will try to download missing weights/tokenizer/config.
    For this gate, we require that the repo-local model directories already exist.
    """

    if case.variant == "ocr1":
        model_dir = REPO_ROOT / "DeepSeek-OCR"
        expected_weights = model_dir / "model-00001-of-000001.safetensors"
        expected_tokenizer = model_dir / "tokenizer.json"
        expected_config = model_dir / "config.json"
    else:
        model_dir = REPO_ROOT / "DeepSeek-OCR-2"
        expected_weights = model_dir / "model-00001-of-000001.safetensors"
        expected_tokenizer = model_dir / "tokenizer.json"
        expected_config = model_dir / "config.json"

    missing: List[str] = []
    for p in [expected_weights, expected_tokenizer, expected_config]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise SystemExit(
            "offline gate requires repo-local model files to exist; missing:\n"
            + "\n".join(f"- {m}" for m in missing)
        )


def ensure_repo_local_dirs() -> None:
    (REPO_ROOT / ".hf-cache").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / ".cli-cache").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / ".cli-config").mkdir(parents=True, exist_ok=True)

    cfg_path = REPO_ROOT / ".cli-config" / "config.toml"
    if not cfg_path.exists():
        # Keep this minimal and deterministic.
        cfg_path.write_text(
            """[models]
active = "deepseek-ocr"

[models.entries.deepseek-ocr]
kind = "deepseek"
config = "DeepSeek-OCR/config.json"
tokenizer = "DeepSeek-OCR/tokenizer.json"
weights = "DeepSeek-OCR/model-00001-of-000001.safetensors"

[models.entries.deepseek-ocr-2]
kind = "deepseek"
config = "DeepSeek-OCR-2/config.json"
tokenizer = "DeepSeek-OCR-2/tokenizer.json"
weights = "DeepSeek-OCR-2/model-00001-of-000001.safetensors"

[inference]
# Overridden per-run via CLI flags.
device = "cpu"
template = "plain"
base_size = 1024
image_size = 640
crop_mode = true
max_new_tokens = 8192
use_cache = true
repetition_penalty = 1.0
no_repeat_ngram_size = 20
""",
            encoding="utf-8",
        )


def cli_help_has_flags(cli: Path, flags: List[str], env: Dict[str, str]) -> bool:
    p = subprocess.run([str(cli), "--help"], env=env, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return all(f in out for f in flags)


def run_checked(cmd: List[str], env: Dict[str, str], *, timeout: Optional[int] = None) -> None:
    p = subprocess.run(cmd, env=env, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"command failed (exit={p.returncode}): {' '.join(cmd)}")


def write_case_report(out_case_dir: Path, report: Dict[str, Any]) -> None:
    (out_case_dir / "compare.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def run_one_case(
    *,
    cli: Path,
    case: CaseSpec,
    device: str,
    dtype: str,
    out_dir: Path,
    env: Dict[str, str],
    case_timeout: Optional[int],
) -> Dict[str, Any]:
    out_case_dir = out_dir / case.name
    out_case_dir.mkdir(parents=True, exist_ok=True)
    cli_json = out_case_dir / "cli_output.json"

    report: Dict[str, Any] = {
        "schema_version": 1,
        "case": case.name,
        "variant": case.variant,
        "device": device,
        "dtype": dtype,
        "pass": False,
        "baseline_dir": str(case.baseline_dir),
        "baseline_json": str(case.baseline_json),
        "cli_output_json": str(cli_json),
    }

    # Sanity: required inputs must exist.
    missing: List[str] = []
    for p in [case.prompt_json, case.output_tokens_json]:
        if not p.exists():
            missing.append(str(p))
    for p in case.image_paths:
        if not p.exists():
            missing.append(str(p))
    config_toml = REPO_ROOT / "session" / "cli_matrix" / (
        "config_ocr1.toml" if case.variant == "ocr1" else "config_ocr2.toml"
    )
    if not config_toml.exists():
        missing.append(str(config_toml))
    if missing:
        report["error"] = "missing required files"
        report["missing"] = missing
        write_case_report(out_case_dir, report)
        return report

    require_local_model_files(case)

    out_tokens = read_json(case.output_tokens_json)
    max_new = int(out_tokens.get("generated_len", 0))
    if max_new <= 0:
        max_new = int(read_json(case.baseline_json).get("max_new_tokens", 0))
    if max_new <= 0:
        report["error"] = "missing max_new_tokens"
        write_case_report(out_case_dir, report)
        return report

    baseline_meta = read_json(case.baseline_json)
    no_repeat_ngram_size = parse_no_repeat_ngram(case.name, baseline_meta)
    repetition_penalty = float(baseline_meta.get("repetition_penalty", 1.0))

    cmd = [
        str(cli),
        "--config",
        str(config_toml),
        "--model",
        "deepseek-ocr" if case.variant == "ocr1" else "deepseek-ocr-2",
        "--prompt-json",
        str(case.prompt_json),
        "--device",
        device,
        "--dtype",
        dtype,
        "--base-size",
        str(case.base_size),
        "--image-size",
        str(case.image_size),
        "--crop-mode",
        "true" if case.crop_mode else "false",
        "--max-new-tokens",
        str(max_new),
        "--repetition-penalty",
        f"{repetition_penalty}",
        "--output-json",
        str(cli_json),
        "--quiet",
    ]
    if case.seed is not None:
        cmd.extend(["--seed", str(case.seed)])
    if no_repeat_ngram_size is not None:
        cmd.extend(["--no-repeat-ngram-size", str(no_repeat_ngram_size)])
    for img in case.image_paths:
        cmd.extend(["--image", str(img)])

    started = time.time()
    try:
        run_checked(cmd, env, timeout=case_timeout)
    except Exception as e:
        report["error"] = f"cli_failed: {e}"
        report["cli_cmd"] = cmd
        write_case_report(out_case_dir, report)
        return report

    # Compare tokens.
    cli_out = read_json(cli_json)
    got = [int(x) for x in cli_out.get("tokens", [])]
    expected = expected_generated_tokens(out_tokens)
    mismatch = first_mismatch(got, expected)
    ok = mismatch is None

    report.update(
        {
            "elapsed_sec": round(time.time() - started, 3),
            "pass": ok,
            "expected_len": len(expected),
            "got_len": len(got),
            "expected_sha256": sha256_int_list(expected),
            "got_sha256": sha256_int_list(got),
            "earliest_divergence": None,
            "context_expected": None,
            "context_got": None,
            "max_new_tokens": max_new,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
        }
    )

    if not ok:
        assert mismatch is not None
        report["earliest_divergence"] = {
            "index": mismatch,
            "expected": expected[mismatch] if mismatch < len(expected) else None,
            "got": got[mismatch] if mismatch < len(got) else None,
        }
        report["context_expected"] = window(expected, mismatch)
        report["context_got"] = window(got, mismatch)

        # If the CLI supports the env-driven logits debug, capture it deterministically.
        if cli_help_has_flags(cli, ["--prompt-json", "--output-json"], env):
            env2 = dict(env)
            env2["DEEPSEEK_OCR_DEBUG_LOGITS_STEP"] = str(mismatch)
            env2["DEEPSEEK_OCR_DEBUG_LOGITS_JSON"] = str(out_case_dir / "debug_logits.json")
            try:
                run_checked(cmd, env2, timeout=case_timeout)
                dbg_path = out_case_dir / "debug_logits.json"
                if dbg_path.exists():
                    report["debug_logits"] = read_json(dbg_path)
            except Exception:
                # Keep compare result stable even if debug capture fails.
                report["debug_logits"] = {"error": "debug logits capture failed"}

    write_case_report(out_case_dir, report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["cpu", "metal"])
    ap.add_argument("--dtype", required=True, choices=["f32", "f16"])
    ap.add_argument(
        "--cli",
        type=Path,
        default=REPO_ROOT / "target" / "release" / "deepseek-ocr-cli",
        help="Path to deepseek-ocr-cli binary (must support --prompt-json/--output-json)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root. Default: session/gates/<device>_<dtype>/<timestamp>",
    )
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--only-variant", choices=["ocr1", "ocr2"], default=None)
    ap.add_argument("--case-timeout", type=int, default=1800, help="Per-case timeout in seconds (default: 1800)")
    args = ap.parse_args()

    ensure_repo_local_dirs()

    env = ensure_offline_env(os.environ.copy())

    cli: Path = args.cli
    if not cli.exists():
        raise SystemExit(
            f"missing CLI binary at {cli}. Build e.g.: cargo build -p deepseek-ocr-cli --release --features metal,cli-debug"
        )

    # Require prompt-json/output-json support so our parameters are baseline-driven and auditable.
    if not cli_help_has_flags(cli, ["--prompt-json", "--output-json"], env):
        raise SystemExit(
            "CLI missing --prompt-json/--output-json. Rebuild with: cargo build -p deepseek-ocr-cli --release --features metal,cli-debug"
        )

    cases = discover_cases(args.only_variant)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.out_dir or (
        REPO_ROOT / "session" / "gates" / f"{args.device}_{args.dtype}" / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    iterator = cases
    if tqdm is not None:
        iterator = tqdm(cases, desc=f"gate {args.device} {args.dtype}", unit="case")

    results: List[Dict[str, Any]] = []
    passed = 0
    failed = 0
    total = len(cases)
    for idx, case in enumerate(iterator, 1):
        rep = run_one_case(
            cli=cli,
            case=case,
            device=args.device,
            dtype=args.dtype,
            out_dir=out_dir,
            env=env,
            case_timeout=args.case_timeout,
        )
        results.append(rep)
        if rep.get("pass"):
            passed += 1
        else:
            failed += 1
        if tqdm is None:
            status = "PASS" if rep.get("pass") else "FAIL"
            print(
                f"[{idx}/{total}] {case.name}: {status} (pass={passed} fail={failed})",
                flush=True,
            )
        if not rep.get("pass") and args.fail_fast:
            break

    total = len(results)

    summary = {
        "schema_version": 1,
        "device": args.device,
        "dtype": args.dtype,
        "repo_root": str(REPO_ROOT),
        "out_dir": str(out_dir),
        "total": total,
        "passed": passed,
        "failed": failed,
        "results": [
            {
                "case": r.get("case"),
                "variant": r.get("variant"),
                "pass": r.get("pass"),
                "earliest_divergence": r.get("earliest_divergence"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    # Console summary (stable, greppable).
    print(f"\nRESULT: {passed}/{total} PASS ({failed} FAIL)")
    if failed:
        for r in results:
            if not r.get("pass"):
                print(f"- {r.get('case')}: {r.get('earliest_divergence') or r.get('error')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
