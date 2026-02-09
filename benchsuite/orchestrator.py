from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from benchsuite.common import earliest_divergence, has_mps, read_json, repo_root, runtime_paths, write_json
from benchsuite.registry import get_adapter, list_default_models

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class BenchOrchestrator:
    @staticmethod
    def _resolve_run_root(args: Any, run: str) -> Path:
        if getattr(args, "output_root", None) is not None:
            return Path(args.output_root)
        return Path("baselines") / "benchsuite" / "runs" / run

    @staticmethod
    def _format_table(header: list[str], rows: list[list[str]]) -> str:
        table = [header] + rows
        widths = [max(len(row[idx]) for row in table) for idx in range(len(header))]

        def fmt(row: list[str]) -> str:
            return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))

        lines = [fmt(header), "-+-".join("-" * w for w in widths)]
        lines.extend(fmt(row) for row in rows)
        return "\n".join(lines)

    @staticmethod
    def _pairs_rows(pairs: list[dict[str, Any]]) -> list[list[str]]:
        rows: list[list[str]] = []
        for pair in pairs:
            py = pair.get("python_metrics")
            rs = pair.get("rust_metrics")
            strict_status = str(pair.get("strict_status") or "")

            rs_prefill_s = "-"
            rs_total_s = "-"
            rs_decode_s = "-"
            if isinstance(rs, dict):
                try:
                    rs_prefill_s = f"{float(rs.get('prefill_time_s', 0.0)):.3f}"
                except Exception:
                    rs_prefill_s = "-"
                try:
                    rs_total_s = f"{float(rs.get('total_time_s', 0.0)):.3f}"
                except Exception:
                    rs_total_s = "-"
                try:
                    rs_decode_s = f"{float(rs.get('tok_per_s', {}).get('decode', 0.0)):.3f}"
                except Exception:
                    rs_decode_s = "-"

            if strict_status == "skipped":
                reason = str(pair.get("skip_reason") or "skip")
                rows.append(
                    [
                        pair["pair"],
                        "skip",
                        "-",
                        rs_prefill_s,
                        "-",
                        "-",
                        rs_total_s,
                        "-",
                        "-",
                        rs_decode_s,
                        reason,
                    ]
                )
                continue

            if strict_status == "error":
                reason = str(pair.get("error") or pair.get("skip_reason") or "error")
                rows.append(
                    [
                        pair["pair"],
                        "error",
                        "-",
                        rs_prefill_s,
                        "-",
                        "-",
                        rs_total_s,
                        "-",
                        "-",
                        rs_decode_s,
                        reason,
                    ]
                )
                continue

            if strict_status == "compared" and pair.get("all_match") is False:
                detail = str(pair.get("error") or "strict mismatch")
                if pair.get("token_diff"):
                    detail = f"token_diff={pair['token_diff']}"
                elif pair.get("prompt_diff"):
                    detail = f"prompt_diff={pair['prompt_diff']}"
                rows.append(
                    [
                        pair["pair"],
                        "fail",
                        "-",
                        rs_prefill_s,
                        "-",
                        "-",
                        rs_total_s,
                        "-",
                        "-",
                        rs_decode_s,
                        detail,
                    ]
                )
                continue

            if not isinstance(py, dict) or not isinstance(rs, dict):
                rows.append(
                    [
                        pair["pair"],
                        strict_status or "skip",
                        "-",
                        rs_prefill_s,
                        "-",
                        "-",
                        rs_total_s,
                        "-",
                        "-",
                        rs_decode_s,
                        str(pair.get("skip_reason") or "-"),
                    ]
                )
                continue

            py_total = float(py["total_time_s"])
            rs_total = float(rs["total_time_s"])
            py_prefill = float(py["prefill_time_s"])
            rs_prefill = float(rs["prefill_time_s"])
            py_decode = float(py["tok_per_s"]["decode"])
            rs_decode = float(rs["tok_per_s"]["decode"])
            rows.append(
                [
                    pair["pair"],
                    "ok" if pair.get("all_match") is True else "fail",
                    f"{py_prefill:.3f}",
                    f"{rs_prefill:.3f}",
                    f"{(py_prefill / rs_prefill) if rs_prefill > 0 else 0.0:.3f}x",
                    f"{py_total:.3f}",
                    f"{rs_total:.3f}",
                    f"{(py_total / rs_total) if rs_total > 0 else 0.0:.3f}x",
                    f"{py_decode:.3f}",
                    f"{rs_decode:.3f}",
                    f"{(rs_decode / py_decode) if py_decode > 0 else 0.0:.3f}x",
                ]
            )
        return rows

    def _print_perf_report(self, pairs: list[dict[str, Any]]) -> None:
        print("\n=== Python vs Rust (same model/device/dtype) ===")
        table = self._format_table(
            [
                "pair",
                "strict",
                "py_prefill",
                "rs_prefill",
                "py/rs_pf",
                "py_total",
                "rs_total",
                "py/rs_total",
                "py_decode_tps",
                "rs_decode_tps",
                "rs/py_tps_or_reason",
            ],
            self._pairs_rows(pairs),
        )
        print(table)

    @staticmethod
    def _default_models(args: Any) -> list[str]:
        if args.include_models:
            return [model.strip() for model in args.include_models if model.strip()]
        return list_default_models()

    @staticmethod
    def _default_devices(args: Any) -> list[str]:
        if args.include_devices:
            return [device.strip().lower() for device in args.include_devices if device.strip()]
        return ["cpu", "mps"]

    @staticmethod
    def _default_precisions(args: Any) -> list[str]:
        if args.include_precision:
            return [dtype.strip().lower() for dtype in args.include_precision if dtype.strip()]
        return ["f32", "f16"]

    @staticmethod
    def _build_pairs_for_adapter(
        *,
        adapter: Any,
        devices: list[str],
        precisions: list[str],
    ) -> list[dict[str, str]]:
        return adapter.build_device_precision_matrix(
            devices=devices,
            precisions=precisions,
            mps_available=has_mps(),
        )

    @staticmethod
    def _adapter_case_matrix(
        *,
        adapter: Any,
        root: Path,
        limit: int | None,
        cases: list[str] | None,
    ) -> list[dict[str, Any]]:
        rows = adapter.default_case_matrix(root=root)
        if cases:
            selected = set(cases)
            rows = [row for row in rows if str(row.get("case")) in selected]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def _resolve_case_assets(
        self,
        args: Any,
        *,
        adapter: Any,
        root: Path,
        model_root: Path,
    ) -> tuple[Path, str, int, Path | None, dict[str, Any]]:
        baseline_json = args.baseline_json
        data: dict[str, Any] = {}
        fallback_case: dict[str, Any] | None = None

        if baseline_json is not None and not baseline_json.exists():
            raise SystemExit(f"baseline json not found: {baseline_json}")

        if baseline_json is None and args.case_name:
            search_roots: list[Path] = []
            if args.matrix_source is not None:
                search_roots.append(args.matrix_source)
            default_matrix = getattr(adapter, "default_matrix_dir", None)
            if default_matrix is not None:
                search_roots.append(Path(default_matrix))
            if model_root.exists():
                search_roots.extend(sorted((path for path in model_root.glob("matrix_*") if path.is_dir()), reverse=True))

            deduped: list[Path] = []
            seen: set[Path] = set()
            for path in search_roots:
                if path in seen:
                    continue
                seen.add(path)
                deduped.append(path)

            for matrix_root in deduped:
                candidate = matrix_root / args.case_name / "baseline.json"
                if candidate.exists():
                    baseline_json = candidate
                    break

            if baseline_json is None:
                for row in adapter.default_case_matrix(root=root):
                    if str(row.get("case")) == str(args.case_name):
                        fallback_case = row
                        break

            if baseline_json is None and fallback_case is None and not (args.image and args.prompt):
                searched = ", ".join(str(path) for path in deduped) if deduped else "<none>"
                raise SystemExit(
                    "baseline for case not found. "
                    f"case={args.case_name}; searched=[{searched}]. "
                    "use --baseline-json or --matrix-source, or pass --image/--prompt directly"
                )

        if baseline_json is not None:
            data = read_json(baseline_json)
        elif fallback_case is not None:
            data = {
                "image": fallback_case.get("image"),
                "prompt": fallback_case.get("prompt"),
                "max_new_tokens": fallback_case.get("max_new_tokens"),
            }

        image_value = data.get("image") or args.image
        prompt_value = data.get("prompt") or args.prompt
        if image_value is None or prompt_value is None:
            raise SystemExit("missing image/prompt; provide --baseline-json or both --image and --prompt")

        image_raw = Path(str(image_value))
        image = image_raw if image_raw.is_absolute() else root / image_raw
        prompt = str(prompt_value)

        if args.max_new_tokens is not None:
            max_new = int(args.max_new_tokens)
        elif args.case_name and "__n" in args.case_name:
            tail = args.case_name.rsplit("__n", 1)[-1]
            max_new = int(tail) if tail.isdigit() else 64
        else:
            generated_tokens = data.get("generated_tokens")
            if isinstance(generated_tokens, list) and generated_tokens:
                max_new = len(generated_tokens)
            else:
                max_new = int(data.get("max_new_tokens") or 64)

        return image, prompt, max_new, baseline_json, data

    @staticmethod
    def _strict_compare(py_metrics: dict[str, Any], rs_metrics: dict[str, Any]) -> dict[str, Any]:
        py_tokens = py_metrics.get("generated_token_ids", [])
        rs_tokens = rs_metrics.get("generated_token_ids", [])
        if not isinstance(py_tokens, list) or not isinstance(rs_tokens, list):
            return {
                "token_match": False,
                "prompt_match": False,
                "token_diff": {"reason": "missing generated_token_ids"},
                "prompt_diff": {"reason": "missing rendered_prompt"},
            }

        token_diff = earliest_divergence(py_tokens, rs_tokens)
        token_match = token_diff is None

        py_prompt = py_metrics.get("rendered_prompt")
        rs_prompt = rs_metrics.get("rendered_prompt")
        prompt_match = isinstance(py_prompt, str) and isinstance(rs_prompt, str) and py_prompt == rs_prompt

        prompt_diff: dict[str, Any] | None = None
        if not prompt_match:
            prompt_diff = {
                "python_len": len(py_prompt) if isinstance(py_prompt, str) else None,
                "rust_len": len(rs_prompt) if isinstance(rs_prompt, str) else None,
            }

        token_diff_payload: dict[str, Any] | None = None
        if token_diff is not None:
            token_diff_payload = {
                "index": token_diff[0],
                "python": token_diff[1],
                "rust": token_diff[2],
            }

        return {
            "token_match": token_match,
            "prompt_match": prompt_match,
            "token_diff": token_diff_payload,
            "prompt_diff": prompt_diff,
        }

    @staticmethod
    def _find_previous_run(base_root: Path, current_run: str) -> Path | None:
        run_root = base_root / "runs"
        if not run_root.exists():
            return None
        candidates = sorted(path.name for path in run_root.iterdir() if path.is_dir())
        prior = [name for name in candidates if name < current_run]
        if not prior:
            return None
        return run_root / prior[-1]

    @staticmethod
    def _ratio(prev: float, curr: float) -> float:
        return (prev / curr) if curr > 0 else 0.0

    def _build_prev_compare(
        self,
        *,
        current_pairs: list[dict[str, Any]],
        prev_summary: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if prev_summary is None:
            return None

        prev_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in prev_summary.get("pairs", []):
            key = (
                str(row.get("model", "")),
                str(row.get("python_device", "")),
                str(row.get("python_dtype", "")),
                str(row.get("case", "")),
            )
            prev_map[key] = row

        rows: list[dict[str, Any]] = []
        for row in current_pairs:
            if not isinstance(row.get("rust_metrics"), dict):
                continue

            key = (
                row["model"],
                row["python_device"],
                row["python_dtype"],
                row["case"],
            )
            prev = prev_map.get(key)
            if prev is None:
                continue

            curr_rs = row["rust_metrics"]
            prev_rs = prev.get("rust_metrics", {})
            try:
                speedup = {
                    "prefill_time": self._ratio(float(prev_rs["prefill_time_s"]), float(curr_rs["prefill_time_s"])),
                    "decode_time": self._ratio(float(prev_rs["decode_time_s"]), float(curr_rs["decode_time_s"])),
                    "total_time": self._ratio(float(prev_rs["total_time_s"]), float(curr_rs["total_time_s"])),
                    "decode_tok_per_s": float(curr_rs["tok_per_s"]["decode"]) / float(prev_rs["tok_per_s"]["decode"]),
                }
            except Exception:
                continue

            rows.append(
                {
                    "model": row["model"],
                    "case": row["case"],
                    "pair": row["pair"],
                    "previous_run": prev_summary.get("run"),
                    "current_run": row["run"],
                    "speedup": speedup,
                }
            )

        return {
            "schema_version": 1,
            "previous_run": prev_summary.get("run") if isinstance(prev_summary, dict) else None,
            "rows": rows,
        }

    @staticmethod
    def _capability_payload(adapter: Any, *, model_dir: Path, root: Path) -> dict[str, Any]:
        python_enabled, python_reason = adapter.python_baseline_status(model_dir=model_dir)
        rust_enabled, rust_reason = adapter.rust_support_status(root=root)
        strict_enabled, strict_reason = adapter.strict_status(model_dir=model_dir)

        return {
            "python_enabled": python_enabled,
            "python_skip_reason": python_reason,
            "rust_enabled": rust_enabled,
            "rust_skip_reason": rust_reason,
            "strict_enabled": strict_enabled,
            "strict_skip_reason": strict_reason,
            "declared": adapter.describe_capabilities(),
        }

    @staticmethod
    def _model_skip_row(
        *,
        run: str,
        model: str,
        suite: str,
        case: str,
        pair: str,
        reason: str,
        capabilities: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run": run,
            "model": model,
            "suite": suite,
            "case": case,
            "pair": pair,
            "all_match": False,
            "token_match": False,
            "prompt_match": False,
            "strict_status": "skipped",
            "skip_reason": reason,
            "capabilities": capabilities,
            "python_metrics": None,
            "rust_metrics": None,
        }

    def run_perf(self, args: Any) -> int:
        root = repo_root()
        runtime = runtime_paths(runtime_root=getattr(args, "runtime_root", None), create_dirs=True)

        models = self._default_models(args)
        include_devices = self._default_devices(args)
        include_precision = self._default_precisions(args)
        run = args.run or args.tag or "latest"
        run_root = self._resolve_run_root(args, run)

        overall_pairs: list[dict[str, Any]] = []
        overall_skipped: list[dict[str, Any]] = []

        for model_name in models:
            adapter = get_adapter(model_name)
            suite = getattr(adapter, "suite_name", model_name.replace("-", "_"))

            model_dir = adapter.resolve_model_dir(
                root=root,
                override=args.model_dir,
                runtime_root=runtime.root,
            )
            run_root.mkdir(parents=True, exist_ok=True)

            case_rows: list[dict[str, Any]]
            if args.image is not None and args.prompt is not None:
                case_rows = [
                    {
                        "case": args.case_name or "adhoc",
                        "image": str(args.image if args.image.is_absolute() else root / args.image),
                        "prompt": str(args.prompt),
                        "max_new_tokens": int(args.max_new_tokens or 64),
                        "baseline_json": None,
                        "rendered_prompt": None,
                    }
                ]
            elif args.baseline_json is not None or args.case_name:
                image, prompt, max_new, baseline_json, baseline = self._resolve_case_assets(
                    args,
                    adapter=adapter,
                    root=root,
                    model_root=Path("baselines") / suite,
                )
                case_rows = [
                    {
                        "case": args.case_name or "adhoc",
                        "image": str(image),
                        "prompt": prompt,
                        "max_new_tokens": max_new,
                        "baseline_json": str(baseline_json) if baseline_json is not None else None,
                        "rendered_prompt": baseline.get("rendered_prompt") if isinstance(baseline, dict) else None,
                    }
                ]
            else:
                case_rows = self._adapter_case_matrix(
                    adapter=adapter,
                    root=root,
                    limit=args.limit,
                    cases=args.cases,
                )
                for row in case_rows:
                    row["baseline_json"] = None
                    row["rendered_prompt"] = None

            if not case_rows:
                raise SystemExit(f"no perf cases selected for model={model_name}")

            pair_defs = self._build_pairs_for_adapter(
                adapter=adapter,
                devices=include_devices,
                precisions=include_precision,
            )
            if not pair_defs:
                raise SystemExit("no runnable device/precision pairs after filtering")

            model_capabilities = self._capability_payload(adapter, model_dir=model_dir, root=root)

            task_per_pair = 1 + (1 if model_capabilities["python_enabled"] else 0)
            task_total = len(case_rows) * len(pair_defs) * task_per_pair
            task_bar = tqdm(total=task_total, desc=f"perf:{model_name}", unit="task") if tqdm is not None else None

            for case in case_rows:
                case_name = str(case["case"])
                image = Path(str(case["image"]))
                prompt = str(case["prompt"])
                max_new = int(case["max_new_tokens"])
                rendered_prompt = case.get("rendered_prompt")
                baseline_json = case.get("baseline_json")

                for idx, pair in enumerate(pair_defs, 1):
                    pair_name = f"{pair['py_device']}_{pair['py_dtype']}"
                    if task_bar is None:
                        print(f"[{idx}/{len(pair_defs)}] {model_name}:{case_name}:{pair_name}")
                    else:
                        task_bar.set_postfix_str(f"{case_name}:{pair_name}:rs")

                    pair_root = run_root / "perf" / model_name / case_name / pair_name
                    py_out = pair_root / "python" / "bench.json"
                    rs_out = pair_root / "rust" / "bench.json"
                    compare_out = pair_root / "compare.json"

                    if not model_capabilities["rust_enabled"]:
                        skipped = self._model_skip_row(
                            run=run,
                            model=model_name,
                            suite=suite,
                            case=case_name,
                            pair=pair_name,
                            reason=model_capabilities["rust_skip_reason"] or "rust path disabled",
                            capabilities=model_capabilities,
                        )
                        overall_skipped.append(skipped)
                        write_json(compare_out, skipped)
                        if task_bar is not None:
                            task_bar.update(task_per_pair)
                        continue

                    rendered_for_rust = rendered_prompt
                    py_metrics: dict[str, Any] | None = None
                    rs_metrics: dict[str, Any] | None = None

                    if task_bar is not None:
                        task_bar.set_postfix_str(f"{case_name}:{pair_name}:rs")
                    try:
                        rs_metrics = adapter.run_rust_bench(
                            cli=args.cli,
                            image=image,
                            prompt=prompt,
                            rendered_prompt=rendered_for_rust,
                            max_new_tokens=max_new,
                            rs_device=pair["rs_device"],
                            rs_dtype=pair["rs_dtype"],
                            output=rs_out,
                            repo_root=root,
                            runtime_root=runtime.root,
                        )
                    except Exception as exc:
                        error_reason = f"rust bench failed: {exc}"
                        print(
                            f"[perf][error] model={model_name} case={case_name} pair={pair_name} stage=rust error={exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        row = {
                            "schema_version": 1,
                            "run": run,
                            "model": model_name,
                            "suite": suite,
                            "case": case_name,
                            "pair": pair_name,
                            "python_device": pair["py_device"],
                            "python_dtype": pair["py_dtype"],
                            "rust_device": pair["rs_device"],
                            "rust_dtype": pair["rs_dtype"],
                            "image": str(image),
                            "prompt": prompt,
                            "max_new_tokens": max_new,
                            "baseline_json": baseline_json,
                            "capabilities": model_capabilities,
                            "python_metrics": None,
                            "rust_metrics": None,
                            "strict_status": "error",
                            "all_match": False,
                            "token_match": None,
                            "prompt_match": None,
                            "token_diff": None,
                            "prompt_diff": None,
                            "error": error_reason,
                            "skip_reason": error_reason,
                        }
                        overall_pairs.append(row)
                        write_json(compare_out, row)
                        if task_bar is not None:
                            task_bar.update(task_per_pair)
                        continue

                    if task_bar is not None:
                        task_bar.update(1)

                    if model_capabilities["python_enabled"]:
                        if task_bar is not None:
                            task_bar.set_postfix_str(f"{case_name}:{pair_name}:py")
                        try:
                            py_metrics = adapter.run_python_bench(
                                model_dir=model_dir,
                                image=image,
                                prompt=prompt,
                                max_new_tokens=max_new,
                                py_device=pair["py_device"],
                                py_dtype=pair["py_dtype"],
                                output=py_out,
                                repo_root=root,
                                runtime_root=runtime.root,
                            )
                            if rendered_for_rust is None:
                                rendered_for_rust = py_metrics.get("rendered_prompt")
                        except Exception as exc:
                            error_reason = f"python bench failed: {exc}"
                            print(
                                f"[perf][error] model={model_name} case={case_name} pair={pair_name} stage=python error={exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            row = {
                                "schema_version": 1,
                                "run": run,
                                "model": model_name,
                                "suite": suite,
                                "case": case_name,
                                "pair": pair_name,
                                "python_device": pair["py_device"],
                                "python_dtype": pair["py_dtype"],
                                "rust_device": pair["rs_device"],
                                "rust_dtype": pair["rs_dtype"],
                                "image": str(image),
                                "prompt": prompt,
                                "max_new_tokens": max_new,
                                "baseline_json": baseline_json,
                                "capabilities": model_capabilities,
                                "python_metrics": None,
                                "rust_metrics": rs_metrics,
                                "strict_status": "error",
                                "all_match": False,
                                "token_match": None,
                                "prompt_match": None,
                                "token_diff": None,
                                "prompt_diff": None,
                                "error": error_reason,
                                "skip_reason": error_reason,
                            }
                            overall_pairs.append(row)
                            write_json(compare_out, row)
                            if task_bar is not None:
                                task_bar.update(1)
                            continue

                        if task_bar is not None:
                            task_bar.update(1)

                    row = {
                        "schema_version": 1,
                        "run": run,
                        "model": model_name,
                        "suite": suite,
                        "case": case_name,
                        "pair": pair_name,
                        "python_device": pair["py_device"],
                        "python_dtype": pair["py_dtype"],
                        "rust_device": pair["rs_device"],
                        "rust_dtype": pair["rs_dtype"],
                        "image": str(image),
                        "prompt": prompt,
                        "max_new_tokens": max_new,
                        "baseline_json": baseline_json,
                        "capabilities": model_capabilities,
                        "python_metrics": py_metrics,
                        "rust_metrics": rs_metrics,
                    }

                    if model_capabilities["strict_enabled"] and py_metrics is not None:
                        strict = self._strict_compare(py_metrics, rs_metrics)
                        row.update(
                            {
                                "strict_status": "compared",
                                "token_match": strict["token_match"],
                                "prompt_match": strict["prompt_match"],
                                "all_match": bool(strict["token_match"] and strict["prompt_match"]),
                                "token_diff": strict["token_diff"],
                                "prompt_diff": strict["prompt_diff"],
                            }
                        )
                    elif model_capabilities["strict_enabled"] and py_metrics is None:
                        error_reason = "python bench output missing for strict compare"
                        row.update(
                            {
                                "strict_status": "error",
                                "token_match": None,
                                "prompt_match": None,
                                "all_match": False,
                                "token_diff": None,
                                "prompt_diff": None,
                                "error": error_reason,
                                "skip_reason": error_reason,
                            }
                        )
                    else:
                        row.update(
                            {
                                "strict_status": "skipped",
                                "token_match": None,
                                "prompt_match": None,
                                "all_match": None,
                                "token_diff": None,
                                "prompt_diff": None,
                                "skip_reason": model_capabilities["strict_skip_reason"]
                                or model_capabilities["python_skip_reason"]
                                or "strict compare unavailable",
                            }
                        )

                    overall_pairs.append(row)
                    write_json(compare_out, row)

            if task_bar is not None:
                task_bar.close()

        if not overall_pairs and not overall_skipped:
            raise SystemExit("no perf result produced")

        grouped_root = run_root / "perf"
        grouped_root.mkdir(parents=True, exist_ok=True)
        summary_path = grouped_root / "summary.json"
        report_path = grouped_root / "report.txt"

        previous_run_dir = self._find_previous_run(Path("baselines") / "benchsuite", run)
        prev_summary: dict[str, Any] | None = None
        if previous_run_dir is not None:
            candidate = previous_run_dir / "perf" / "summary.json"
            if candidate.exists():
                prev_summary = read_json(candidate)

        prev_compare = self._build_prev_compare(current_pairs=overall_pairs, prev_summary=prev_summary)

        strict_required = [
            row
            for row in overall_pairs
            if isinstance(row.get("capabilities"), dict) and bool(row["capabilities"].get("strict_enabled"))
        ]
        compared_pairs = [row for row in strict_required if row.get("strict_status") == "compared"]
        strict_missing = [row for row in strict_required if row.get("strict_status") != "compared"]
        all_match = (not strict_missing) and all(bool(row.get("all_match")) for row in compared_pairs)

        summary = {
            "schema_version": 3,
            "run": run,
            "runtime_root": str(runtime.root),
            "include": {
                "models": models,
                "devices": include_devices,
                "precision": include_precision,
            },
            "pairs": overall_pairs,
            "skipped": overall_skipped,
            "all_match": all_match,
            "strict_compared": len(compared_pairs),
            "strict_required": len(strict_required),
            "strict_errors": len([row for row in overall_pairs if row.get("strict_status") == "error"]),
            "strict_skipped": len([row for row in overall_pairs if row.get("strict_status") == "skipped"])
            + len(overall_skipped),
            "paths": {
                "summary": str(summary_path),
                "report": str(report_path),
            },
            "previous_run_compare": prev_compare,
        }
        write_json(summary_path, summary)

        header = [
            "pair",
            "strict",
            "py_prefill",
            "rs_prefill",
            "py/rs_pf",
            "py_total",
            "rs_total",
            "py/rs_total",
            "py_decode_tps",
            "rs_decode_tps",
            "rs/py_tps_or_reason",
        ]

        display_rows = overall_pairs + overall_skipped
        rows = self._pairs_rows(display_rows)
        report_lines = [
            f"run: {run}",
            f"runtime_root: {runtime.root}",
            f"all_match: {summary['all_match']}",
            f"strict_required: {summary['strict_required']}",
            f"strict_compared: {summary['strict_compared']}",
            f"strict_errors: {summary['strict_errors']}",
            f"strict_skipped: {summary['strict_skipped']}",
            f"include_models: {','.join(models)}",
            f"include_devices: {','.join(include_devices)}",
            f"include_precision: {','.join(include_precision)}",
            "",
            self._format_table(header, rows),
        ]

        if overall_skipped:
            report_lines.append("")
            report_lines.append("skip reasons:")
            for row in overall_skipped:
                report_lines.append(f"- {row['model']}:{row['case']}:{row['pair']} -> {row['skip_reason']}")

        if prev_compare and prev_compare.get("rows"):
            report_lines.append("")
            report_lines.append(f"previous_run: {prev_compare.get('previous_run')}")
            prev_rows = [
                [
                    f"{row['model']}:{row['pair']}",
                    f"{row['speedup']['prefill_time']:.3f}x",
                    f"{row['speedup']['decode_time']:.3f}x",
                    f"{row['speedup']['total_time']:.3f}x",
                    f"{row['speedup']['decode_tok_per_s']:.3f}x",
                ]
                for row in prev_compare["rows"]
            ]
            report_lines.append(
                self._format_table(
                    ["pair", "prefill", "decode", "total", "decode_tps"],
                    prev_rows,
                )
            )

        report_path.write_text("\n".join(report_lines) + "\n")

        self._print_perf_report(display_rows)
        if prev_compare and prev_compare.get("rows"):
            print(f"\nprevious_run: {prev_compare.get('previous_run')}")
        print(f"runtime_root: {runtime.root}")
        print(f"summary: {summary_path}")
        print(f"report:  {report_path}")
        return 0 if summary["all_match"] else 1

    def run_matrix_gate(self, args: Any) -> int:
        root = repo_root()
        runtime = runtime_paths(runtime_root=getattr(args, "runtime_root", None), create_dirs=True)

        models = self._default_models(args)
        include_devices = self._default_devices(args)
        include_precision = self._default_precisions(args)
        run = args.run or args.tag or "latest"
        run_root = self._resolve_run_root(args, run)

        all_results: list[dict[str, Any]] = []
        all_errors: list[dict[str, Any]] = []

        if args.image is not None and args.prompt is not None:
            case_name = args.case_name or "adhoc"
            case_payloads = [{"case": case_name, "image": str(args.image), "prompt": args.prompt}]
            if args.max_new_tokens is None:
                raise SystemExit("adhoc matrix-gate requires --max-new-tokens")
        else:
            case_payloads = []

        for model_name in models:
            adapter = get_adapter(model_name)
            suite = getattr(adapter, "suite_name", model_name.replace("-", "_"))
            model_dir = adapter.resolve_model_dir(
                root=root,
                override=args.model_dir,
                runtime_root=runtime.root,
            )
            source_matrix = (
                Path(args.source_matrix)
                if args.source_matrix is not None
                else Path(getattr(adapter, "default_matrix_dir", Path("baselines") / suite / "matrix_v20"))
            )

            run_root.mkdir(parents=True, exist_ok=True)

            if not case_payloads:
                if not source_matrix.exists():
                    local_payloads = self._adapter_case_matrix(adapter=adapter, root=root, limit=None, cases=None)
                else:
                    case_dirs = sorted(path for path in source_matrix.iterdir() if path.is_dir() and (path / "baseline.json").exists())
                    if args.cases:
                        selected = set(args.cases)
                        case_dirs = [path for path in case_dirs if path.name in selected]
                    if args.limit is not None:
                        case_dirs = case_dirs[: args.limit]
                    if not case_dirs:
                        raise SystemExit(f"no matrix baseline cases found for model={model_name}")

                    local_payloads = []
                    for case_dir in case_dirs:
                        baseline = read_json(case_dir / "baseline.json")
                        local_payloads.append(
                            {
                                "case": case_dir.name,
                                "image": baseline["image"],
                                "prompt": baseline["prompt"],
                                "generated_tokens": baseline.get("generated_tokens"),
                            }
                        )
            else:
                local_payloads = case_payloads

            if args.cases:
                selected = set(args.cases)
                local_payloads = [row for row in local_payloads if str(row.get("case")) in selected]
            if args.limit is not None:
                local_payloads = local_payloads[: args.limit]
            if not local_payloads:
                raise SystemExit(f"no matrix-gate cases selected for model={model_name}")

            pair_defs = self._build_pairs_for_adapter(
                adapter=adapter,
                devices=include_devices,
                precisions=include_precision,
            )
            if not pair_defs:
                raise SystemExit("no runnable device/precision pairs after filtering")

            model_capabilities = self._capability_payload(adapter, model_dir=model_dir, root=root)

            total = len(local_payloads) * len(pair_defs)
            pbar = tqdm(total=total, desc=f"matrix:{model_name}", unit="case") if tqdm is not None else None

            for case in local_payloads:
                case_name = str(case["case"])
                image_raw = Path(str(case["image"]))
                image = image_raw if image_raw.is_absolute() else root / image_raw
                prompt = str(case["prompt"])

                max_new = args.max_new_tokens
                if max_new is None and "__n" in case_name:
                    tail = case_name.rsplit("__n", 1)[-1]
                    max_new = int(tail) if tail.isdigit() else None
                if max_new is None:
                    tokens = case.get("generated_tokens")
                    max_new = len(tokens) if isinstance(tokens, list) and tokens else 64

                for pair in pair_defs:
                    pair_name = f"{pair['py_device']}_{pair['py_dtype']}"
                    if pbar is None:
                        print(f"[{model_name}] {case_name} {pair_name}")
                    else:
                        pbar.set_postfix_str(f"{case_name}:{pair_name}")

                    out_case = run_root / "matrix" / model_name / case_name / pair_name
                    py_output = out_case / "python" / "bench.json"
                    rust_output = out_case / "rust_output.json"
                    compare_output = out_case / "compare.json"

                    if not model_capabilities["strict_enabled"]:
                        result = {
                            "run": run,
                            "model": model_name,
                            "suite": suite,
                            "case": case_name,
                            "pair": pair_name,
                            "strict_status": "skipped",
                            "all_match": None,
                            "prompt_match": None,
                            "token_match": None,
                            "skip_reason": model_capabilities["strict_skip_reason"]
                            or model_capabilities["python_skip_reason"]
                            or "strict compare unavailable",
                            "capabilities": model_capabilities,
                        }
                        write_json(compare_output, result)
                        all_results.append(result)
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    try:
                        py_metrics = adapter.run_python_bench(
                            model_dir=model_dir,
                            image=image,
                            prompt=prompt,
                            max_new_tokens=int(max_new),
                            py_device=pair["py_device"],
                            py_dtype=pair["py_dtype"],
                            output=py_output,
                            repo_root=root,
                            runtime_root=runtime.root,
                        )

                        rs_metrics = adapter.run_rust_infer(
                            cli=args.cli,
                            image=image,
                            prompt=prompt,
                            rendered_prompt=py_metrics.get("rendered_prompt"),
                            max_new_tokens=int(max_new),
                            rs_device=pair["rs_device"],
                            rs_dtype=pair["rs_dtype"],
                            output=rust_output,
                            repo_root=root,
                            runtime_root=runtime.root,
                        )
                    except Exception as exc:
                        result = {
                            "run": run,
                            "model": model_name,
                            "suite": suite,
                            "case": case_name,
                            "pair": pair_name,
                            "strict_status": "error",
                            "all_match": False,
                            "error": str(exc),
                            "capabilities": model_capabilities,
                        }
                        all_results.append(result)
                        all_errors.append(result)
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    strict = self._strict_compare(
                        py_metrics,
                        {
                            "generated_token_ids": rs_metrics.get("tokens", []),
                            "rendered_prompt": rs_metrics.get("rendered_prompt"),
                        },
                    )

                    result = {
                        "run": run,
                        "model": model_name,
                        "suite": suite,
                        "case": case_name,
                        "pair": pair_name,
                        "device": pair["rs_device"],
                        "dtype": pair["rs_dtype"],
                        "strict_status": "compared",
                        "prompt_match": bool(strict["prompt_match"]),
                        "token_match": bool(strict["token_match"]),
                        "all_match": bool(strict["prompt_match"] and strict["token_match"]),
                        "python_tokens": int(py_metrics.get("generated_tokens", 0)),
                        "rust_tokens": len(rs_metrics.get("tokens", [])) if isinstance(rs_metrics.get("tokens"), list) else None,
                        "earliest_divergence": strict.get("token_diff"),
                        "prompt_diff": strict.get("prompt_diff"),
                        "capabilities": model_capabilities,
                    }
                    write_json(compare_output, result)
                    all_results.append(result)
                    if not result["all_match"]:
                        all_errors.append(result)

                    if pbar is not None:
                        pbar.update(1)

            if pbar is not None:
                pbar.close()

        report_root = run_root / "matrix"
        report_root.mkdir(parents=True, exist_ok=True)
        report_json = report_root / "summary.json"
        report_txt = report_root / "report.txt"

        earliest = None
        earliest_idx: int | None = None
        compared_rows = [row for row in all_results if row.get("strict_status") == "compared"]
        for row in compared_rows:
            if row.get("all_match"):
                continue
            divergence = row.get("earliest_divergence")
            if isinstance(divergence, dict) and isinstance(divergence.get("index"), int):
                index = int(divergence["index"])
                if earliest_idx is None or index < earliest_idx:
                    earliest_idx = index
                    earliest = {
                        "model": row["model"],
                        "case": row["case"],
                        "pair": row["pair"],
                        "detail": divergence,
                    }
            elif earliest is None:
                earliest = {
                    "model": row["model"],
                    "case": row["case"],
                    "pair": row["pair"],
                    "detail": row.get("error", "unknown mismatch"),
                }

        mismatches = sum(1 for row in compared_rows if row.get("all_match") is False)
        skipped = len([row for row in all_results if row.get("strict_status") == "skipped"])

        summary = {
            "schema_version": 3,
            "run": run,
            "runtime_root": str(runtime.root),
            "include": {
                "models": models,
                "devices": include_devices,
                "precision": include_precision,
            },
            "cases": len(all_results),
            "strict_compared": len(compared_rows),
            "strict_skipped": skipped,
            "mismatches": mismatches,
            "all_match": mismatches == 0,
            "earliest_divergence": earliest,
            "results": all_results,
            "errors": all_errors,
            "paths": {
                "summary": str(report_json),
                "report": str(report_txt),
            },
        }
        write_json(report_json, summary)

        rows = []
        for row in all_results:
            status = row.get("strict_status")
            if status == "skipped":
                detail = row.get("skip_reason", "skip")
                rows.append(
                    [
                        f"{row['model']}:{row['case']}:{row['pair']}",
                        "skip",
                        "-",
                        "-",
                        detail,
                    ]
                )
                continue

            detail = "-"
            if not row.get("all_match"):
                if row.get("earliest_divergence"):
                    detail = f"idx={row['earliest_divergence']['index']}"
                elif row.get("error"):
                    detail = "error"
                elif not row.get("prompt_match"):
                    detail = "prompt"
                else:
                    detail = "mismatch"
            rows.append(
                [
                    f"{row['model']}:{row['case']}:{row['pair']}",
                    "ok" if row.get("all_match") else "fail",
                    "ok" if row.get("prompt_match") else "fail",
                    "ok" if row.get("token_match") else "fail",
                    detail,
                ]
            )

        lines = [
            f"run: {run}",
            f"runtime_root: {runtime.root}",
            f"all_match: {summary['all_match']}",
            f"strict_compared: {summary['strict_compared']}",
            f"strict_skipped: {summary['strict_skipped']}",
            f"mismatches: {mismatches}",
            f"include_models: {','.join(models)}",
            f"include_devices: {','.join(include_devices)}",
            f"include_precision: {','.join(include_precision)}",
            "",
            self._format_table(["case", "strict", "prompt", "tokens", "detail"], rows),
        ]
        report_txt.write_text("\n".join(lines) + "\n")

        print("\n=== Matrix Strict Gate ===")
        print(self._format_table(["case", "strict", "prompt", "tokens", "detail"], rows))
        print(f"runtime_root: {runtime.root}")
        print(f"summary: {report_json}")
        print(f"report:  {report_txt}")
        return 0 if mismatches == 0 else 1
