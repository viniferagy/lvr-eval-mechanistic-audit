#!/usr/bin/env python
"""Merge T4/VSI output-accuracy runs into a paper-ready summary artifact."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.analysis import build_summary_with_ci
from pipeline.results import make_metric_result
from pipeline.sanity.report import run_sanity_for_metric_results, save_sanity_reports


REQUIRED_MODELS = {"qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b"}
DATASET_BOUNDARY = (
    "Official nyu-visionx/VSI-Bench visual subset staged from available HF video "
    "zips/frame-grids. Report prepare_stats source, per-dataset coverage, and "
    "whether this is scannetpp-only or the full three-source visual set."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_accuracy_envelopes(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for path in sorted(root.glob("**/metrics/*output_accuracy_sanity*.json")):
            if "/merged/" in str(path):
                continue
            env = read_json(path)
            if not isinstance(env, dict) or env.get("metric_id") != "output_accuracy_sanity":
                continue
            payload = env.get("payload")
            if not isinstance(payload, dict):
                continue
            run_dir = path.parent.parent
            payload.setdefault("task", "vsi")
            payload.setdefault("source_run_dir", str(run_dir))
            config = payload.setdefault("config", {})
            if isinstance(config, dict):
                config.setdefault("task", "vsi")
            env["task"] = "vsi"
            env["source_run_dir"] = str(run_dir)
            env["source_file"] = str(path)
            out.append(env)
    return out


def entry_from_envelope(env: dict) -> dict:
    payload = env.get("payload") or {}
    reduction = payload.get("reduction") or {}
    samples = payload.get("samples") or []
    n = int(reduction.get("n") or 0)
    n_total = int(reduction.get("n_total") or len(samples) or n)
    n_error = int(reduction.get("n_error") or max(0, n_total - n))
    accuracy = reduction.get("accuracy")
    try:
        accuracy = float(accuracy)
    except (TypeError, ValueError):
        accuracy = None
    hits = sum(1 for row in samples if row.get("error") is None and bool(row.get("hit")))
    return {
        "model": str(env.get("model") or payload.get("model") or "unknown"),
        "accuracy": accuracy,
        "n_samples": n,
        "n_total": n_total,
        "n_errors": n_error,
        "error_ratio": (n_error / n_total) if n_total else None,
        "n_hits": hits,
        "source_run_dir": env.get("source_run_dir"),
        "source_file": env.get("source_file"),
    }


def validate_summary(summary: dict, *, required_models: set[str], min_samples: int, max_error_ratio: float) -> list[str]:
    errors = []
    models = summary.get("models") or {}
    missing = required_models - set(models)
    if missing:
        errors.append(f"missing required model(s): {sorted(missing)}")
    for model_id, entry in sorted(models.items()):
        n = int(entry.get("n_samples") or 0)
        n_total = int(entry.get("n_total") or n)
        n_errors = int(entry.get("n_errors") or 0)
        accuracy = entry.get("accuracy")
        if n < min_samples:
            errors.append(f"{model_id} n_samples={n} < {min_samples}")
        try:
            acc = float(accuracy)
        except (TypeError, ValueError):
            errors.append(f"{model_id} accuracy={accuracy!r} is not numeric")
        else:
            if not math.isfinite(acc) or not 0.0 <= acc <= 1.0:
                errors.append(f"{model_id} accuracy={accuracy!r} not finite in [0, 1]")
        if n_total <= 0:
            errors.append(f"{model_id} n_total={n_total} <= 0")
        else:
            err_ratio = n_errors / n_total
            if err_ratio > max_error_ratio:
                errors.append(f"{model_id} error_ratio={err_ratio:.3f} > {max_error_ratio:.3f}")
    return errors


def candidate_rank(entry: dict, *, min_samples: int, max_error_ratio: float) -> tuple[int, int, float]:
    """Rank duplicate artifacts for the same model.

    Prefer paper-valid artifacts, then higher n, then lower error ratio. This
    lets a fixed rerun supersede an earlier failed metric JSON without deleting
    generated evidence from disk.
    """
    n = int(entry.get("n_samples") or 0)
    n_total = int(entry.get("n_total") or n)
    n_errors = int(entry.get("n_errors") or 0)
    err_ratio = (n_errors / n_total) if n_total else 1.0
    accuracy = entry.get("accuracy")
    try:
        acc = float(accuracy)
    except (TypeError, ValueError):
        acc = float("nan")
    valid = (
        n >= min_samples
        and n_total > 0
        and err_ratio <= max_error_ratio
        and math.isfinite(acc)
        and 0.0 <= acc <= 1.0
    )
    return (1 if valid else 0, n, -err_ratio)


def write_metric_copies(envelopes: list[dict], out_dir: Path) -> None:
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for env in envelopes:
        model = str(env.get("model") or "unknown")
        path = metrics_dir / f"vsi_output_accuracy_sanity_{model}.json"
        path.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_config(envelopes: list[dict], out_dir: Path) -> None:
    for env in envelopes:
        source = env.get("source_run_dir")
        if not source:
            continue
        cfg = Path(str(source)) / "config_snapshot.yaml"
        if cfg.is_file():
            shutil.copyfile(cfg, out_dir / "config_snapshot.yaml")
            return


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/main_vsi_accuracy/merged"))
    ap.add_argument("--min-samples", type=int, default=800)
    ap.add_argument("--max-error-ratio", type=float, default=0.2)
    ap.add_argument("--allow-missing-models", action="store_true")
    args = ap.parse_args(argv)

    envelopes = load_accuracy_envelopes(args.run_roots)
    if not envelopes:
        raise SystemExit("FAIL: no output_accuracy_sanity metric artifacts found")

    candidates_by_model: dict[str, list[tuple[dict, dict]]] = {}
    for env in envelopes:
        model = str(env.get("model") or "unknown")
        candidates_by_model.setdefault(model, []).append((env, entry_from_envelope(env)))

    by_model: dict[str, dict] = {}
    chosen: list[dict] = []
    skipped_duplicates: list[dict] = []
    for model, candidates in sorted(candidates_by_model.items()):
        ranked = sorted(
            candidates,
            key=lambda pair: candidate_rank(pair[1], min_samples=args.min_samples, max_error_ratio=args.max_error_ratio),
            reverse=True,
        )
        chosen_env, chosen_entry = ranked[0]
        by_model[model] = chosen_entry
        chosen.append(chosen_env)
        for _env, entry in ranked[1:]:
            skipped_duplicates.append(entry)

    summary = {
        "schema": "lvr-eval t4_vsi_accuracy_summary v1",
        "task": "vsi",
        "dataset_boundary": DATASET_BOUNDARY,
        "n_models": len(by_model),
        "models": by_model,
        "n_candidates": len(envelopes),
        "skipped_duplicate_candidates": skipped_duplicates,
        "required_models": sorted(REQUIRED_MODELS),
        "min_samples": args.min_samples,
        "max_error_ratio": args.max_error_ratio,
    }
    required = set() if args.allow_missing_models else REQUIRED_MODELS
    errors = validate_summary(
        summary,
        required_models=required,
        min_samples=args.min_samples,
        max_error_ratio=args.max_error_ratio,
    )
    if errors:
        raise SystemExit("FAIL: " + "; ".join(errors))

    args.out.mkdir(parents=True, exist_ok=True)
    write_metric_copies(chosen, args.out)
    copy_config(chosen, args.out)
    (args.out / "vsi_accuracy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ci_rows = build_summary_with_ci(chosen)
    (args.out / "summary_with_ci.json").write_text(
        json.dumps(ci_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    reports = run_sanity_for_metric_results(
        chosen,
        cfg={"validation": {"output_accuracy": {"min_samples": 1, "max_error_ratio": args.max_error_ratio}}},
    )
    save_sanity_reports(reports, str(args.out))
    (args.out / "merge_manifest.json").write_text(
        json.dumps({
            "run_roots": [str(root) for root in args.run_roots],
            "n_metric_results": len(chosen),
            "models": sorted(by_model),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"MERGED_VSI_ACCURACY {args.out} models={','.join(sorted(by_model))}")


if __name__ == "__main__":
    main()
