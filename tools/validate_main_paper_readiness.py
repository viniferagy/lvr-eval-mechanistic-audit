#!/usr/bin/env python
"""Validate Main-readiness artifacts at configurable strictness levels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PRIMARY_SCALARS = {
    "pf_a_corruption_selectivity": "selectivity",
    "pf_b_patch_alignment": "native_alignment",
    "bf_patch_answer_transfer": "continuous_margin_shift",
    "bf_swap_latent_replacement": "continuous_margin_shift",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_delta",
}
MAIN_TASKS = {"maze", "spd_faith", "blink"}
MAIN_MODELS = {"qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b"}
T4_MODELS = {"qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b"}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def _read_json(path: Path):
    if not path.is_file():
        fail(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_has_evidence_policy(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    required = [
        "qwen_query_span_control",
        "lvr_generation_trace_latent",
        "monet_transformers_latent_range_gate",
    ]
    return all(item in text for item in required)


def _ci_keys(ci: list[dict]) -> set[tuple[str, str, str, str]]:
    return {
        (
            str(row.get("task") or "unknown"),
            str(row.get("model") or "unknown"),
            str(row.get("metric_id") or "unknown"),
            str(row.get("scalar") or "unknown"),
        )
        for row in ci
        if int(row.get("n") or 0) > 0
    }


def _require_full_main_ci(ci: list[dict]) -> None:
    keys = _ci_keys(ci)
    missing = []
    for task in sorted(MAIN_TASKS):
        metrics = PRIMARY_SCALARS if task == "spd_faith" else {
            metric: scalar
            for metric, scalar in PRIMARY_SCALARS.items()
            if metric not in {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}
        }
        for model in sorted(MAIN_MODELS):
            for metric_id, scalar in metrics.items():
                if (task, model, metric_id, scalar) not in keys:
                    missing.append(f"{task}/{model}/{metric_id}/{scalar}")
    if missing:
        fail(f"full_main_matrix missing CI rows: {missing[:10]}")


def _accuracy_summary_path(run_dir: Path, override: str | None = None) -> Path:
    if override:
        path = Path(override)
        if path.is_file():
            return path
        fail(f"paper_ready T4 accuracy summary override is missing: {path}")
    candidates = [
        run_dir / "vsi_accuracy_summary.json",
        run_dir / "merged" / "vsi_accuracy_summary.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    fail("paper_ready requires T4 vsi_accuracy_summary.json")


def _require_t4_accuracy(run_dir: Path, *, summary_path: str | None, min_samples: int, max_error_ratio: float) -> None:
    acc_path = _accuracy_summary_path(run_dir, summary_path)
    payload = _read_json(acc_path)
    model_entries = payload.get("models")
    if not isinstance(model_entries, dict) or not model_entries:
        fail(f"T4 accuracy summary has no model entries: {acc_path}")
    missing = T4_MODELS - set(model_entries)
    if missing:
        fail(f"T4 accuracy missing required models: {sorted(missing)}")
    errors = []
    for model_id in sorted(T4_MODELS):
        entry = model_entries.get(model_id) or {}
        n = int(entry.get("n_samples") or entry.get("n") or 0)
        n_total = int(entry.get("n_total") or n)
        n_errors = int(entry.get("n_errors") or entry.get("n_error") or 0)
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
    if errors:
        fail("T4 accuracy validation failed: " + "; ".join(errors))


def _require_paper_ready(
    run_dir: Path,
    ci: list[dict],
    *,
    t4_accuracy_summary: str | None,
    t4_min_samples: int,
    t4_max_error_ratio: float,
) -> None:
    _require_full_main_ci(ci)
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        fail("paper_ready requires merged metrics directory")
    if t4_accuracy_summary is None and not any("output_accuracy_sanity" in path.name for path in metrics_dir.glob("*.json")):
        fail("paper_ready requires T4 output_accuracy_sanity metric artifact")
    _require_t4_accuracy(
        run_dir,
        summary_path=t4_accuracy_summary,
        min_samples=t4_min_samples,
        max_error_ratio=t4_max_error_ratio,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--manifest", default="prereg/manifest.yaml")
    ap.add_argument("--require-margin-quality", action="store_true")
    ap.add_argument("--t4-accuracy-summary", default=None)
    ap.add_argument("--t4-min-samples", type=int, default=800)
    ap.add_argument("--t4-max-error-ratio", type=float, default=0.2)
    ap.add_argument(
        "--mode",
        choices=["artifact_contract", "full_main_matrix", "paper_ready"],
        default="artifact_contract",
        help="artifact_contract checks files; full_main_matrix requires T1/T2/T3 CI coverage; paper_ready also requires T4 accuracy.",
    )
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    manifest = Path(args.manifest)
    if not manifest.is_file():
        fail(f"missing manifest: {manifest}")
    if not _manifest_has_evidence_policy(manifest):
        fail("manifest missing explicit evidence_level policy keys")

    ci = _read_json(run_dir / "summary_with_ci.json")
    if not isinstance(ci, list) or not ci:
        fail("summary_with_ci.json is empty")
    if args.mode in {"full_main_matrix", "paper_ready"}:
        _require_full_main_ci(ci)
    if args.mode == "paper_ready":
        _require_paper_ready(
            run_dir,
            ci,
            t4_accuracy_summary=args.t4_accuracy_summary,
            t4_min_samples=args.t4_min_samples,
            t4_max_error_ratio=args.t4_max_error_ratio,
        )

    mixed = _read_json(run_dir / "mixed_effects_summary.json")
    if not str(mixed.get("status", "")).startswith("implemented"):
        fail(f"mixed_effects_summary status is not implemented: {mixed.get('status')}")

    sanity = _read_json(run_dir / "sanity" / "summary_sanity.json")
    if sanity.get("overall_status") not in {"pass", "warn"}:
        fail(f"sanity overall_status={sanity.get('overall_status')}")

    if args.require_margin_quality:
        from tools.validate_trace_margin_quality import main as validate_margin

        validate_margin([str(run_dir)])

    label = {
        "artifact_contract": "MAIN ARTIFACT CONTRACT VALIDATION PASSED",
        "full_main_matrix": "MAIN MATRIX READINESS VALIDATION PASSED",
        "paper_ready": "MAIN PAPER READINESS VALIDATION PASSED",
    }[args.mode]
    print(label)


if __name__ == "__main__":
    main()
