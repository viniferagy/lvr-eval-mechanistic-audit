#!/usr/bin/env python
"""Validate Main-readiness artifacts at configurable strictness levels."""
from __future__ import annotations

import argparse
import json
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


def _require_paper_ready(run_dir: Path, ci: list[dict]) -> None:
    _require_full_main_ci(ci)
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        fail("paper_ready requires merged metrics directory")
    if not any("output_accuracy_sanity" in path.name for path in metrics_dir.glob("*.json")):
        fail("paper_ready requires T4 output_accuracy_sanity metric artifact")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--manifest", default="prereg/manifest.yaml")
    ap.add_argument("--require-margin-quality", action="store_true")
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
        _require_paper_ready(run_dir, ci)

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
