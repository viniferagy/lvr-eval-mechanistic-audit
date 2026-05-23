#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_METRICS = {
    "pf_a_corruption_selectivity",
    "pf_b_patch_alignment",
    "bf_patch_answer_transfer",
    "bf_swap_latent_replacement",
    "bf_conf_calibrated_progression",
    "cf_stage_decay",
}
PAIRED_METRICS = {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}
PRIMARY_SCALARS = {
    "pf_a_corruption_selectivity": "selectivity",
    "pf_b_patch_alignment": "native_alignment",
    "bf_patch_answer_transfer": "logprob_margin_shift",
    "bf_swap_latent_replacement": "swap_margin_shift",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_retention",
}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-pairs", type=int, default=50)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metric_files = sorted((run_dir / "metrics").glob("*.json"))
    if not metric_files:
        fail(f"no metric files under {run_dir}/metrics")

    seen = {}
    by_metric: dict[str, list[dict]] = {}
    for path in metric_files:
        obj = json.loads(path.read_text())
        metric_id = obj.get("metric_id")
        payload = obj.get("payload", {})
        if metric_id not in seen:
            seen[metric_id] = payload
        by_metric.setdefault(metric_id, []).append(payload)

        n_paired = payload.get("n_paired")
        reduction = payload.get("reduction")
        cells = payload.get("cells", [])

        print(f"{metric_id}: n_paired={n_paired}, reduction={reduction}")
        if metric_id in REQUIRED_METRICS:
            if reduction is None:
                fail(f"{metric_id} reduction is null")
            scalar = PRIMARY_SCALARS[metric_id]
            if scalar not in reduction or reduction.get(scalar) is None:
                fail(f"{metric_id} missing primary scalar {scalar} in reduction")
        if metric_id in PAIRED_METRICS:
            if n_paired is None or int(n_paired) < args.min_pairs:
                fail(f"{metric_id} n_paired too small: {n_paired}")
            if not cells:
                fail(f"{metric_id} has no cells")
            total_success = sum(int(c.get("n_success", 0)) for c in cells)
            total_error = sum(int(c.get("n_error", 0)) for c in cells)
            print(f"  total_success={total_success} total_error={total_error}")
            if total_success <= 0:
                fail(f"{metric_id} has zero successful patch cells")
            if total_error > total_success:
                fail(f"{metric_id} has too many errors: {total_error}>{total_success}")

    missing = REQUIRED_METRICS - set(seen)
    if missing:
        fail(f"missing metrics: {sorted(missing)}")
    for metric_id in REQUIRED_METRICS:
        n_models = len(by_metric.get(metric_id, []))
        if n_models < 3:
            fail(f"{metric_id} expected 3 model payloads, found {n_models}")

    ci_path = run_dir / "summary_with_ci.json"
    if not ci_path.is_file():
        fail("missing summary_with_ci.json")
    ci = json.loads(ci_path.read_text())
    if not ci:
        fail("empty summary_with_ci.json")
    ci_metrics = {row.get("metric_id") for row in ci if int(row.get("n") or 0) > 0}
    missing_ci = REQUIRED_METRICS - ci_metrics
    if missing_ci:
        fail(f"summary_with_ci missing metric rows: {sorted(missing_ci)}")
    primary_ci = {
        (row.get("metric_id"), row.get("model"), row.get("scalar"))
        for row in ci
        if int(row.get("n") or 0) > 0
    }
    for metric_id, scalar in PRIMARY_SCALARS.items():
        for model in ("qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b"):
            if (metric_id, model, scalar) not in primary_ci:
                fail(f"summary_with_ci missing primary row: {metric_id}/{model}/{scalar}")
    sanity_path = run_dir / "sanity" / "summary_sanity.json"
    if not sanity_path.is_file():
        fail("missing sanity/summary_sanity.json")
    sanity = json.loads(sanity_path.read_text())
    if sanity.get("overall_status") != "pass":
        fail(f"sanity overall_status={sanity.get('overall_status')}")
    print(f"summary_with_ci rows={len(ci)}")
    print("SPD RANGE VALIDATION PASSED")


if __name__ == "__main__":
    main()
