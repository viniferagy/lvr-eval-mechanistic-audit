#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_ID = "lvr_latent_patch_answer_transfer"
MODEL_ID = "lvr_7b"
PRIMARY_SCALAR = "best_step_transfer_rate"
REQUIRED_SCALARS = {
    "best_step_transfer_rate",
    "step_transfer_auc",
    "last_step_transfer_rate",
    "n_steps_evaluated",
}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-pairs", type=int, default=50)
    ap.add_argument("--min-steps", type=int, default=2)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metric_path = run_dir / "metrics" / f"{METRIC_ID}_{MODEL_ID}.json"
    if not metric_path.is_file():
        fail(f"missing metric file: {metric_path}")
    envelope = load_json(metric_path)
    payload = envelope.get("payload") or {}
    reduction = payload.get("reduction") or {}
    if envelope.get("metric_id") != METRIC_ID:
        fail(f"unexpected metric id: {envelope.get('metric_id')}")
    if envelope.get("model") != MODEL_ID:
        fail(f"unexpected model id: {envelope.get('model')}")
    missing = [scalar for scalar in sorted(REQUIRED_SCALARS) if reduction.get(scalar) is None]
    if missing:
        fail(f"missing W4 scalar(s): {missing}")

    n_paired = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
    n_success = int(reduction.get("n_success") or 0)
    n_error = int(reduction.get("n_error") or 0)
    n_steps = int(reduction.get("n_steps_evaluated") or 0)
    print(f"{METRIC_ID}: n_paired={n_paired} n_steps={n_steps} reduction={reduction}")
    if n_paired < args.min_pairs:
        fail(f"n_paired too small: {n_paired}")
    if n_success < args.min_pairs:
        fail(f"n_success too small: {n_success}")
    if n_error > max(1, n_success) * 0.2:
        fail(f"too many errors: {n_error}>{n_success}*0.2")
    if n_steps < args.min_steps:
        fail(f"n_steps_evaluated too small: {n_steps}")
    per_step = reduction.get("per_step") or []
    if len(per_step) < args.min_steps:
        fail(f"per_step rows too few: {len(per_step)}")

    valid_trace = 0
    valid_step_records = 0
    for record in payload.get("samples") or []:
        if record.get("error") is not None:
            continue
        if record.get("trace_v2_error"):
            fail(f"trace fallback in sample {record.get('id')}: {record.get('trace_v2_error')}")
        if record.get("missing_modules") not in ([], None):
            fail(f"missing modules in sample {record.get('id')}: {record.get('missing_modules')}")
        if record.get("trace_quality") == "instrumented_sparse_v0" and record.get("source_trace_quality") == "instrumented_sparse_v0":
            valid_trace += 1
        step_results = record.get("step_results") or []
        if len(step_results) >= args.min_steps:
            valid_step_records += 1
        for step in step_results:
            if step.get("trace_v2_error"):
                fail(f"trace fallback in sample {record.get('id')} step {step.get('step_index')}")
            if step.get("missing_modules") not in ([], None):
                fail(f"missing modules in sample {record.get('id')} step {step.get('step_index')}")
            if int(step.get("n_patch_applied") or 0) < 1:
                fail(f"patch not applied in sample {record.get('id')} step {step.get('step_index')}")
    if valid_trace < args.min_pairs:
        fail(f"valid instrumented traces too few: {valid_trace}")
    if valid_step_records < args.min_pairs:
        fail(f"samples with per-step records too few: {valid_step_records}")

    if not (run_dir / "prereg.lock.json").is_file():
        fail("missing prereg.lock.json")
    sanity_path = run_dir / "sanity" / "summary_sanity.json"
    if not sanity_path.is_file():
        fail("missing sanity/summary_sanity.json")
    sanity = load_json(sanity_path)
    if sanity.get("overall_status") != "pass":
        fail(f"sanity overall_status={sanity.get('overall_status')}")
    ci_path = run_dir / "summary_with_ci.json"
    if not ci_path.is_file():
        fail("missing summary_with_ci.json")
    ci = load_json(ci_path)
    rows = {
        (row.get("metric_id"), row.get("model"), row.get("scalar"))
        for row in ci
        if int(row.get("n") or 0) > 0
    }
    if (METRIC_ID, MODEL_ID, PRIMARY_SCALAR) not in rows:
        fail(f"summary_with_ci missing primary row {METRIC_ID}/{MODEL_ID}/{PRIMARY_SCALAR}")
    if (METRIC_ID, MODEL_ID, "step_transfer_auc") not in rows:
        fail(f"summary_with_ci missing step_transfer_auc row {METRIC_ID}/{MODEL_ID}")

    print(f"summary_with_ci rows={len(ci)}")
    print("W4 STEP SWEEP VALIDATION PASSED")


if __name__ == "__main__":
    main()
