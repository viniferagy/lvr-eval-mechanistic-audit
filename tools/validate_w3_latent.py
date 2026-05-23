#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_ID = "lvr_latent_patch_answer_transfer"
MODEL_ID = "lvr_7b"
PRIMARY_SCALAR = "latent_answer_transfer_rate"


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-pairs", type=int, default=50)
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
    if reduction.get(PRIMARY_SCALAR) is None:
        fail(f"missing primary scalar {PRIMARY_SCALAR}")

    n_paired = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
    n_success = int(reduction.get("n_success") or 0)
    n_error = int(reduction.get("n_error") or 0)
    n_patch_applied = int(reduction.get("n_patch_applied") or 0)
    n_with_lvr_mode = int(reduction.get("n_with_lvr_mode") or 0)
    n_with_captured_state = int(reduction.get("n_with_captured_state") or 0)
    print(f"{METRIC_ID}: n_paired={n_paired} reduction={reduction}")
    if n_paired < args.min_pairs:
        fail(f"n_paired too small: {n_paired}")
    if n_success <= 0:
        fail("no successful latent patch samples")
    if n_error > max(1, n_success) * 0.2:
        fail(f"too many errors: {n_error}>{n_success}*0.2")
    if n_patch_applied < args.min_pairs:
        fail(f"n_patch_applied too small: {n_patch_applied}")
    if n_with_lvr_mode < args.min_pairs:
        fail(f"n_with_lvr_mode too small: {n_with_lvr_mode}")
    if n_with_captured_state < args.min_pairs:
        fail(f"n_with_captured_state too small: {n_with_captured_state}")

    valid_trace = 0
    valid_answers = 0
    valid_shapes = 0
    for record in payload.get("samples") or []:
        if record.get("error") is not None:
            continue
        if record.get("trace_v2_error"):
            fail(f"trace fallback in sample {record.get('id')}: {record.get('trace_v2_error')}")
        if record.get("missing_modules") not in ([], None):
            fail(f"missing modules in sample {record.get('id')}: {record.get('missing_modules')}")
        if record.get("trace_quality") == "instrumented_sparse_v0" and record.get("source_trace_quality") == "instrumented_sparse_v0":
            valid_trace += 1
        if record.get("clean_answer") is not None and record.get("patched_answer") is not None:
            valid_answers += 1
        shapes = record.get("captured_state_shapes") or []
        patched_shapes = record.get("patched_captured_state_shapes") or []
        if shapes and patched_shapes and shapes[-1] and patched_shapes[-1] and shapes[-1][-1] == patched_shapes[-1][-1]:
            valid_shapes += 1
    if valid_trace < args.min_pairs:
        fail(f"valid instrumented traces too few: {valid_trace}")
    if valid_answers < args.min_pairs:
        fail(f"answer records too few: {valid_answers}")
    if valid_shapes < args.min_pairs:
        fail(f"shape match records too few: {valid_shapes}")

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
    if (METRIC_ID, MODEL_ID, PRIMARY_SCALAR) not in {
        (row.get("metric_id"), row.get("model"), row.get("scalar"))
        for row in ci
        if int(row.get("n") or 0) > 0
    }:
        fail(f"summary_with_ci missing primary row {METRIC_ID}/{MODEL_ID}/{PRIMARY_SCALAR}")

    print(f"summary_with_ci rows={len(ci)}")
    print("W3 LATENT VALIDATION PASSED")


if __name__ == "__main__":
    main()
