#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_ID = "monet_latent_patch_answer_transfer"
MODEL_ID = "monet_7b"
PRIMARY_SCALAR = "latent_answer_transfer_rate"
TRACE_QUALITY = "monet_transformers_latent_mode_v0"


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def load_json(path: Path):
    if not path.is_file():
        fail(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Validate W13 Monet Transformers latent patch gate.")
    ap.add_argument("run_dir")
    ap.add_argument("--min-pairs", type=int, default=50)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    envelope = load_json(run_dir / "metrics" / f"{METRIC_ID}_{MODEL_ID}.json")
    if envelope.get("metric_id") != METRIC_ID:
        fail(f"unexpected metric id: {envelope.get('metric_id')}")
    if envelope.get("model") != MODEL_ID:
        fail(f"unexpected model id: {envelope.get('model')}")
    payload = envelope.get("payload") or {}
    reduction = payload.get("reduction") or {}
    if reduction.get(PRIMARY_SCALAR) is None:
        fail(f"missing primary scalar {PRIMARY_SCALAR}")
    if (payload.get("config") or {}).get("vllm_scheduler_native") is not False:
        fail("Monet W13 config must explicitly mark vllm_scheduler_native=false")
    if (payload.get("config") or {}).get("latent_mode_path") != "transformers_ce_patch_vec":
        fail("unexpected Monet latent mode path")

    n_paired = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
    n_success = int(reduction.get("n_success") or 0)
    n_error = int(reduction.get("n_error") or 0)
    n_patch_applied = int(reduction.get("n_patch_applied") or 0)
    n_captured = int(reduction.get("n_with_captured_state") or 0)
    print(f"{METRIC_ID}: n_paired={n_paired} reduction={reduction}")
    if n_paired < args.min_pairs:
        fail(f"n_paired too small: {n_paired}")
    if n_success < args.min_pairs:
        fail(f"n_success too small: {n_success}")
    if n_error > max(1, n_success) * 0.2:
        fail(f"too many errors: {n_error}>{n_success}*0.2")
    if n_patch_applied < args.min_pairs:
        fail(f"n_patch_applied too small: {n_patch_applied}")
    if n_captured < args.min_pairs:
        fail(f"n_with_captured_state too small: {n_captured}")

    valid_trace = 0
    answer_records = 0
    shape_records = 0
    for record in payload.get("samples") or []:
        if record.get("error") is not None:
            continue
        if (
            record.get("trace_quality") == TRACE_QUALITY
            and record.get("source_trace_quality") == TRACE_QUALITY
            and record.get("target_trace_quality") == TRACE_QUALITY
        ):
            valid_trace += 1
        if record.get("clean_answer") is not None and record.get("patched_answer") is not None:
            answer_records += 1
        shapes = record.get("captured_state_shapes") or []
        target_shapes = record.get("target_captured_state_shapes") or []
        patch_shapes = record.get("patch_state_shapes") or []
        if shapes and target_shapes and patch_shapes and shapes[-1] and target_shapes[-1] and patch_shapes[-1]:
            if shapes[-1][-1] == target_shapes[-1][-1] == patch_shapes[-1][-1]:
                shape_records += 1
    if valid_trace < args.min_pairs:
        fail(f"valid Monet latent traces too few: {valid_trace}")
    if answer_records < args.min_pairs:
        fail(f"answer records too few: {answer_records}")
    if shape_records < args.min_pairs:
        fail(f"shape match records too few: {shape_records}")

    lock = load_json(run_dir / "prereg.lock.json")
    manifest = lock.get("manifest") or lock
    metric_ids = {
        item.get("metric_id")
        for item in manifest.get("experimental_metrics", [])
        if isinstance(item, dict)
    }
    if METRIC_ID not in metric_ids:
        fail("prereg lock missing Monet latent experimental metric")

    sanity = load_json(run_dir / "sanity" / "summary_sanity.json")
    if sanity.get("overall_status") != "pass":
        fail(f"sanity overall_status={sanity.get('overall_status')}")

    ci = load_json(run_dir / "summary_with_ci.json")
    if (METRIC_ID, MODEL_ID, PRIMARY_SCALAR) not in {
        (row.get("metric_id"), row.get("model"), row.get("scalar"))
        for row in ci
        if int(row.get("n") or 0) > 0
    }:
        fail(f"summary_with_ci missing primary row {METRIC_ID}/{MODEL_ID}/{PRIMARY_SCALAR}")
    print(f"summary_with_ci rows={len(ci)}")
    print("MONET LATENT VALIDATION PASSED")


if __name__ == "__main__":
    main()
