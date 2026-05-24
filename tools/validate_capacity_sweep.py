#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_ID = "lvr_latent_patch_answer_transfer"
MODEL_ID = "lvr_7b"
PRIMARY_SCALARS = {
    "best_step_transfer_rate",
    "step_transfer_auc",
    "last_step_transfer_rate",
}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_payload(run_dir: Path) -> dict:
    path = run_dir / "metrics" / f"{METRIC_ID}_{MODEL_ID}.json"
    if not path.is_file():
        fail(f"missing metric file: {path}")
    envelope = load_json(path)
    if envelope.get("metric_id") != METRIC_ID or envelope.get("model") != MODEL_ID:
        fail(f"unexpected envelope in {path}")
    return envelope.get("payload") or {}


def validate_run(run_dir: Path, min_pairs: int, min_steps: int) -> dict:
    payload = metric_payload(run_dir)
    reduction = payload.get("reduction") or {}
    n_paired = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
    n_success = int(reduction.get("n_success") or 0)
    n_error = int(reduction.get("n_error") or 0)
    n_steps = int(reduction.get("n_steps_evaluated") or 0)
    if n_paired < min_pairs:
        fail(f"{run_dir}: n_paired too small: {n_paired}")
    if n_success < min_pairs:
        fail(f"{run_dir}: n_success too small: {n_success}")
    if n_error > max(1, n_success) * 0.2:
        fail(f"{run_dir}: too many errors: {n_error}>{n_success}*0.2")
    if n_steps < min_steps:
        fail(f"{run_dir}: n_steps_evaluated too small: {n_steps}")
    for scalar in PRIMARY_SCALARS:
        if reduction.get(scalar) is None:
            fail(f"{run_dir}: missing scalar {scalar}")
    if not (run_dir / "prereg.lock.json").is_file():
        fail(f"{run_dir}: missing prereg.lock.json")
    sanity_path = run_dir / "sanity" / "summary_sanity.json"
    if not sanity_path.is_file():
        fail(f"{run_dir}: missing sanity summary")
    sanity = load_json(sanity_path)
    if sanity.get("overall_status") != "pass":
        fail(f"{run_dir}: sanity overall_status={sanity.get('overall_status')}")
    ci_path = run_dir / "summary_with_ci.json"
    if not ci_path.is_file():
        fail(f"{run_dir}: missing summary_with_ci.json")
    ci = load_json(ci_path)
    rows = {
        (row.get("metric_id"), row.get("model"), row.get("scalar"))
        for row in ci
        if int(row.get("n") or 0) > 0
    }
    for scalar in PRIMARY_SCALARS:
        if (METRIC_ID, MODEL_ID, scalar) not in rows:
            fail(f"{run_dir}: summary_with_ci missing {scalar}")
    print(f"{run_dir.name}: n={n_paired} steps={n_steps} reduction={reduction}")
    return reduction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--min-pairs", type=int, default=50)
    ap.add_argument("--min-steps", type=int, default=2)
    ap.add_argument("--expected-steps", nargs="*", type=int, default=None)
    args = ap.parse_args()

    if len(args.run_dirs) < 2:
        fail("capacity sweep requires at least two run dirs")
    reductions = [
        validate_run(Path(run_dir), args.min_pairs, args.min_steps)
        for run_dir in args.run_dirs
    ]
    if len({int(r.get("n_steps_evaluated") or 0) for r in reductions}) < 2:
        fail("capacity sweep did not produce distinct n_steps_evaluated values")
    if args.expected_steps:
        observed = sorted(int(r.get("n_steps_evaluated") or 0) for r in reductions)
        expected = sorted(int(v) for v in args.expected_steps)
        if observed != expected:
            fail(f"capacity sweep steps mismatch: observed={observed} expected={expected}")
    print("CAPACITY SWEEP VALIDATION PASSED")


if __name__ == "__main__":
    main()
