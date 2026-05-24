#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


LATENT_METRIC = "lvr_latent_patch_answer_transfer"
LVR_MODEL = "lvr_7b"
SPD_MODELS = ("qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b")
SPD_PRIMARY = {
    "pf_a_corruption_selectivity": "selectivity",
    "pf_b_patch_alignment": "native_alignment",
    "bf_patch_answer_transfer": "logprob_margin_shift",
    "bf_swap_latent_replacement": "swap_margin_shift",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_delta",
}
MAZE_PRIMARY = {
    "pf_a_corruption_selectivity": "selectivity",
    "pf_b_patch_alignment": "native_alignment",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_delta",
}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def load_json(path: Path):
    if not path.is_file():
        fail(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_sanity_pass(run_dir: Path) -> None:
    sanity = load_json(run_dir / "sanity" / "summary_sanity.json")
    if sanity.get("overall_status") != "pass":
        fail(f"{run_dir}: sanity overall_status={sanity.get('overall_status')}")


def require_prereg(run_dir: Path) -> None:
    lock = load_json(run_dir / "prereg.lock.json")
    if not lock.get("sha256"):
        fail(f"{run_dir}: prereg lock missing sha256")


def require_ci_rows(run_dir: Path, required: dict[str, str], models: tuple[str, ...]) -> int:
    rows = load_json(run_dir / "summary_with_ci.json")
    keys = {
        (row.get("metric_id"), row.get("model"), row.get("scalar"))
        for row in rows
        if int(row.get("n") or 0) > 0
    }
    missing = []
    for metric_id, scalar in required.items():
        for model in models:
            if (metric_id, model, scalar) not in keys:
                missing.append(f"{metric_id}/{model}/{scalar}")
    if missing:
        fail(f"{run_dir}: missing CI rows: {missing[:8]}")
    return len(rows)


def require_metric_payload(run_dir: Path, metric_id: str, model: str) -> dict:
    path = run_dir / "metrics" / f"{metric_id}_{model}.json"
    envelope = load_json(path)
    if envelope.get("metric_id") != metric_id or envelope.get("model") != model:
        fail(f"{path}: envelope mismatch")
    payload = envelope.get("payload") or {}
    if not isinstance(payload, dict):
        fail(f"{path}: payload is not an object")
    return payload


def validate_latent_run(run_dir: Path, *, min_pairs: int) -> dict:
    require_prereg(run_dir)
    require_sanity_pass(run_dir)
    payload = require_metric_payload(run_dir, LATENT_METRIC, LVR_MODEL)
    reduction = payload.get("reduction") or {}
    n_paired = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
    n_success = int(reduction.get("n_success") or 0)
    n_patch_applied = int(reduction.get("n_patch_applied") or 0)
    n_captured = int(reduction.get("n_with_captured_state") or 0)
    if min(n_paired, n_success, n_patch_applied, n_captured) < min_pairs:
        fail(
            f"{run_dir}: latent gate below min_pairs={min_pairs}: "
            f"n_paired={n_paired} n_success={n_success} n_patch_applied={n_patch_applied} "
            f"n_with_captured_state={n_captured}"
        )
    if reduction.get("latent_answer_transfer_rate") is None:
        fail(f"{run_dir}: missing latent_answer_transfer_rate")
    require_ci_rows(run_dir, {LATENT_METRIC: "latent_answer_transfer_rate"}, (LVR_MODEL,))
    return reduction


def validate_step_run(run_dir: Path, *, min_pairs: int, min_steps: int) -> dict:
    require_prereg(run_dir)
    require_sanity_pass(run_dir)
    payload = require_metric_payload(run_dir, LATENT_METRIC, LVR_MODEL)
    reduction = payload.get("reduction") or {}
    n_paired = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
    n_success = int(reduction.get("n_success") or 0)
    n_steps = int(reduction.get("n_steps_evaluated") or 0)
    if n_paired < min_pairs or n_success < min_pairs:
        fail(f"{run_dir}: step run below min_pairs={min_pairs}: n_paired={n_paired} n_success={n_success}")
    if n_steps < min_steps:
        fail(f"{run_dir}: n_steps_evaluated too small: {n_steps}")
    for scalar in ("best_step_transfer_rate", "step_transfer_auc"):
        if reduction.get(scalar) is None:
            fail(f"{run_dir}: missing {scalar}")
    require_ci_rows(
        run_dir,
        {LATENT_METRIC: "best_step_transfer_rate"},
        (LVR_MODEL,),
    )
    return reduction


def validate_scale_run(
    run_dir: Path,
    *,
    required: dict[str, str],
    models: tuple[str, ...],
    min_samples: int,
    paired_metrics: set[str] | None = None,
) -> dict:
    paired_metrics = paired_metrics or set()
    require_prereg(run_dir)
    require_sanity_pass(run_dir)
    for metric_id, scalar in required.items():
        for model in models:
            payload = require_metric_payload(run_dir, metric_id, model)
            reduction = payload.get("reduction") or {}
            if reduction.get(scalar) is None:
                fail(f"{run_dir}: missing primary scalar {metric_id}/{model}/{scalar}")
            n = int(
                reduction.get("n")
                or payload.get("n_paired")
                or reduction.get("n_paired")
                or len(payload.get("samples") or [])
                or 0
            )
            if metric_id in paired_metrics:
                n = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
            if n < min_samples:
                fail(f"{run_dir}: {metric_id}/{model} n too small: {n} < {min_samples}")
    n_ci = require_ci_rows(run_dir, required, models)
    return {"n_ci_rows": n_ci}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Validate the Findings-level gate across latent, SPD, and Maze artifacts.")
    ap.add_argument("--w3", default="runs/w3_lvr_latent_patch_n50")
    ap.add_argument("--w4", default="runs/w4_lvr_latent_stepsweep_n50_s8")
    ap.add_argument("--w6", default="runs/w6_lvr_beststep4_s8_n50")
    ap.add_argument("--spd", default="runs/w7_spd_scale_m0_m1_m2_n200")
    ap.add_argument("--maze", default="runs/w9_maze_findings_m0_m1_m2_n200_bbox")
    ap.add_argument("--min-latent-pairs", type=int, default=50)
    ap.add_argument("--min-spd-samples", type=int, default=200)
    ap.add_argument("--min-maze-samples", type=int, default=200)
    ap.add_argument("--min-steps", type=int, default=2)
    ap.add_argument("--allow-missing-maze", action="store_true")
    args = ap.parse_args(argv)

    summary = {
        "w3": validate_latent_run(Path(args.w3), min_pairs=args.min_latent_pairs),
        "w4": validate_step_run(Path(args.w4), min_pairs=args.min_latent_pairs, min_steps=args.min_steps),
        "w6": validate_latent_run(Path(args.w6), min_pairs=args.min_latent_pairs),
        "spd": validate_scale_run(
            Path(args.spd),
            required=SPD_PRIMARY,
            models=SPD_MODELS,
            min_samples=args.min_spd_samples,
            paired_metrics={"bf_patch_answer_transfer", "bf_swap_latent_replacement"},
        ),
    }
    maze_dir = Path(args.maze)
    if maze_dir.exists():
        summary["maze"] = validate_scale_run(
            maze_dir,
            required=MAZE_PRIMARY,
            models=SPD_MODELS,
            min_samples=args.min_maze_samples,
        )
    elif args.allow_missing_maze:
        summary["maze"] = {"status": "missing_allowed", "run_dir": str(maze_dir)}
    else:
        fail(f"missing Maze Findings run: {maze_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("FINDINGS GATE VALIDATION PASSED")


if __name__ == "__main__":
    main()
