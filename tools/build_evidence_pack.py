#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_ID = "lvr_latent_patch_answer_transfer"
MODEL = "lvr_7b"
W7_MODELS = ("qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b")
W7_PRIMARY_SCALARS = {
    "pf_a_corruption_selectivity": "selectivity",
    "pf_b_patch_alignment": "native_alignment",
    "bf_patch_answer_transfer": "logprob_margin_shift",
    "bf_swap_latent_replacement": "swap_margin_shift",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_delta",
}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def require_file(path: Path) -> Path:
    if not path.is_file():
        fail(f"required file missing: {path}")
    return path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_required_json(path: Path):
    return load_json(require_file(path))


def metric_payload(run_dir: Path, metric_id: str, model: str = MODEL, *, allow_missing: bool = False) -> dict:
    path = run_dir / "metrics" / f"{metric_id}_{model}.json"
    if not path.is_file():
        if allow_missing:
            return {}
        fail(f"metric file missing: {path}")
    envelope = load_json(path)
    if envelope.get("metric_id") != metric_id:
        fail(f"{path} metric_id mismatch: {envelope.get('metric_id')} != {metric_id}")
    if envelope.get("model") != model:
        fail(f"{path} model mismatch: {envelope.get('model')} != {model}")
    payload = envelope.get("payload") or {}
    if not isinstance(payload, dict):
        fail(f"{path} payload is not an object")
    return payload


def metric_reduction(run_dir: Path, metric_id: str, model: str = MODEL, *, allow_missing: bool = False) -> dict:
    payload = metric_payload(run_dir, metric_id, model, allow_missing=allow_missing)
    if not payload and allow_missing:
        return {}
    reduction = payload.get("reduction")
    if not isinstance(reduction, dict) or not reduction:
        fail(f"metric reduction missing: {run_dir}/metrics/{metric_id}_{model}.json")
    for key in ("n_paired", "n_success"):
        value = reduction.get(key)
        if value is None:
            fail(f"metric reduction missing {key}: {run_dir}")
        if int(value) <= 0:
            fail(f"metric reduction {key} must be > 0 in {run_dir}: {value}")
    return reduction


def sanity_status(run_dir: Path, *, allow_missing: bool = False) -> str | None:
    path = run_dir / "sanity" / "summary_sanity.json"
    if not path.is_file():
        if allow_missing:
            return None
        fail(f"sanity summary missing: {path}")
    status = (load_json(path) or {}).get("overall_status")
    if status != "pass" and not allow_missing:
        fail(f"sanity did not pass in {run_dir}: {status}")
    return status


def w7_ci_rows(run_dir: Path, *, allow_missing: bool = False) -> list:
    path = run_dir / "summary_with_ci.json"
    if not path.is_file():
        if allow_missing:
            return []
        fail(f"W7 CI summary missing: {path}")
    rows = load_json(path)
    if not isinstance(rows, list) or not rows:
        fail(f"W7 CI summary is empty or invalid: {path}")
    return rows


def validate_w7_artifacts(run_dir: Path, ci_rows: list, *, allow_missing: bool = False) -> None:
    if allow_missing:
        return
    for metric_id in W7_PRIMARY_SCALARS:
        for model in W7_MODELS:
            path = run_dir / "metrics" / f"{metric_id}_{model}.json"
            if not path.is_file():
                fail(f"W7 metric file missing: {path}")
            envelope = load_json(path)
            if envelope.get("metric_id") != metric_id or envelope.get("model") != model:
                fail(f"W7 metric envelope mismatch: {path}")
            payload = envelope.get("payload") or {}
            reduction = payload.get("reduction") or {}
            scalar = W7_PRIMARY_SCALARS[metric_id]
            if reduction.get(scalar) is None:
                fail(f"W7 primary scalar missing: {metric_id}/{model}/{scalar}")
    ci_keys = {
        (row.get("metric_id"), row.get("model"), row.get("scalar"))
        for row in ci_rows
        if int(row.get("n") or 0) > 0
    }
    missing = []
    for metric_id, scalar in W7_PRIMARY_SCALARS.items():
        for model in W7_MODELS:
            if (metric_id, model, scalar) not in ci_keys:
                missing.append(f"{metric_id}/{model}/{scalar}")
    if missing:
        fail(f"W7 summary_with_ci missing primary row(s): {missing[:8]}")


def validate_capacity_runs(run_dirs: list[Path], reductions: list[dict], *, allow_missing: bool = False) -> None:
    if allow_missing:
        return
    if len(run_dirs) < 2:
        fail("W5 capacity sweep requires at least 2 run dirs")
    step_counts = {
        int(red.get("n_steps_evaluated") or 0)
        for red in reductions
    }
    if len(step_counts) < 2:
        fail(f"W5 capacity sweep needs at least two distinct n_steps_evaluated values: {sorted(step_counts)}")
    if min(step_counts) <= 0:
        fail(f"W5 capacity sweep has invalid n_steps_evaluated values: {sorted(step_counts)}")


def require_latent_run(run_dir: Path, *, allow_missing: bool = False) -> dict:
    reduction = metric_reduction(run_dir, METRIC_ID, MODEL, allow_missing=allow_missing)
    if allow_missing and not reduction:
        return {}
    sanity_status(run_dir, allow_missing=allow_missing)
    return reduction


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evidence_pack_w5_w8.md")
    ap.add_argument("--w3", default="runs/w3_lvr_latent_patch_n50")
    ap.add_argument("--w4", default="runs/w4_lvr_latent_stepsweep_n50_s8")
    ap.add_argument("--w5", nargs="*", default=[
        "runs/w5_lvr_capacity_s2_n50",
        "runs/w5_lvr_capacity_s4_n50",
        "runs/w5_lvr_capacity_s8_n50",
        "runs/w5_lvr_capacity_s16_n50",
    ])
    ap.add_argument("--w6", default="runs/w6_lvr_beststep4_s8_n50")
    ap.add_argument("--w7", default="runs/w7_spd_scale_m0_m1_m2_n200")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args(argv)

    allow_missing = bool(args.allow_missing)
    w5_dirs = [Path(item) for item in args.w5]
    w5_reductions = [require_latent_run(run_dir, allow_missing=allow_missing) for run_dir in w5_dirs]
    if w5_dirs:
        validate_capacity_runs(w5_dirs, w5_reductions, allow_missing=allow_missing)

    w7_rows = []
    if args.w7:
        run_dir = Path(args.w7)
        sanity_status(run_dir, allow_missing=allow_missing)
        w7_rows = w7_ci_rows(run_dir, allow_missing=allow_missing)
        validate_w7_artifacts(run_dir, w7_rows, allow_missing=allow_missing)

    lines = [
        "# W5-W8 Evidence Pack",
        "",
        "This file is generated from local run artifacts. It summarizes gate-level evidence, not final paper-scale claims.",
        "",
        "## Latent Intervention Gates",
        "",
        "| gate | run_dir | sanity | key reductions |",
        "|---|---|---|---|",
    ]
    for gate, run_dir_text in [("W3 last-step", args.w3), ("W4 step sweep", args.w4), ("W6 best-step", args.w6)]:
        if not run_dir_text:
            continue
        run_dir = Path(run_dir_text)
        reduction = require_latent_run(run_dir, allow_missing=allow_missing)
        keep = {
            key: reduction.get(key)
            for key in (
                "n_paired",
                "n_success",
                "latent_answer_transfer_rate",
                "best_step_transfer_rate",
                "best_step_index",
                "step_transfer_auc",
                "last_step_transfer_rate",
                "n_steps_evaluated",
            )
            if key in reduction
        }
        lines.append(
            f"| {gate} | `{run_dir}` | {sanity_status(run_dir, allow_missing=allow_missing)} | "
            f"`{json.dumps(keep, sort_keys=True)}` |"
        )

    if args.w5:
        lines += ["", "## W5 Capacity Sweep", "", "| run_dir | sanity | key reductions |", "|---|---|---|"]
        for run_dir, reduction in zip(w5_dirs, w5_reductions):
            keep = {
                key: reduction.get(key)
                for key in (
                    "n_paired",
                    "n_success",
                    "best_step_transfer_rate",
                    "best_step_index",
                    "step_transfer_auc",
                    "last_step_transfer_rate",
                    "n_steps_evaluated",
                )
                if key in reduction
            }
            lines.append(
                f"| `{run_dir}` | {sanity_status(run_dir, allow_missing=allow_missing)} | "
                f"`{json.dumps(keep, sort_keys=True)}` |"
            )

    if args.w7:
        run_dir = Path(args.w7)
        lines += [
            "",
            "## W7 SPD Regression Scale-Up",
            "",
            f"- Run dir: `{run_dir}`",
            f"- Sanity: `{sanity_status(run_dir, allow_missing=allow_missing)}`",
            f"- CI rows: `{len(w7_rows)}`",
        ]

    lines += [
        "",
        "## Boundary Statement",
        "",
        "W3-W6 are true inference-time LVR hidden-feedback intervention gates on SPD-Faith constrained answers. W7 remains query-span regression evidence across Qwen/LVR weights. W5 capacity validation does not establish monotonic capacity scaling. Broader-task, larger-n, and layer-level localization remain future work.",
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    if not allow_missing:
        print("EVIDENCE PACK VALIDATION PASSED")


if __name__ == "__main__":
    main()
