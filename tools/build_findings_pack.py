#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_findings_gate import (
    LATENT_METRIC,
    LVR_MODEL,
    MAZE_PRIMARY,
    SPD_MODELS,
    SPD_PRIMARY,
    fail,
    require_metric_payload,
    validate_latent_run,
    validate_scale_run,
    validate_step_run,
)


def reductions_for(run_dir: Path, required: dict[str, str], models: tuple[str, ...]) -> list[dict]:
    rows = []
    for metric_id, scalar in required.items():
        row = {"metric_id": metric_id, "primary_scalar": scalar}
        for model in models:
            payload = require_metric_payload(run_dir, metric_id, model)
            row[model] = (payload.get("reduction") or {}).get(scalar)
        rows.append(row)
    return rows


def md_table(rows: list[dict], models: tuple[str, ...]) -> list[str]:
    lines = ["| metric | scalar | " + " | ".join(models) + " |"]
    lines.append("|---|---|" + "|".join(["---:"] * len(models)) + "|")
    for row in rows:
        values = []
        for model in models:
            value = row.get(model)
            values.append("NA" if value is None else f"{float(value):.6g}")
        lines.append(f"| `{row['metric_id']}` | `{row['primary_scalar']}` | " + " | ".join(values) + " |")
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build Findings-level evidence summary.")
    ap.add_argument("--out", default="docs/findings_evidence_pack.md")
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

    w3 = validate_latent_run(Path(args.w3), min_pairs=args.min_latent_pairs)
    w4 = validate_step_run(Path(args.w4), min_pairs=args.min_latent_pairs, min_steps=args.min_steps)
    w6 = validate_latent_run(Path(args.w6), min_pairs=args.min_latent_pairs)
    validate_scale_run(
        Path(args.spd),
        required=SPD_PRIMARY,
        models=SPD_MODELS,
        min_samples=args.min_spd_samples,
        paired_metrics={"bf_patch_answer_transfer", "bf_swap_latent_replacement"},
    )
    maze_status = "pass"
    if Path(args.maze).exists():
        validate_scale_run(Path(args.maze), required=MAZE_PRIMARY, models=SPD_MODELS, min_samples=args.min_maze_samples)
        maze_rows = reductions_for(Path(args.maze), MAZE_PRIMARY, SPD_MODELS)
    elif args.allow_missing_maze:
        maze_status = "blocked_missing_artifact"
        maze_rows = []
    else:
        fail(f"missing Maze Findings run: {args.maze}")

    spd_rows = reductions_for(Path(args.spd), SPD_PRIMARY, SPD_MODELS)
    lines = [
        "# Findings Evidence Pack",
        "",
        "This pack is generated from local run artifacts. It is intended as the Findings-level review gate, not a Main/Spotlight claim-finalizing package.",
        "",
        "## Gate Status",
        "",
        "| component | status | run dir |",
        "|---|---|---|",
        f"| W3 true latent patch | pass | `{args.w3}` |",
        f"| W4 latent step sweep | pass | `{args.w4}` |",
        f"| W6 best-step replication | pass | `{args.w6}` |",
        f"| W7 SPD regression matrix | pass | `{args.spd}` |",
        f"| W9 Maze Findings subset | {maze_status} | `{args.maze}` |",
        "",
        "## True LVR Hidden-Feedback Gates",
        "",
        "| gate | key reductions |",
        "|---|---|",
        f"| W3 last-step | `{json.dumps({k: w3.get(k) for k in ('latent_answer_transfer_rate', 'n_paired', 'n_success')}, sort_keys=True)}` |",
        f"| W4 step sweep | `{json.dumps({k: w4.get(k) for k in ('best_step_index', 'best_step_transfer_rate', 'step_transfer_auc', 'n_steps_evaluated')}, sort_keys=True)}` |",
        f"| W6 best-step | `{json.dumps({k: w6.get(k) for k in ('latent_answer_transfer_rate', 'best_step_index', 'n_paired', 'n_success')}, sort_keys=True)}` |",
        "",
        "## SPD-Faith Findings Matrix",
        "",
        *md_table(spd_rows, SPD_MODELS),
        "",
        "## Maze Findings Matrix",
        "",
    ]
    if maze_rows:
        lines.extend(md_table(maze_rows, SPD_MODELS))
    else:
        lines.extend([
            "Maze artifacts are not present locally. The Findings gate remains blocked until a real Maze run is produced and validated.",
        ])
    lines += [
        "",
        "## Boundary Statement",
        "",
        "Findings-level claims should be limited to a reproducible causal audit toolkit, SPD-Faith paired evidence, and true LVR hidden-feedback patch gates on constrained answers. Main/Spotlight claims still require broader tasks, larger n, additional LVR paradigms, and layer/position localization.",
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    if maze_status == "pass":
        print("FINDINGS EVIDENCE PACK VALIDATION PASSED")
    else:
        print("FINDINGS EVIDENCE PACK BLOCKED: missing Maze artifact")


if __name__ == "__main__":
    main()
