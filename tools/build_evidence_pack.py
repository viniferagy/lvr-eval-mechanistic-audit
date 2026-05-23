#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_reduction(run_dir: Path, metric_id: str, model: str = "lvr_7b") -> dict:
    path = run_dir / "metrics" / f"{metric_id}_{model}.json"
    if not path.is_file():
        return {}
    payload = (load_json(path).get("payload") or {})
    return payload.get("reduction") or {}


def sanity_status(run_dir: Path) -> str | None:
    path = run_dir / "sanity" / "summary_sanity.json"
    if not path.is_file():
        return None
    return (load_json(path) or {}).get("overall_status")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evidence_pack_w5_w8.md")
    ap.add_argument("--w3", default="runs/w3_lvr_latent_patch_n50")
    ap.add_argument("--w4", default="runs/w4_lvr_latent_stepsweep_n50_s8")
    ap.add_argument("--w5", nargs="*", default=[])
    ap.add_argument("--w6", default="")
    ap.add_argument("--w7", default="")
    args = ap.parse_args()

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
        reduction = metric_reduction(run_dir, "lvr_latent_patch_answer_transfer")
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
        lines.append(f"| {gate} | `{run_dir}` | {sanity_status(run_dir)} | `{json.dumps(keep, sort_keys=True)}` |")

    if args.w5:
        lines += ["", "## W5 Capacity Sweep", "", "| run_dir | sanity | key reductions |", "|---|---|---|"]
        for run_dir_text in args.w5:
            run_dir = Path(run_dir_text)
            reduction = metric_reduction(run_dir, "lvr_latent_patch_answer_transfer")
            keep = {
                key: reduction.get(key)
                for key in (
                    "n_paired",
                    "best_step_transfer_rate",
                    "best_step_index",
                    "step_transfer_auc",
                    "last_step_transfer_rate",
                    "n_steps_evaluated",
                )
                if key in reduction
            }
            lines.append(f"| `{run_dir}` | {sanity_status(run_dir)} | `{json.dumps(keep, sort_keys=True)}` |")

    if args.w7:
        run_dir = Path(args.w7)
        ci_path = run_dir / "summary_with_ci.json"
        n_ci = len(load_json(ci_path)) if ci_path.is_file() else 0
        lines += [
            "",
            "## W7 SPD Regression Scale-Up",
            "",
            f"- Run dir: `{run_dir}`",
            f"- Sanity: `{sanity_status(run_dir)}`",
            f"- CI rows: `{n_ci}`",
        ]

    lines += [
        "",
        "## Boundary Statement",
        "",
        "W3-W6 are true inference-time LVR hidden-feedback intervention gates on SPD-Faith constrained answers. W7 remains query-span regression evidence across Qwen/LVR weights. Broader-task and layer-level localization remain future work.",
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
