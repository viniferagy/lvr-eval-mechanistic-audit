#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.stats.bootstrap import paired_bootstrap


METRIC_PATH = "metrics/lvr_latent_patch_answer_transfer_lvr_7b.json"


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def load_payload(run_dir: Path) -> dict:
    path = run_dir / METRIC_PATH
    if not path.is_file():
        fail(f"metric file missing: {path}")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload") or {}
    if envelope.get("metric_id") != "lvr_latent_patch_answer_transfer":
        fail(f"unexpected metric_id in {path}: {envelope.get('metric_id')}")
    return payload


def sample_key(sample: dict) -> str:
    return str(sample.get("paired_id") or sample.get("id"))


def sample_values(payload: dict) -> dict[str, dict[str, float]]:
    out = {}
    for sample in payload.get("samples") or []:
        key = sample_key(sample)
        steps = sample.get("step_results") or []
        step_transfers = [
            float(bool(step.get("answer_transferred")))
            for step in steps
            if step.get("answer_transferred") is not None
        ]
        if step_transfers:
            best = max(step_transfers)
            last = step_transfers[-1]
            auc = sum(step_transfers) / len(step_transfers)
        else:
            red = sample.get("reduction") or {}
            best = float(red.get("best_step_transfer_rate", red.get("latent_answer_transfer_rate", 0.0)))
            last = float(red.get("last_step_transfer_rate", red.get("latent_answer_transfer_rate", 0.0)))
            auc = float(red.get("step_transfer_auc", best))
        out[key] = {
            "best_step_transfer": best,
            "last_step_transfer": last,
            "step_auc": auc,
        }
    return out


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def label_for(run_dir: Path, payload: dict) -> str:
    steps = (payload.get("reduction") or {}).get("n_steps_evaluated")
    return f"{run_dir.name} (steps={steps})"


def paired_diff(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]], scalar: str, *, seed: int) -> dict:
    keys = sorted(set(a) & set(b))
    left = [a[key][scalar] for key in keys]
    right = [b[key][scalar] for key in keys]
    ci = paired_bootstrap(right, left, seed=seed, n_resamples=2000)
    return {
        "scalar": scalar,
        "n_paired": len(keys),
        "mean_diff": ci.mean if ci else None,
        "ci_low": ci.ci_low if ci else None,
        "ci_high": ci.ci_high if ci else None,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare W5 latent capacity sweep runs without hard-failing on trend.")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out-dir", default="docs/capacity_sweep")
    ap.add_argument("--seed", type=int, default=260523)
    args = ap.parse_args(argv)

    runs = []
    for text in args.run_dirs:
        run_dir = Path(text)
        payload = load_payload(run_dir)
        values = sample_values(payload)
        red = payload.get("reduction") or {}
        runs.append({
            "run_dir": str(run_dir),
            "label": label_for(run_dir, payload),
            "n_samples": len(values),
            "n_steps_evaluated": red.get("n_steps_evaluated"),
            "best_step_transfer_mean": mean([v["best_step_transfer"] for v in values.values()]),
            "last_step_transfer_mean": mean([v["last_step_transfer"] for v in values.values()]),
            "step_auc_mean": mean([v["step_auc"] for v in values.values()]),
            "_values": values,
        })
    comparisons = []
    for left, right in zip(runs, runs[1:]):
        for scalar in ("best_step_transfer", "last_step_transfer", "step_auc"):
            diff = paired_diff(left["_values"], right["_values"], scalar, seed=args.seed)
            diff["left"] = left["run_dir"]
            diff["right"] = right["run_dir"]
            comparisons.append(diff)

    public_runs = [{k: v for k, v in run.items() if k != "_values"} for run in runs]
    summary = {"runs": public_runs, "paired_differences": comparisons, "seed": args.seed}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "capacity_summary.json"
    md_path = out_dir / "capacity_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# W5 Capacity Sweep Comparison",
        "",
        "This report describes paired differences between adjacent latent feedback budgets. It does not require monotonicity.",
        "",
        "## Runs",
        "",
        "| run | n | steps | best | last | auc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in public_runs:
        lines.append(
            f"| `{run['run_dir']}` | {run['n_samples']} | {run['n_steps_evaluated']} | "
            f"{run['best_step_transfer_mean']} | {run['last_step_transfer_mean']} | {run['step_auc_mean']} |"
        )
    lines += ["", "## Adjacent Paired Differences", "", "| right-left | scalar | n | mean diff | 95% CI |", "|---|---|---:|---:|---|"]
    for diff in comparisons:
        lines.append(
            f"| `{Path(diff['right']).name}` - `{Path(diff['left']).name}` | {diff['scalar']} | "
            f"{diff['n_paired']} | {diff['mean_diff']} | [{diff['ci_low']}, {diff['ci_high']}] |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
