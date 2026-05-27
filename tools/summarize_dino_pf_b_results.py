#!/usr/bin/env python
"""Summarize W21 PF-B DINO sensitivity pilot results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_LABELS = {
    "qwen2_5_vl_3b": "Qwen2.5-VL-3B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "lvr_7b": "LVR-7B",
}
MODEL_ORDER = ["qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b"]
TASK_LABELS = {
    "spd_faith": "SPD-Faith n=100",
    "vstar": "V*Bench n=191",
}
SCALARS = [
    "native_alignment",
    "dino_region_selectivity",
    "dino_relevant_alignment",
    "dino_irrelevant_alignment",
    "dino_random_alignment",
]


def fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        row for row in rows
        if row.get("metric_id") == "pf_b_patch_alignment"
        and row.get("scalar") in SCALARS
    ]


def by_key(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {
        (str(row.get("task")), str(row.get("model")), str(row.get("scalar"))): row
        for row in rows
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "model", "scalar", "n", "mean", "ci_low", "ci_high"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (str(r.get("task")), str(r.get("scalar")), MODEL_ORDER.index(str(r.get("model"))) if str(r.get("model")) in MODEL_ORDER else 99)):
            writer.writerow({field: row.get(field) for field in fields})


def plot_native_and_dino(rows: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = by_key(rows)
    tasks = sorted({str(row.get("task")) for row in rows})
    made: list[Path] = []

    for scalar, title, ylabel in [
        ("native_alignment", "PF-B Native Alignment", "native alignment"),
        ("dino_region_selectivity", "DINO Region Selectivity", "relevant delta - mean(control deltas)"),
    ]:
        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            xs = []
            means = []
            lo = []
            hi = []
            labels = []
            for i, model in enumerate(MODEL_ORDER):
                row = index.get((task, model, scalar))
                if row is None:
                    continue
                xs.append(i)
                mean = float(row["mean"])
                means.append(mean)
                lo.append(mean - float(row["ci_low"]))
                hi.append(float(row["ci_high"]) - mean)
                labels.append(MODEL_LABELS.get(model, model))
            ax.errorbar(xs, means, yerr=[lo, hi], marker="o", capsize=4, lw=1.8)
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title(TASK_LABELS.get(task, task))
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(title)
        fig.tight_layout()
        path = out_dir / f"w21_{scalar}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        made.append(path)
    return made


def markdown_table(rows: list[dict], scalar: str) -> str:
    index = by_key(rows)
    tasks = sorted({str(row.get("task")) for row in rows})
    lines = ["| task | model | n | mean | 95% CI |", "|---|---:|---:|---:|---:|"]
    for task in tasks:
        for model in MODEL_ORDER:
            row = index.get((task, model, scalar))
            if row is None:
                continue
            lines.append(
                f"| {TASK_LABELS.get(task, task)} | {MODEL_LABELS.get(model, model)} | "
                f"{row.get('n')} | {fmt(row.get('mean'))} | "
                f"[{fmt(row.get('ci_low'))}, {fmt(row.get('ci_high'))}] |"
            )
    return "\n".join(lines)


def write_report(rows: list[dict], figures: list[Path], csv_path: Path, out_path: Path) -> None:
    text = f"""# W21 PF-B DINO Sensitivity Pilot

Date: 2026-05-26

This pilot adds an optional DINOv2 image-backbone sensitivity readout to PF-B. Native PF-B remains the model-facing primary scalar. DINO is an external visual-backbone check over the same clean/relevant/irrelevant/random masked images, so its DINO rows are expected to be identical across model wrappers for a fixed dataset.

## Native PF-B

{markdown_table(rows, "native_alignment")}

Interpretation: native PF-B preserves the earlier pattern. LVR-7B is highest on both SPD-Faith and V*Bench. This is the model-dependent alignment claim.

## DINO Region Selectivity

{markdown_table(rows, "dino_region_selectivity")}

Interpretation: DINO region selectivity is near zero and its confidence intervals cross zero on both datasets. The external image backbone does not show a strong relevant-region perturbation advantage under the current mask construction. This should be reported as a sensitivity/null check, not as support for a stronger bbox-localization claim.

## Artifacts

- CSV: `{csv_path}`
"""
    for fig in figures:
        text += f"- Figure: `{fig}`\n"
    text += """
## Next Decision

Priority 1/2 pilot succeeded technically: DINO is implemented, cached, validated by sanity, and produces CI rows. The signal is mixed scientifically: native PF-B supports LVR > Qwen, while DINO selectivity is effectively null. Before expanding to SPD n=1000, prewarm the DINO cache or run DINO on GPU in a separate low-memory phase; otherwise the CPU DINO backend becomes the runtime bottleneck.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="merged run directory")
    parser.add_argument("--fig-dir", type=Path, default=Path("docs/figures/w21_dino_pf_b"))
    parser.add_argument("--report", type=Path, default=Path("docs/validation_report_w21_dino_pf_b.md"))
    args = parser.parse_args()

    rows = load_rows(args.run / "summary_with_ci.json")
    csv_path = args.run / "w21_dino_pf_b_results.csv"
    write_csv(rows, csv_path)
    figures = plot_native_and_dino(rows, args.fig_dir)
    write_report(rows, figures, csv_path, args.report)
    print(f"WROTE {csv_path}")
    print(f"WROTE {args.report}")
    for fig in figures:
        print(f"WROTE {fig}")


if __name__ == "__main__":
    main()
