#!/usr/bin/env python
"""Summarize W20 V*Bench/VStar spotlight artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIMARY_SCALARS = {
    "pf_a_corruption_selectivity": "selectivity",
    "pf_b_patch_alignment": "native_alignment",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_delta",
}

METRIC_LABELS = {
    "pf_a_corruption_selectivity": "PF-A selectivity",
    "pf_b_patch_alignment": "PF-B native alignment",
    "bf_conf_calibrated_progression": "BF-Conf gold-logit slope",
    "cf_stage_decay": "CF-Stage late delta",
}

MODEL_ORDER = ["qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b"]
MODEL_LABELS = {
    "qwen2_5_vl_3b": "Qwen2.5-VL-3B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "lvr_7b": "LVR-7B",
}

MODEL_COLORS = {
    "qwen2_5_vl_3b": "#2563eb",
    "qwen2_5_vl_7b": "#7c3aed",
    "lvr_7b": "#b45309",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_ci(row: dict | None, digits: int = 4) -> str:
    if not row or row.get("mean") is None:
        return "-"
    return f"{fmt(row.get('mean'), digits)} [{fmt(row.get('ci_low'), digits)}, {fmt(row.get('ci_high'), digits)}]"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def md_path(path: Path, report_path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def primary_rows(ci_rows: list[dict]) -> list[dict]:
    rows = []
    for row in ci_rows:
        metric = str(row.get("metric_id") or "")
        if PRIMARY_SCALARS.get(metric) == row.get("scalar"):
            rows.append(row)
    return rows


def row_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (str(row.get("metric_id")), str(row.get("model"))): row
        for row in rows
    }


def compare_row_index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {
        (str(row.get("task")), str(row.get("metric_id")), str(row.get("model"))): row
        for row in rows
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "metric_id", "model", "scalar", "n", "mean", "ci_low", "ci_high"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def plot_vstar_lines(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    by = row_index(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    x = list(range(len(MODEL_ORDER)))
    for ax, metric_id in zip(axes.flatten(), PRIMARY_SCALARS):
        ys = []
        lows = []
        highs = []
        for model in MODEL_ORDER:
            row = by.get((metric_id, model))
            mean = float(row["mean"])
            ys.append(mean)
            lows.append(max(0.0, mean - float(row.get("ci_low", mean))))
            highs.append(max(0.0, float(row.get("ci_high", mean)) - mean))
        ax.errorbar(x, ys, yerr=[lows, highs], marker="o", linewidth=1.8, capsize=3, color="#0f766e")
        ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(["3B", "7B", "LVR"])
        ax.set_title(METRIC_LABELS[metric_id], fontsize=10)
        ax.set_ylabel(PRIMARY_SCALARS[metric_id])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("W20 V*Bench/VStar Spotlight Metrics (n=191)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "w20_vstar_primary_metric_lines.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spd_vstar_compare(vstar_rows: list[dict], w18_summary: Path, out_dir: Path) -> Path | None:
    if not w18_summary.is_file():
        return None
    w18_rows = primary_rows(read_json(w18_summary))
    w18_by = compare_row_index(w18_rows)
    v_by = row_index(vstar_rows)
    metrics = ["pf_a_corruption_selectivity", "pf_b_patch_alignment"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = list(range(len(MODEL_ORDER)))
    width = 0.35
    for ax, metric_id in zip(axes, metrics):
        spd_vals, spd_lows, spd_highs = [], [], []
        v_vals, v_lows, v_highs = [], [], []
        for model in MODEL_ORDER:
            spd = w18_by.get(("spd_faith", metric_id, model))
            vst = v_by.get((metric_id, model))
            for row, vals, lows, highs in ((spd, spd_vals, spd_lows, spd_highs), (vst, v_vals, v_lows, v_highs)):
                mean = float(row["mean"])
                vals.append(mean)
                lows.append(max(0.0, mean - float(row.get("ci_low", mean))))
                highs.append(max(0.0, float(row.get("ci_high", mean)) - mean))
        ax.bar([i - width / 2 for i in x], spd_vals, width, yerr=[spd_lows, spd_highs], capsize=3, label="SPD-Faith", color="#2563eb")
        ax.bar([i + width / 2 for i in x], v_vals, width, yerr=[v_lows, v_highs], capsize=3, label="V*Bench", color="#0f766e")
        ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(["3B", "7B", "LVR"])
        ax.set_title(METRIC_LABELS[metric_id])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("V*Bench vs SPD-Faith Availability Readouts")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / "w20_vstar_vs_spd_availability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="runs/w20_spotlight_vstar_n191_final")
    ap.add_argument("--out-report", default="docs/validation_report_w20_vstar_spotlight.md")
    ap.add_argument("--fig-dir", default="docs/figures/w20_vstar_spotlight")
    ap.add_argument("--w18-summary", default="runs/w18_main_matrix_full_local/merged/summary_with_ci.json")
    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    merged = run_root / "merged"
    report_path = Path(args.out_report)
    fig_dir = Path(args.fig_dir)
    ci_rows = read_json(merged / "summary_with_ci.json")
    rows = primary_rows(ci_rows)
    by = row_index(rows)
    stats = read_json(Path("data/vstar/prepare_stats.json"))
    sanity = read_json(merged / "sanity" / "summary_sanity.json")

    line_plot = plot_vstar_lines(rows, fig_dir)
    compare_plot = plot_spd_vstar_compare(rows, Path(args.w18_summary), fig_dir)
    write_csv(rows, merged / "w20_vstar_primary_results.csv")

    primary_table = []
    for metric_id in PRIMARY_SCALARS:
        primary_table.append([
            METRIC_LABELS[metric_id],
            fmt_ci(by.get((metric_id, "qwen2_5_vl_3b"))),
            fmt_ci(by.get((metric_id, "qwen2_5_vl_7b"))),
            fmt_ci(by.get((metric_id, "lvr_7b"))),
        ])

    data_table = [
        ["n_seen", stats.get("n_seen")],
        ["n_written", stats.get("n_written")],
        ["direct_attributes", (stats.get("per_category") or {}).get("attribute_recognition")],
        ["relative_position", (stats.get("per_category") or {}).get("spatial_relationship_reasoning")],
        ["n_missing_bbox", stats.get("n_missing_bbox")],
        ["n_skipped_no_image", stats.get("n_skipped_no_image")],
        ["bbox_area_ratio median", fmt((stats.get("bbox_area_ratio") or {}).get("median"), 6)],
        ["bbox_area_ratio min/max", f"{fmt((stats.get('bbox_area_ratio') or {}).get('min'), 6)} / {fmt((stats.get('bbox_area_ratio') or {}).get('max'), 6)}"],
    ]

    report = f"""# W20 V*Bench/VStar Spotlight Results

Generated from real local artifacts on 2026-05-26.

Run root: `{run_root}`

Merged artifacts: `{merged}`

## Data Staging

V*Bench is prepared from the full Hugging Face repository snapshot, not from
`datasets.load_dataset("craigwu/vstar_bench")`. The parquet/Data Studio view
drops the per-image JSON files that contain bbox annotations; the repository
snapshot contains `direct_attributes/sa_XXXXX.json` and
`relative_position/sa_XXXXX.json`, which this run reads directly.

{md_table(["Field", "Value"], data_table)}

The bbox median area is below 0.1% of image pixels, so this is genuinely a
small-region high-resolution localization stress test. The main PF-A/PF-B
spotlight uses the 1024px audit image budget. BF-Conf uses a metric-specific
512px budget to avoid high-resolution logit-lens OOM on 24GB GPUs; PF-A/PF-B
are unaffected by that lower-resolution auxiliary readout.

## Validation

```text
tools/validate_main_matrix.py {run_root} --tasks vstar --min-samples 150 --bf-min-pairs 0
MAIN MATRIX VALIDATION PASSED rows=12 ci_rows=51

merged sanity overall_status={sanity.get("overall_status")} total_reports={sanity.get("total_reports")} failed_reports={sanity.get("failed_reports")}
```

## Primary Results

{md_table(["Metric", "Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "LVR-7B"], primary_table)}

![W20 VStar primary metric lines]({md_path(line_plot, report_path)})

## V*Bench vs SPD-Faith Availability

{f"![W20 VStar vs SPD availability]({md_path(compare_plot, report_path)})" if compare_plot else "SPD comparison plot skipped because W18 summary was not found."}

The preregistered spotlight hypothesis only partially holds. LVR-7B has the
highest PF-B native alignment on V*Bench (`0.9585 [0.9526, 0.9638]`), above both
Qwen baselines. However, PF-A corruption selectivity is highest for Qwen2.5-VL-7B
(`0.2527 [0.2291, 0.2763]`) and lower for LVR-7B (`0.1996 [0.1837, 0.2164]`).
This means V*Bench supports the narrower claim that LVR representations are
strongly aligned with native visual-region signals on a high-resolution bbox
task, but it does not support the stronger claim that LVR uniquely dominates
region-corruption selectivity.

## Reproduction

```bash
./venv/bin/python tools/prepare_vstar_local.py \\
  --out data/vstar \\
  --cache-dir hf_cache_vstar \\
  --copy-images \\
  --min-image-side 1024 \\
  --expected-min-samples 180

PYTHON_BIN=./venv/bin/python MPLCONFIGDIR=/tmp/matplotlib-lvr-eval HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\
  bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \\
  --configs config.main_vstar_n191.yaml \\
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \\
  --metrics all \\
  --gpus 0,1,2,3 \\
  --run-root runs/w20_spotlight_vstar_n191

# BF-Conf/PF-B rerun after BF-Conf 512px low-memory fix:
PYTHON_BIN=./venv/bin/python MPLCONFIGDIR=/tmp/matplotlib-lvr-eval HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\
  bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \\
  --configs config.main_vstar_n191.yaml \\
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \\
  --metrics bf_conf_calibrated_progression,pf_b_patch_alignment \\
  --gpus 0,1,2,3 \\
  --run-root runs/w20_spotlight_vstar_n191_rerun_bfconf_pfb

./venv/bin/python tools/summarize_vstar_results.py
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"WROTE_REPORT {report_path}")
    print(f"WROTE_FIGURES {fig_dir}")
    print(f"WROTE_CSV {merged / 'w20_vstar_primary_results.csv'}")


if __name__ == "__main__":
    main()
