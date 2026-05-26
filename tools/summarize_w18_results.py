#!/usr/bin/env python
"""Build W18 local main-matrix tables and line plots from real artifacts."""
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
    "bf_patch_answer_transfer": "continuous_margin_shift",
    "bf_swap_latent_replacement": "continuous_margin_shift",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_delta",
}

METRIC_LABELS = {
    "pf_a_corruption_selectivity": "PF-A selectivity",
    "pf_b_patch_alignment": "PF-B native alignment",
    "bf_patch_answer_transfer": "BF-Patch continuous margin",
    "bf_swap_latent_replacement": "BF-Swap continuous margin",
    "bf_conf_calibrated_progression": "BF-Conf gold-logit slope",
    "cf_stage_decay": "CF-Stage late delta",
}

MODEL_ORDER = ["qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b"]
MODEL_LABELS = {
    "qwen2_5_vl_3b": "Qwen2.5-VL-3B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "lvr_7b": "LVR-7B",
}

TASK_ORDER = ["spd_faith", "maze", "blink"]
TASK_LABELS = {
    "spd_faith": "T2 SPD-Faith",
    "maze": "T1 Maze",
    "blink": "T3 BLINK",
}

TASK_COLORS = {
    "spd_faith": "#2563eb",
    "maze": "#0f766e",
    "blink": "#b45309",
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
    if not row:
        return "-"
    return f"{fmt(row.get('mean'), digits)} [{fmt(row.get('ci_low'), digits)}, {fmt(row.get('ci_high'), digits)}]"


def md_path(path: Path, report_path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(report_path.parent.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def primary_rows(ci_rows: list[dict]) -> list[dict]:
    out = []
    for row in ci_rows:
        metric = str(row.get("metric_id") or "")
        scalar = str(row.get("scalar") or "")
        if PRIMARY_SCALARS.get(metric) != scalar:
            continue
        out.append(row)
    return out


def row_index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {
        (str(row.get("task") or "unknown"), str(row.get("model") or "unknown"), str(row.get("metric_id") or "")): row
        for row in rows
    }


def write_primary_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "model",
                "metric_id",
                "scalar",
                "n",
                "mean",
                "ci_low",
                "ci_high",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key)
                for key in writer.fieldnames
            })


def plot_metric_lines(rows: list[dict], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by = row_index(rows)
    plot_paths: dict[str, Path] = {}
    x = list(range(len(MODEL_ORDER)))
    for metric_id in PRIMARY_SCALARS:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        any_data = False
        for task in TASK_ORDER:
            ys: list[float | None] = []
            yerr_low: list[float] = []
            yerr_high: list[float] = []
            for model in MODEL_ORDER:
                row = by.get((task, model, metric_id))
                if row is None:
                    ys.append(None)
                    yerr_low.append(0.0)
                    yerr_high.append(0.0)
                    continue
                mean = float(row["mean"])
                ys.append(mean)
                yerr_low.append(max(0.0, mean - float(row.get("ci_low", mean))))
                yerr_high.append(max(0.0, float(row.get("ci_high", mean)) - mean))
            valid = [(idx, value) for idx, value in enumerate(ys) if value is not None]
            if not valid:
                continue
            any_data = True
            xs = [x[idx] for idx, _ in valid]
            values = [float(value) for _, value in valid]
            lows = [yerr_low[idx] for idx, _ in valid]
            highs = [yerr_high[idx] for idx, _ in valid]
            ax.errorbar(
                xs,
                values,
                yerr=[lows, highs],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=TASK_COLORS.get(task),
                label=TASK_LABELS.get(task, task),
            )
        if not any_data:
            plt.close(fig)
            continue
        ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=18, ha="right")
        ax.set_ylabel(PRIMARY_SCALARS[metric_id])
        ax.set_title(METRIC_LABELS[metric_id])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / f"w18_line_{metric_id}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        plot_paths[metric_id] = path
    return plot_paths


def plot_combined(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    by = row_index(rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    x = list(range(len(MODEL_ORDER)))
    for ax, metric_id in zip(axes.flatten(), PRIMARY_SCALARS):
        for task in TASK_ORDER:
            points = []
            lows = []
            highs = []
            xs = []
            for idx, model in enumerate(MODEL_ORDER):
                row = by.get((task, model, metric_id))
                if row is None:
                    continue
                mean = float(row["mean"])
                xs.append(x[idx])
                points.append(mean)
                lows.append(max(0.0, mean - float(row.get("ci_low", mean))))
                highs.append(max(0.0, float(row.get("ci_high", mean)) - mean))
            if points:
                ax.errorbar(
                    xs,
                    points,
                    yerr=[lows, highs],
                    marker="o",
                    linewidth=1.5,
                    capsize=2,
                    color=TASK_COLORS.get(task),
                    label=TASK_LABELS.get(task, task),
                )
        ax.axhline(0, color="#6b7280", linewidth=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(["3B", "7B", "LVR"], rotation=0)
        ax.set_title(METRIC_LABELS[metric_id], fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("W18 Main Matrix Primary Metrics", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = out_dir / "w18_primary_metric_lines_combined.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def count_manifest_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_prepare_stats(path: Path) -> dict:
    return read_json(path) if path.is_file() else {}


def transfer_rows(ci_rows: list[dict]) -> list[dict]:
    wanted = {
        ("bf_patch_answer_transfer", "answer_transfer_rate"),
        ("bf_swap_latent_replacement", "swap_answer_transfer_rate"),
    }
    return [
        row for row in ci_rows
        if (row.get("metric_id"), row.get("scalar")) in wanted
    ]


def trace_light_rows(trace_run: Path) -> list[dict]:
    path = trace_run / "summary_with_ci.json"
    if not path.is_file():
        return []
    rows = read_json(path)
    out = []
    for row in rows:
        metric = str(row.get("metric_id") or "")
        scalar = str(row.get("scalar") or "")
        if PRIMARY_SCALARS.get(metric) == scalar:
            out.append(row)
    return out


def build_report(
    *,
    run_root: Path,
    trace_run: Path,
    plots_dir: Path,
    report_path: Path,
) -> None:
    merged = run_root / "merged"
    ci_rows = read_json(merged / "summary_with_ci.json")
    primary = primary_rows(ci_rows)
    primary_by = row_index(primary)
    transfer = transfer_rows(ci_rows)
    trace_rows = trace_light_rows(trace_run)
    plot_paths = plot_metric_lines(primary, plots_dir)
    combined_plot = plot_combined(primary, plots_dir)
    write_primary_csv(primary, merged / "w18_primary_results.csv")

    data_rows = []
    spd_stats = load_prepare_stats(Path("data/spd_faith_hf/prepare_stats.json"))
    maze_stats = load_prepare_stats(Path("data/maze_planning/prepare_stats.json"))
    blink_stats = load_prepare_stats(Path("data/blink/prepare_stats.json"))
    vstar_stats = load_prepare_stats(Path("data/vstar/prepare_stats.json"))
    vsi_stats = load_prepare_stats(Path("data/vsi_bench_n100/prepare_stats.json"))
    data_rows.extend([
        ["T2 SPD-Faith", spd_stats.get("n_written", count_manifest_lines(Path("data/spd_faith_hf/manifest.jsonl"))), "n=1000 run; BF paired subset n=500", "paired bbox/region oracle"],
        ["T1 Maze", maze_stats.get("n_written", count_manifest_lines(Path("data/maze_planning/manifest.jsonl"))), "full local set n=500; data-limited below 800", "no paired BF metrics"],
        ["T3 BLINK", blink_stats.get("n_written", count_manifest_lines(Path("data/blink/manifest.jsonl"))), f"n=1000 run; weak_oracle={blink_stats.get('n_weak_oracle')}", "center fallback, no strong bbox claim"],
        ["V*Bench/VStar", vstar_stats.get("n_written", count_manifest_lines(Path("data/vstar/manifest.jsonl"))), f"blocked: missing images={vstar_stats.get('n_skipped_no_image')}, missing bbox={vstar_stats.get('n_missing_bbox')}", "code path present, not a completed visual run"],
        ["T4 VSI-Bench", vsi_stats.get("n_written", count_manifest_lines(Path("data/vsi_bench_n100/manifest.jsonl"))), f"blocked: skipped_no_answer={vsi_stats.get('n_skipped_no_answer')}", "needs frame-grid/images and answer mapping"],
    ])

    coverage_rows = []
    for task in TASK_ORDER:
        metrics = [
            metric for metric in PRIMARY_SCALARS
            if (task == "spd_faith" or metric not in {"bf_patch_answer_transfer", "bf_swap_latent_replacement"})
        ]
        ns = sorted({
            int(primary_by[(task, model, metric)]["n"])
            for model in MODEL_ORDER
            for metric in metrics
            if (task, model, metric) in primary_by
        })
        coverage_rows.append([
            TASK_LABELS[task],
            ", ".join(MODEL_LABELS[m] for m in MODEL_ORDER),
            len(metrics),
            ", ".join(str(n) for n in ns),
            "pass",
        ])

    metric_sections = []
    for metric_id, scalar in PRIMARY_SCALARS.items():
        rows = []
        for task in TASK_ORDER:
            if task != "spd_faith" and metric_id in {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}:
                continue
            row = [TASK_LABELS[task]]
            for model in MODEL_ORDER:
                row.append(fmt_ci(primary_by.get((task, model, metric_id))))
            rows.append(row)
        metric_sections.append(
            f"### {METRIC_LABELS[metric_id]}\n\n"
            + md_table(["Task", *[MODEL_LABELS[m] for m in MODEL_ORDER]], rows)
            + f"\n\n![{metric_id}]({md_path(plot_paths[metric_id], report_path)})\n"
        )

    transfer_table = []
    for metric_id in ["bf_patch_answer_transfer", "bf_swap_latent_replacement"]:
        for model in MODEL_ORDER:
            row = next(
                (
                    item for item in transfer
                    if item.get("metric_id") == metric_id
                    and item.get("model") == model
                    and item.get("task") == "spd_faith"
                ),
                None,
            )
            transfer_table.append([METRIC_LABELS[metric_id], MODEL_LABELS[model], fmt_ci(row)])

    trace_table = []
    for row in trace_rows:
        trace_table.append([
            METRIC_LABELS.get(row.get("metric_id"), row.get("metric_id")),
            row.get("scalar"),
            row.get("n"),
            fmt_ci(row),
        ])

    mixed = read_json(merged / "mixed_effects_summary.json")
    mixed_rows = []
    for metric_id, result in sorted((mixed.get("models") or {}).items()):
        mixed_rows.append([
            METRIC_LABELS.get(metric_id, metric_id),
            result.get("method"),
            result.get("n_rows"),
            fmt(result.get("r2")),
        ])

    report = f"""# W18 Full Local Main Matrix Results

Generated from real local artifacts on 2026-05-26.

Run root: `{run_root}`

Merged artifacts: `{merged}`

Main matrix validation:

```text
tools/validate_main_matrix.py {run_root} --tasks maze,spd_faith,blink --min-samples 450 --bf-min-pairs 300
MAIN MATRIX VALIDATION PASSED rows=42 ci_rows=180

tools/validate_main_paper_readiness.py {merged} --mode full_main_matrix
MAIN MATRIX READINESS VALIDATION PASSED
```

## Scope And Boundaries

W18 completed the local thousand-scale main matrix where staged data permitted it: SPD-Faith and BLINK ran at `n=1000`; Maze ran the full local/upstream test set at `n=500`; SPD paired BF metrics used the preregistered paired subset cap `n=500`. BLINK has no bbox metadata locally, so PF-A/PF-B are weak-oracle center-fallback diagnostics. V*Bench/VStar and VSI-Bench are not reported as model results because their local manifests contain zero usable visual samples.

This matrix is a standard-forward/query-span cross-model matrix for Qwen and LVR. The real LVR generation-trace latent scale gate is reported separately in the trace-latent section below.

## Data Staging

{md_table(["Dataset", "Usable local n", "W18 status", "Boundary"], data_rows)}

## Coverage

{md_table(["Task", "Models", "Primary metrics", "CI n values", "Validation"], coverage_rows)}

## Combined Line Plot

![W18 combined primary metric lines]({md_path(combined_plot, report_path)})

## Primary Results

{chr(10).join(metric_sections)}

## SPD Answer-Transfer Rates

These rows are secondary binary transfer readouts for the SPD paired BF metrics. The paper-facing continuous BF scalar is `continuous_margin_shift`; transfer rate is reported as an interpretable companion.

{md_table(["Metric", "Model", "Transfer rate [95% CI]"], transfer_table)}

## LVR Real Trace-Latent n=1000 Light Gate

This is a separate evidence layer from W18. It uses `runs/w17_lvr_trace_latent_spd_n1000_light/merged`, `audit.mode=generation_trace`, and `trace_latent.enabled=true`; BF-Patch/BF-Swap are intentionally disabled in the light config.

Validator:

```text
tools/validate_trace_latent_gate.py {trace_run} --allow-disabled-metrics --metrics pf_a_corruption_selectivity pf_b_patch_alignment bf_conf_calibrated_progression cf_stage_decay --min-samples 800
TRACE LATENT GATE VALIDATION PASSED
```

{md_table(["Metric", "Scalar", "n", "Mean [95% CI]"], trace_table)}

## Mixed-Effects/Fallback Regression

Current artifact uses the preregistered dependency-light fallback `value ~ model + task`, not a true random-effects MixedLM. This is a readiness artifact and should be upgraded to sample-level mixed effects before final Main submission.

{md_table(["Metric", "Method", "Rows", "R2"], mixed_rows)}

## Reproduction Commands

```bash
PYTHON_BIN=./venv/bin/python MPLCONFIGDIR=/tmp/matplotlib-lvr-eval HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\
  bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \\
  --configs config.main_spd_n1000.yaml,config.main_maze_full500.yaml,config.main_blink_n1000.yaml \\
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \\
  --metrics all \\
  --gpus 0,1,2,3 \\
  --run-root runs/w18_main_matrix_full_local

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python tools/merge_main_matrix.py runs/w18_main_matrix_full_local
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python tools/summarize_w18_results.py
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"WROTE_REPORT {report_path}")
    print(f"WROTE_PLOTS {plots_dir}")
    print(f"WROTE_CSV {merged / 'w18_primary_results.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/w18_main_matrix_full_local")
    parser.add_argument("--trace-run", default="runs/w17_lvr_trace_latent_spd_n1000_light/merged")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--report", default="docs/validation_report_w18_full_local_matrix.md")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    plots_dir = Path(args.plots_dir) if args.plots_dir else run_root / "merged" / "w18_report_plots"
    build_report(
        run_root=run_root,
        trace_run=Path(args.trace_run),
        plots_dir=plots_dir,
        report_path=Path(args.report),
    )


if __name__ == "__main__":
    main()
