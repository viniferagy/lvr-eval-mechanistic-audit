#!/usr/bin/env python
"""Summarize BF usage diagnostics from saved BF-Patch/BF-Swap artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.stats.bf_usage_diagnostics import build_diagnostics


MODEL_LABELS = {
    "qwen2_5_vl_3b": "Qwen2.5-VL-3B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "lvr_7b": "LVR-7B",
}

METRIC_LABELS = {
    "bf_patch_answer_transfer": "BF-Patch",
    "bf_swap_latent_replacement": "BF-Swap",
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


def pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


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
        try:
            return path.resolve().relative_to(report_path.parent.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def load_envelopes(run_roots: list[Path]) -> list[dict]:
    envelopes: list[dict] = []
    for root in run_roots:
        for path in sorted((root / "metrics").glob("*.json")):
            try:
                env = read_json(path)
            except json.JSONDecodeError:
                continue
            if str(env.get("metric_id") or "") not in {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}:
                continue
            env = dict(env)
            env.setdefault("source_run_root", str(root))
            env.setdefault("source_file", str(path))
            payload = env.get("payload")
            if isinstance(payload, dict):
                payload.setdefault("source_run_root", str(root))
                payload.setdefault("source_file", str(path))
            envelopes.append(env)
    return envelopes


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_flat_rows(diagnostics: dict) -> list[dict]:
    rows = []
    for summary in diagnostics.get("summaries") or []:
        agreement = summary.get("agreement") or {}
        rows.append({
            "metric_id": summary.get("metric_id"),
            "task": summary.get("task"),
            "model": summary.get("model"),
            "source_run_root": summary.get("source_run_root"),
            "n_records": summary.get("n_records"),
            "n_shift": summary.get("n_shift"),
            "continuous_margin_coverage": summary.get("continuous_margin_coverage"),
            "mean_signed_shift": summary.get("mean_signed_shift"),
            "mean_abs_shift": summary.get("mean_abs_shift"),
            "median_shift": summary.get("median_shift"),
            "directional_accuracy": summary.get("directional_accuracy"),
            "answer_transfer_rate": summary.get("answer_transfer_rate"),
            **agreement,
        })
    return rows


def source_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("source_run_root")),
        str(row.get("task")),
        str(row.get("metric_id")),
        str(row.get("model")),
    )


def run_label(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    return Path(text).parent.name if Path(text).name == "merged" else Path(text).name


def plot_shift_histograms(record_rows: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in record_rows:
        if row.get("group") != "primary" or row.get("continuous_margin_shift") is None:
            continue
        key = (run_label(row.get("source_run_root")), str(row.get("metric_id")), str(row.get("task")))
        grouped.setdefault(key, []).append(row)
    for (run, metric_id, task), rows in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        any_data = False
        for model, label in MODEL_LABELS.items():
            vals = [
                float(row["continuous_margin_shift"])
                for row in rows
                if row.get("model") == model and row.get("continuous_margin_shift") is not None
            ]
            if not vals:
                continue
            any_data = True
            ax.hist(vals, bins=40, alpha=0.42, density=True, label=label)
        if not any_data:
            plt.close(fig)
            continue
        ax.axvline(0, color="#4b5563", linestyle="--", linewidth=0.9)
        ax.set_title(f"{METRIC_LABELS.get(metric_id, metric_id)} shift distribution ({task})")
        ax.set_xlabel("continuous_margin_shift")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / f"bf_usage_hist_{run}_{task}_{metric_id}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_boundary_bins(boundary_rows: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in boundary_rows:
        key = (run_label(row.get("source_run_root")), str(row.get("metric_id")), str(row.get("task")))
        grouped.setdefault(key, []).append(row)
    bins = ["near", "medium", "far"]
    for (run, metric_id, task), rows in sorted(grouped.items()):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True)
        any_data = False
        offsets = {"qwen2_5_vl_3b": -0.24, "qwen2_5_vl_7b": 0.0, "lvr_7b": 0.24}
        width = 0.22
        for model, label in MODEL_LABELS.items():
            model_rows = {
                str(row.get("boundary_bin")): row
                for row in rows
                if row.get("model") == model
            }
            if not model_rows:
                continue
            any_data = True
            xs = [idx + offsets[model] for idx in range(len(bins))]
            signed = [
                model_rows.get(bin_name, {}).get("mean_signed_shift") or 0.0
                for bin_name in bins
            ]
            transfers = [
                model_rows.get(bin_name, {}).get("answer_transfer_rate") or 0.0
                for bin_name in bins
            ]
            axes[0].bar(xs, signed, width=width, label=label)
            axes[1].bar(xs, transfers, width=width, label=label)
        if not any_data:
            plt.close(fig)
            continue
        for ax, ylabel in zip(axes, ["mean signed shift", "transfer rate"]):
            ax.axhline(0, color="#4b5563", linestyle="--", linewidth=0.8)
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(bins)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_title("Margin movement by clean-margin bin")
        axes[1].set_title("Transfer by clean-margin bin")
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{METRIC_LABELS.get(metric_id, metric_id)} boundary diagnostics ({task})")
        fig.tight_layout()
        path = out_dir / f"bf_usage_boundary_{run}_{task}_{metric_id}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_control_normalized(control_rows: list[dict], out_dir: Path) -> list[Path]:
    rows = [row for row in control_rows if row.get("control") == "random_pair_swap"]
    if not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((run_label(row.get("source_run_root")), str(row.get("task"))), []).append(row)
    paths: list[Path] = []
    for (run, task), task_rows in sorted(grouped.items()):
        labels = []
        signed = []
        absolute = []
        for model in MODEL_LABELS:
            row = next((item for item in task_rows if item.get("model") == model), None)
            if row is None:
                continue
            labels.append(MODEL_LABELS[model])
            signed.append(row.get("signed_shift_minus_control") or 0.0)
            absolute.append(row.get("abs_shift_minus_control") or 0.0)
        if not labels:
            continue
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.bar([v - 0.18 for v in x], signed, width=0.34, label="signed minus random")
        ax.bar([v + 0.18 for v in x], absolute, width=0.34, label="abs minus random")
        ax.axhline(0, color="#4b5563", linestyle="--", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"BF-Swap random-control normalization ({task}, {run})")
        ax.set_ylabel("primary - random_pair_swap")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / f"bf_usage_control_normalized_{run}_{task}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def aggregate_control_rows(control_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in control_rows:
        key = (
            str(row.get("source_run_root")),
            str(row.get("task")),
            str(row.get("model")),
            str(row.get("control")),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for (source_run_root, task, model, control), rows in sorted(groups.items()):
        def avg(key: str) -> float | None:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        control_abs = avg("control_mean_abs_shift")
        out.append({
            "metric_id": "bf_swap_latent_replacement",
            "source_run_root": source_run_root,
            "task": task,
            "model": model,
            "control": control,
            "n_cells": len(rows),
            "signed_shift_minus_control": avg("signed_shift_minus_control"),
            "abs_shift_minus_control": avg("abs_shift_minus_control"),
            "abs_specificity_ratio": avg("abs_specificity_ratio") if control_abs is not None and abs(control_abs) > 1e-9 else None,
            "primary_answer_transfer_rate": avg("primary_answer_transfer_rate"),
            "control_answer_transfer_rate": avg("control_answer_transfer_rate"),
        })
    return out


def report_text(
    diagnostics: dict,
    *,
    run_roots: list[Path],
    out_dir: Path,
    report_path: Path,
    figure_paths: list[Path],
) -> str:
    flat = summary_flat_rows(diagnostics)
    summary_rows = []
    for row in sorted(flat, key=source_key):
        summary_rows.append([
            run_label(row.get("source_run_root")),
            row["task"],
            METRIC_LABELS.get(row["metric_id"], row["metric_id"]),
            MODEL_LABELS.get(row["model"], row["model"]),
            row["n_records"],
            row["n_shift"],
            pct(row["continuous_margin_coverage"]),
            fmt(row["mean_signed_shift"]),
            fmt(row["mean_abs_shift"]),
            pct(row["directional_accuracy"]),
            pct(row["answer_transfer_rate"]),
        ])

    boundary_rows = []
    for row in sorted(diagnostics.get("boundary_rows") or [], key=lambda r: (
        str(r.get("source_run_root")),
        str(r.get("task")),
        str(r.get("metric_id")),
        str(r.get("model")),
        str(r.get("boundary_bin")),
    )):
        if row.get("boundary_bin") == "unknown":
            continue
        boundary_rows.append([
            run_label(row.get("source_run_root")),
            row.get("task"),
            METRIC_LABELS.get(row.get("metric_id"), row.get("metric_id")),
            MODEL_LABELS.get(row.get("model"), row.get("model")),
            row.get("boundary_bin"),
            row.get("n_records"),
            row.get("n_shift"),
            fmt(row.get("mean_signed_shift")),
            fmt(row.get("mean_abs_shift")),
            pct(row.get("directional_accuracy")),
            pct(row.get("answer_transfer_rate")),
        ])

    source_rows = []
    for row in sorted(diagnostics.get("margin_source_rows") or [], key=lambda r: (
        str(r.get("source_run_root")),
        str(r.get("task")),
        str(r.get("metric_id")),
        str(r.get("model")),
        str(r.get("margin_source")),
    )):
        source_rows.append([
            run_label(row.get("source_run_root")),
            row.get("task"),
            METRIC_LABELS.get(row.get("metric_id"), row.get("metric_id")),
            MODEL_LABELS.get(row.get("model"), row.get("model")),
            row.get("margin_source"),
            row.get("n_records"),
            row.get("n_shift"),
            fmt(row.get("mean_signed_shift")),
            fmt(row.get("mean_abs_shift")),
            pct(row.get("answer_transfer_rate")),
        ])

    control_rows = []
    for row in aggregate_control_rows(diagnostics.get("control_normalized_rows") or []):
        control_rows.append([
            run_label(row.get("source_run_root")),
            row.get("task"),
            MODEL_LABELS.get(row.get("model"), row.get("model")),
            row.get("control"),
            row.get("n_cells"),
            fmt(row.get("signed_shift_minus_control")),
            fmt(row.get("abs_shift_minus_control")),
            fmt(row.get("abs_specificity_ratio")),
            pct(row.get("primary_answer_transfer_rate")),
            pct(row.get("control_answer_transfer_rate")),
        ])

    figure_lines = "\n".join(
        f"![{path.stem}]({md_path(path, report_path)})"
        for path in figure_paths
    ) or "_No figures were generated because no continuous BF shifts were available._"

    roots = ", ".join(f"`{root}`" for root in run_roots)
    return f"""# BF Usage Diagnostics

Generated from saved BF-Patch/BF-Swap artifacts. No model inference is run.

Input roots: {roots}

Output directory: `{out_dir}`

## Interpretation

This report separates the BF evidence into directional, magnitude, boundary, control, and provenance views. `mean_signed_shift` is the usual directional continuous margin. `mean_abs_shift` measures whether patching moves the margin at all, even if positive and negative effects cancel. `directional_accuracy` is `P(continuous_margin_shift > 0)`. Boundary bins use `abs(clean_margin)`: `near < 0.1`, `medium < 0.5`, `far >= 0.5`.

## Summary

{md_table(["Run", "Task", "Metric", "Model", "Records", "Shift n", "Coverage", "Signed", "Abs", "Direction", "Transfer"], summary_rows)}

## Boundary Bins

{md_table(["Run", "Task", "Metric", "Model", "Bin", "Records", "Shift n", "Signed", "Abs", "Direction", "Transfer"], boundary_rows)}

## Margin Provenance

{md_table(["Run", "Task", "Metric", "Model", "Source", "Records", "n shift", "Signed", "Abs", "Transfer"], source_rows)}

## BF-Swap Control Normalization

`primary - random_pair_swap` is the key source-specificity diagnostic. Positive absolute-shift difference means the paired counterfactual swap moves the margin more than a random latent replacement; near-zero values mean the effect is not specific to the source pair.

{md_table(["Run", "Task", "Model", "Control", "Cells", "Signed delta", "Abs delta", "Abs ratio", "Primary transfer", "Control transfer"], control_rows)}

## Figures

{figure_lines}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_roots",
        nargs="*",
        type=Path,
        default=[Path("runs/w18_main_matrix_full_local/merged")],
        help="Run roots containing metrics/*.json artifacts.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/bf_usage_diagnostics"))
    parser.add_argument("--report", type=Path, default=Path("docs/validation_report_bf_usage_diagnostics.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures/bf_usage_diagnostics"))
    args = parser.parse_args()

    envelopes = load_envelopes(args.run_roots)
    diagnostics = build_diagnostics(envelopes)
    control_agg = aggregate_control_rows(diagnostics.get("control_normalized_rows") or [])
    diagnostics["control_normalized_aggregate_rows"] = control_agg

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "bf_usage_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "bf_usage_diagnostics_summary.csv", summary_flat_rows(diagnostics))
    write_csv(args.out_dir / "bf_usage_diagnostics_records.csv", diagnostics.get("record_rows") or [])
    write_csv(args.out_dir / "bf_usage_diagnostics_boundary_bins.csv", diagnostics.get("boundary_rows") or [])
    write_csv(args.out_dir / "bf_usage_diagnostics_margin_sources.csv", diagnostics.get("margin_source_rows") or [])
    write_csv(args.out_dir / "bf_usage_diagnostics_control_normalized.csv", control_agg)

    figure_paths = []
    figure_paths.extend(plot_shift_histograms(diagnostics.get("record_rows") or [], args.figures_dir))
    figure_paths.extend(plot_boundary_bins(diagnostics.get("boundary_rows") or [], args.figures_dir))
    figure_paths.extend(plot_control_normalized(control_agg, args.figures_dir))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        report_text(
            diagnostics,
            run_roots=args.run_roots,
            out_dir=args.out_dir,
            report_path=args.report,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )
    print(f"WROTE_JSON {args.out_dir / 'bf_usage_diagnostics.json'}")
    print(f"WROTE_SUMMARY_CSV {args.out_dir / 'bf_usage_diagnostics_summary.csv'}")
    print(f"WROTE_REPORT {args.report}")
    print(f"WROTE_FIGURES {args.figures_dir}")


if __name__ == "__main__":
    main()
