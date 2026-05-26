"""Minimal preregistered main-matrix regression summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_SCALARS = {
    "pf_a_corruption_selectivity": "selectivity",
    "pf_b_patch_alignment": "native_alignment",
    "bf_patch_answer_transfer": "continuous_margin_shift",
    "bf_swap_latent_replacement": "continuous_margin_shift",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_delta",
}


def _task_for_envelope(envelope: dict) -> str:
    payload = envelope.get("payload") or {}
    config = payload.get("config") or {}
    return str(envelope.get("task") or payload.get("task") or config.get("task") or "unknown")


def _rows_from_summary_ci(summary_rows: list[dict]) -> list[dict]:
    out = []
    for row in summary_rows:
        metric_id = str(row.get("metric_id") or "")
        scalar = str(row.get("scalar") or "")
        if PRIMARY_SCALARS.get(metric_id) != scalar:
            continue
        if row.get("mean") is None:
            continue
        out.append({
            "metric_id": metric_id,
            "scalar": scalar,
            "model": str(row.get("model") or "unknown"),
            "task": str(row.get("task") or "unknown"),
            "value": float(row["mean"]),
            "n": int(row.get("n") or 0),
        })
    return out


def _rows_from_metric_results(metric_results: list[dict]) -> list[dict]:
    out = []
    for envelope in metric_results:
        metric_id = str(envelope.get("metric_id") or "")
        scalar = PRIMARY_SCALARS.get(metric_id)
        if scalar is None:
            continue
        payload = envelope.get("payload") or {}
        reduction = payload.get("reduction") or {}
        value = reduction.get(scalar)
        if value is None and scalar == "continuous_margin_shift":
            if metric_id == "bf_patch_answer_transfer" and not (payload.get("config") or {}).get("trace_latent"):
                value = reduction.get("logprob_margin_shift")
            elif metric_id == "bf_swap_latent_replacement" and not (payload.get("config") or {}).get("trace_latent"):
                value = reduction.get("swap_margin_shift")
        if value is None:
            samples = payload.get("samples") or []
            values = []
            for sample in samples:
                if not isinstance(sample, dict) or sample.get("error") is not None:
                    continue
                sample_reduction = sample.get("reduction") or {}
                sample_value = sample_reduction.get(scalar)
                if sample_value is None and scalar == "continuous_margin_shift":
                    if metric_id == "bf_patch_answer_transfer" and not (payload.get("config") or {}).get("trace_latent"):
                        sample_value = sample_reduction.get("logprob_margin_shift")
                    elif metric_id == "bf_swap_latent_replacement" and not (payload.get("config") or {}).get("trace_latent"):
                        sample_value = sample_reduction.get("swap_margin_shift")
                if sample_value is not None:
                    values.append(sample_value)
            if values:
                value = float(np.mean([float(item) for item in values]))
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        out.append({
            "metric_id": metric_id,
            "scalar": scalar,
            "model": str(envelope.get("model") or payload.get("model") or "unknown"),
            "task": _task_for_envelope(envelope),
            "value": numeric,
            "n": int(reduction.get("n") or reduction.get("n_paired") or payload.get("n_paired") or 0),
        })
    return out


def _design_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    models = sorted({row["model"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    names = ["intercept"]
    names.extend(f"model={name}" for name in models[1:])
    names.extend(f"task={name}" for name in tasks[1:])
    x = np.ones((len(rows), len(names)), dtype=float)
    col = 1
    for name in models[1:]:
        x[:, col] = [1.0 if row["model"] == name else 0.0 for row in rows]
        col += 1
    for name in tasks[1:]:
        x[:, col] = [1.0 if row["task"] == name else 0.0 for row in rows]
        col += 1
    return x, names


def _ols_summary(rows: list[dict]) -> dict:
    y = np.asarray([row["value"] for row in rows], dtype=float)
    x, names = _design_matrix(rows)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    resid = y - pred
    dof = max(int(x.shape[0] - x.shape[1]), 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    return {
        "method": "ols_fixed_effects_fallback",
        "formula": "value ~ model + task",
        "n_rows": len(rows),
        "r2": float(1.0 - (np.sum(resid ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12))),
        "coefficients": [
            {
                "term": name,
                "estimate": float(beta[idx]),
                "std_error": float(se[idx]),
            }
            for idx, name in enumerate(names)
        ],
    }


def fit_mixed_effects(
    metric_results: list[dict] | None = None,
    *,
    summary_rows: list[dict] | None = None,
    out_path: str | Path | None = None,
) -> dict:
    """Fit the minimal Main-track regression summary.

    The function prefers a simple, dependency-light fixed-effect fallback so it
    is runnable in smoke and offline environments. It records the fallback
    explicitly; callers can replace it with statsmodels MixedLM later without
    changing the artifact contract.
    """
    rows = _rows_from_summary_ci(summary_rows or []) if summary_rows else []
    if not rows and metric_results:
        rows = _rows_from_metric_results(metric_results)

    by_metric: dict[str, list[dict]] = {}
    for row in rows:
        by_metric.setdefault(row["metric_id"], []).append(row)

    models: dict[str, Any] = {}
    for metric_id, metric_rows in sorted(by_metric.items()):
        if len(metric_rows) < 2:
            models[metric_id] = {
                "method": "insufficient_rows",
                "n_rows": len(metric_rows),
                "formula": "value ~ model + task",
            }
            continue
        models[metric_id] = _ols_summary(metric_rows)

    report = {
        "status": "implemented_fixed_effects_fallback",
        "intended_mixed_effects_formula": "value ~ model + task + (1 | dataset_or_sample_group)",
        "implemented_formula": "value ~ model + task",
        "n_input_rows": len(rows),
        "models": models,
    }
    if out_path is not None:
        path = Path(out_path)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def fit_from_files(summary_with_ci: str | Path, out_path: str | Path) -> dict:
    rows = json.loads(Path(summary_with_ci).read_text(encoding="utf-8"))
    return fit_mixed_effects(summary_rows=rows, out_path=out_path)
