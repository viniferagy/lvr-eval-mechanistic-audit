"""
pipeline/analysis.py
====================
后处理与可视化(内部指标版):

  BF-1: 逐层 Δbf3 / Δpf3 折线(多模型 overlay) + Δ 热力图
        + baseline BF-3 entropy curve / PF-3 KL curve
  CF-2: 每个 corruption family 一张 decay 曲线
  联动: 4 维 radar(BF-1 criticality / BF-3 sharpening / PF-3 modality-dep / CF-2 AUC)
        + 关键层 Δbf3~Δpf3 的 Spearman

全部存图到 run 目录, Agg backend 无需 GUI。
"""
from __future__ import annotations

import json
import logging
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .results import split_metric_results
from .stats.bootstrap import paired_bootstrap

logger = logging.getLogger("lvr_eval.analysis")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _curve_array(values) -> np.ndarray | None:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr


# --------------------------------------------------------------------------- #
#  BF-1
# --------------------------------------------------------------------------- #
def plot_ablation(ablation: dict[str, dict], out_dir: str):
    for which in ("bf3", "pf3"):
        fig, ax = plt.subplots(figsize=(10, 5))
        any_data = False
        for tag, res in ablation.items():
            layers = [r["layer"] for r in res["layers"] if r["delta"].get(which) is not None]
            deltas = [r["delta"][which] for r in res["layers"] if r["delta"].get(which) is not None]
            if not layers:
                continue
            any_data = True
            ax.plot(layers, deltas, marker="o", ms=3, label=tag)
        if not any_data:
            plt.close(fig)
            continue
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        scal = next(iter(ablation.values())).get(f"{which}_scalar", which)
        ax.set_xlabel("ablated decoder layer")
        ax.set_ylabel(f"Δ {which} ({scal})  baseline − ablated")
        ax.set_title(f"BF-1 Latent Ablation → {which.upper()} criticality")
        ax.legend()
        _save(fig, os.path.join(out_dir, f"bf1_layerwise_{which}.png"))

    # baseline 原始 curve 复刻
    for which, ylab, fname in [("bf3_curve", "entropy (nats)", "bf1_baseline_bf3_curve.png"),
                               ("pf3_curve", "attention KL", "bf1_baseline_pf3_curve.png")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        any_data = False
        for tag, res in ablation.items():
            c = res.get("baseline", {}).get(which)
            if not c:
                continue
            any_data = True
            ax.plot(range(1, len(c) + 1), c, marker="o", ms=3, label=tag)
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel("layer index")
        ax.set_ylabel(ylab)
        ax.set_title(f"Baseline {which.split('_')[0].upper()} curve")
        ax.legend()
        _save(fig, os.path.join(out_dir, fname))


# --------------------------------------------------------------------------- #
#  CF-2
# --------------------------------------------------------------------------- #
def plot_decay(decay: dict[str, dict], out_dir: str):
    fams = set()
    for res in decay.values():
        fams.update(res["families"].keys())
    for fam in sorted(fams):
        fig, ax = plt.subplots(figsize=(7, 5))
        any_data = False
        readout = "metric"
        for tag, res in decay.items():
            if fam not in res["families"]:
                continue
            any_data = True
            readout = res["readout"]
            d = res["families"][fam]
            auc = d["features"]["auc"]
            ax.plot(d["severities"], d["curve"], marker="o",
                    label=f"{tag} (AUC={auc:.3f})")
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel(f"{fam} severity")
        ax.set_ylabel(readout.upper())
        ax.set_title(f"CF-2 PF Decay Curve — {fam}")
        ax.legend()
        _save(fig, os.path.join(out_dir, f"cf2_decay_{fam}.png"))


# --------------------------------------------------------------------------- #
#  联动 radar
# --------------------------------------------------------------------------- #
def build_summary(ablation: dict, decay: dict) -> dict:
    summary: dict[str, dict] = {}
    for tag in set(ablation) | set(decay):
        row = {}
        if tag in ablation:
            ar = ablation[tag]
            d_bf3 = [r["delta"]["bf3"] for r in ar["layers"] if r["delta"].get("bf3") is not None]
            d_pf3 = [r["delta"]["pf3"] for r in ar["layers"] if r["delta"].get("pf3") is not None]
            row["bf1_criticality"] = float(max(np.abs(d_bf3 + d_pf3))) if (d_bf3 or d_pf3) else None
            base = ar.get("baseline", {})
            row["bf3"] = base.get("bf3", {}).get("early_to_late_drop")
            row["pf3"] = base.get("pf3", {}).get("mean_kl")
        if tag in decay:
            aucs = [f["features"]["auc"] for f in decay[tag]["families"].values()]
            row["cf2_robustness"] = float(np.mean(aucs)) if aucs else None
        summary[tag] = row
    return summary


def plot_radar(summary: dict, out_dir: str):
    keys = ["bf1_criticality", "bf3", "cf2_robustness", "pf3"]
    labels = ["BF-1\ncriticality", "BF-3\nsharpening", "CF-2\nAUC", "PF-3\nmodality-dep"]
    vals = {k: np.array([summary[t].get(k) if summary[t].get(k) is not None else np.nan
                         for t in summary], float) for k in keys}
    norm = {}
    for k in keys:
        v = vals[k]
        lo, hi = np.nanmin(v), np.nanmax(v)
        norm[k] = (v - lo) / (hi - lo + 1e-9) if hi > lo else np.ones_like(v) * 0.5
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, tag in enumerate(summary):
        data = [norm[k][i] for k in keys]
        data += data[:1]
        ax.plot(ang, data, marker="o", label=tag)
        ax.fill(ang, data, alpha=0.1)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("4-Metric Radar (per-axis min-max normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    _save(fig, os.path.join(out_dir, "radar_4metric.png"))


def rank_correlation(ablation: dict, out_dir: str):
    try:
        from scipy.stats import spearmanr
    except Exception:  # noqa: BLE001
        return
    preferred = ("lvr_7b", "M2")
    tag = next((name for name in preferred if name in ablation), next(iter(ablation), None))
    if tag is None:
        return
    ar = ablation[tag]
    pairs = [(r["delta"].get("bf3"), r["delta"].get("pf3")) for r in ar["layers"]]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return
    a, b = zip(*pairs)
    rho, p = spearmanr(a, b)
    with open(os.path.join(out_dir, "rank_correlation.json"), "w") as f:
        json.dump({"model": tag, "layerwise_dBF3_vs_dPF3":
                   {"spearman": float(rho), "p": float(p)}}, f, indent=2)
    logger.info("[%s] layerwise Δbf3~Δpf3 spearman=%.3f (p=%.3g)", tag, rho, p)


def plot_generic_curve(metric_id: str, model: str, label: str,
                       y_values, out_dir: str, x_values=None):
    y = _curve_array(y_values)
    if y is None:
        return
    x = np.arange(1, len(y) + 1) if x_values is None else _curve_array(x_values)
    if x is None or len(x) != len(y):
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, marker="o", ms=3)
    ax.set_xlabel("index")
    ax.set_ylabel(label)
    ax.set_title(f"{metric_id} / {model} / {label}")
    path = os.path.join(
        out_dir,
        f"{_safe_name(metric_id)}_{_safe_name(model)}_{_safe_name(label)}.png",
    )
    _save(fig, path)


def plot_generic_layer_deltas(metric_id: str, model: str, payload: dict, out_dir: str):
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return
    keys = sorted({
        key
        for layer in layers
        for key in (layer.get("delta") or {})
        if layer.get("delta", {}).get(key) is not None
    })
    if not keys:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    any_data = False
    for key in keys:
        xs, ys = [], []
        for layer in layers:
            value = (layer.get("delta") or {}).get(key)
            if value is None:
                continue
            xs.append(layer.get("layer"))
            ys.append(value)
        if xs:
            any_data = True
            ax.plot(xs, ys, marker="o", ms=3, label=f"delta.{key}")
    if not any_data:
        plt.close(fig)
        return
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("layer")
    ax.set_ylabel("delta")
    ax.set_title(f"{metric_id} / {model} / layer deltas")
    ax.legend()
    _save(fig, os.path.join(out_dir, f"{_safe_name(metric_id)}_{_safe_name(model)}_layer_deltas.png"))


def plot_generic_metric_results(metric_results: list[dict], out_dir: str):
    """Create metric-id based plots without assuming BF-1/CF-2 file names."""
    if not metric_results:
        return
    generic_dir = os.path.join(out_dir, "metric_plots")
    os.makedirs(generic_dir, exist_ok=True)

    summary = []
    for envelope in metric_results:
        metric_id = str(envelope.get("metric_id", "metric"))
        model = str(envelope.get("model", "model"))
        payload = envelope.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        summary.append({
            "metric_id": metric_id,
            "model": model,
            "result_type": envelope.get("result_type"),
            "payload_keys": sorted(payload.keys()),
        })

        baseline = payload.get("baseline")
        if isinstance(baseline, dict):
            for key, value in baseline.items():
                if key.endswith("_curve"):
                    plot_generic_curve(metric_id, model, f"baseline_{key}", value, generic_dir)

        if payload.get("curve") is not None:
            plot_generic_curve(metric_id, model, "curve", payload.get("curve"), generic_dir)

        plot_generic_layer_deltas(metric_id, model, payload, generic_dir)

        families = payload.get("families")
        if isinstance(families, dict):
            for family, family_result in families.items():
                if not isinstance(family_result, dict):
                    continue
                plot_generic_curve(
                    metric_id,
                    model,
                    f"{family}_curve",
                    family_result.get("curve"),
                    generic_dir,
                    x_values=family_result.get("severities"),
                )

    with open(os.path.join(out_dir, "metric_results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _numeric_sample_scalars(payload: dict) -> dict[str, list[float]]:
    scalars: dict[str, list[float]] = {}
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return scalars
    for record in samples:
        if not isinstance(record, dict) or record.get("error") is not None:
            continue
        reduction = record.get("reduction")
        if isinstance(reduction, dict):
            for key, value in reduction.items():
                try:
                    scalars.setdefault(str(key), []).append(float(value))
                except (TypeError, ValueError):
                    continue
        for key in ("scalar", "value", "selectivity", "answer_transfer_rate"):
            if key in record:
                try:
                    scalars.setdefault(key, []).append(float(record[key]))
                except (TypeError, ValueError):
                    continue
    return scalars


def build_summary_with_ci(metric_results: list[dict], seed: int = 260523) -> list[dict]:
    rows: list[dict] = []
    for envelope in metric_results:
        payload = envelope.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        metric_id = str(envelope.get("metric_id"))
        model = str(envelope.get("model"))
        for scalar, values in sorted(_numeric_sample_scalars(payload).items()):
            ci = paired_bootstrap(values, seed=seed)
            if ci is None:
                rows.append({
                    "metric_id": metric_id,
                    "model": model,
                    "scalar": scalar,
                    "n": 0,
                    "skip_reason": "empty_samples",
                    "seed": seed,
                })
                continue
            rows.append({
                "metric_id": metric_id,
                "model": model,
                "scalar": scalar,
                **ci.as_dict(),
            })
    return rows


def run_analysis(ablation: dict | None, decay: dict | None, out_dir: str,
                 metric_results: list[dict] | None = None, **_):
    os.makedirs(out_dir, exist_ok=True)
    ablation = dict(ablation or {})
    decay = dict(decay or {})
    if metric_results:
        metric_ablation, metric_decay = split_metric_results(metric_results)
        ablation.update({k: v for k, v in metric_ablation.items() if k not in ablation})
        decay.update({k: v for k, v in metric_decay.items() if k not in decay})
        plot_generic_metric_results(metric_results, out_dir)
        with open(os.path.join(out_dir, "summary_with_ci.json"), "w", encoding="utf-8") as f:
            json.dump(build_summary_with_ci(metric_results), f, indent=2, ensure_ascii=False)

    if ablation:
        plot_ablation(ablation, out_dir)
        rank_correlation(ablation, out_dir)
    if decay:
        plot_decay(decay, out_dir)
    summary = build_summary(ablation, decay)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if summary:
        plot_radar(summary, out_dir)
    logger.info("analysis done -> %s", out_dir)
