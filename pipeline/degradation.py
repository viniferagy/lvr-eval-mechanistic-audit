"""
pipeline/degradation.py
=======================
CF-2: PF Decay Curve —— 与 PF-3 / BF-3 联动版。

把 PF-3 的离散 corruption 推广到 severity 连续轴, 观察内部指标随退化的变化:

  x 轴 = corruption severity
  y 轴(默认 pf3) = PF-3 attention KL(intact || corrupted_at_s) 的 mean_kl
       severity↑ 通常 KL↑ (注意越破坏越敏感 -> "PF 上升曲线")
  y 轴(可选 bf3) = 把退化图喂给 BF-3, 取 final-layer entropy
       severity↑ -> entropy↑ (置信下降)

severity 含义随 corruption family:
  mask_*        : mask ratio (0..1)
  gaussian_blur : blur radius

每个 severity 点跑整个 probe set 取均值, 最后拟合 AUC / 特征 severity。
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import numpy as np

from .data import ProbeSample
from . import internal_metrics as IM
from .metrics import resolve_readout

logger = logging.getLogger("lvr_eval.degradation")


def _empty_span_stats() -> dict:
    return {"n_total": 0, "n_success": 0, "n_skipped": 0,
            "skip_reasons": {}, "query_target_counts": {}, "examples": []}


def _record_span_success(stats: dict, meta: dict):
    stats["n_success"] += 1
    kind = meta.get("query_target_kind", "unknown")
    stats["query_target_counts"][kind] = stats["query_target_counts"].get(kind, 0) + 1
    if len(stats["examples"]) < 5:
        stats["examples"].append({
            "query_target_kind": kind,
            "query_span": meta.get("query_span"),
            "image_span": meta.get("image_span"),
            "image_preprocess": meta.get("image_preprocess"),
            "adapter_notes": meta.get("adapter_notes", {}),
        })


def _record_span_skip(stats: dict, reason: str):
    stats["n_skipped"] += 1
    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1


def _merge_span_stats(dst: dict, src: dict):
    dst["n_total"] += src.get("n_total", 0)
    dst["n_success"] += src.get("n_success", 0)
    dst["n_skipped"] += src.get("n_skipped", 0)
    for reason, count in src.get("skip_reasons", {}).items():
        dst["skip_reasons"][reason] = dst["skip_reasons"].get(reason, 0) + count
    for kind, count in src.get("query_target_counts", {}).items():
        dst["query_target_counts"][kind] = dst["query_target_counts"].get(kind, 0) + count
    remaining = max(0, 5 - len(dst["examples"]))
    dst["examples"].extend(src.get("examples", [])[:remaining])
    dst["valid_rate"] = dst["n_success"] / max(dst["n_total"], 1)


# --------------------------------------------------------------------------- #
#  曲线特征 (对“上升”与“下降”型都适用)
# --------------------------------------------------------------------------- #
def curve_features(sev: list[float], y: list[float]) -> dict:
    s = np.asarray(sev, float)
    v = np.asarray(y, float)
    order = np.argsort(s)
    s, v = s[order], v[order]
    span = s[-1] - s[0]
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trapz(v, s) / span) if span > 0 else float(v.mean())

    clean = v[0]
    final = v[-1]
    rel_change = float((final - clean) / (abs(clean) + 1e-9))  # >0 上升, <0 下降
    half = clean + 0.5 * (final - clean)                       # 半程值
    char_sev = None
    for i in range(1, len(v)):
        lo, hi = sorted((v[i - 1], v[i]))
        if lo <= half <= hi:
            denom = (v[i] - v[i - 1])
            t = (half - v[i - 1]) / denom if abs(denom) > 1e-9 else 0.0
            char_sev = float(s[i - 1] + t * (s[i] - s[i - 1]))
            break
    return {"auc": auc, "char_severity": char_sev,
            "rel_change": rel_change, "clean": float(clean), "final": float(final)}


# --------------------------------------------------------------------------- #
#  单 severity 点
# --------------------------------------------------------------------------- #
def _metric_at_severity(wrapper, samples: list[ProbeSample], mode: str,
                        severity: float, readout: str, cfg: dict) -> tuple[Optional[float], dict]:
    metric = resolve_readout(readout)
    readout_name = metric.legacy_name
    span_stats = _empty_span_stats()

    if readout_name == "pf3":
        pf3_reduce = metric.require_reduce()
        seeds = cfg["pf3"]["num_seeds"]
        vals = []
        for s in samples:
            span_stats["n_total"] += 1
            try:
                meta = IM.pf3_curve_with_meta_from_sample(
                    wrapper, s,
                    corruption_mode=mode, num_seeds=seeds,
                    severity=severity,
                )
                if meta["curve"] is not None:
                    vals.append(pf3_reduce(meta["curve"])[metric.default_scalar])
                    _record_span_success(span_stats, meta)
                else:
                    _record_span_skip(span_stats, "pf3_no_valid_seed")
            except Exception as e:  # noqa: BLE001
                _record_span_skip(span_stats, f"pf3:{type(e).__name__}")
                logger.debug("pf3@sev fail: %s", e)
        span_stats["valid_rate"] = span_stats["n_success"] / max(span_stats["n_total"], 1)
        return (float(np.mean(vals)) if vals else None), span_stats

    elif readout_name == "bf3":
        bf3_reduce = metric.require_reduce()
        vals = []
        for s in samples:
            span_stats["n_total"] += 1
            processed, image_meta = IM.prepare_image_for_audit(wrapper, s.image)
            img = (processed if severity == 0
                   else IM.corrupt_image(processed, mode, seed=0, severity=severity))
            s_corr = replace(s, image=img)
            try:
                meta = IM.bf3_curve_with_meta_from_sample(wrapper, s_corr)
                meta["image_preprocess"] = image_meta
                vals.append(bf3_reduce(meta["curve"])["final_entropy"])
                _record_span_success(span_stats, meta)
            except Exception as e:  # noqa: BLE001
                _record_span_skip(span_stats, f"bf3:{type(e).__name__}")
                logger.debug("bf3@sev fail: %s", e)
        span_stats["valid_rate"] = span_stats["n_success"] / max(span_stats["n_total"], 1)
        return (float(np.mean(vals)) if vals else None), span_stats
    raise ValueError(f"unknown readout: {readout}")


# --------------------------------------------------------------------------- #
#  完整 decay sweep
# --------------------------------------------------------------------------- #
def run_decay_sweep(wrapper, samples: list[ProbeSample],
                    cfg: dict, model_tag: str) -> dict:
    cf2 = cfg["cf2"]
    readout = cf2.get("readout", "pf3")           # "pf3"/"bf3" or metric id
    readout_label = resolve_readout(readout).legacy_name
    out_fams: dict[str, dict] = {}

    for fam, sev_list in cf2["families"].items():
        severities = [0.0] + list(sev_list)        # 0 = 干净基线
        ys = []
        fam_span_stats = _empty_span_stats()
        for sev in severities:
            val, span_stats = _metric_at_severity(wrapper, samples, fam, sev, readout, cfg)
            _merge_span_stats(fam_span_stats, span_stats)
            ys.append(val if val is not None else float("nan"))
            logger.info("[%s] CF-2 %s sev=%.3f %s=%.4f",
                        model_tag, fam, sev, readout_label, ys[-1])
        feats = curve_features(severities, ys)
        out_fams[fam] = {
            "severities": severities,
            "curve": ys,
            "features": feats,
            "span_metadata": fam_span_stats,
        }

    return {"model": model_tag, "readout": readout_label, "families": out_fams}
