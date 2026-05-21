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
from typing import Optional

import numpy as np

from .data import ProbeSample
from . import internal_metrics as IM
from .metrics import resolve_readout

logger = logging.getLogger("lvr_eval.degradation")


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
                        severity: float, readout: str, cfg: dict) -> Optional[float]:
    metric = resolve_readout(readout)
    readout_name = metric.legacy_name

    if readout_name == "pf3":
        pf3_curve = metric.require_curve()
        pf3_reduce = metric.require_reduce()
        seeds = cfg["pf3"]["num_seeds"]
        vals = []
        for s in samples:
            try:
                c, _ = pf3_curve(wrapper, s.image, s.question,
                                 corruption_mode=mode, num_seeds=seeds,
                                 severity=(None if severity == 0 else severity))
                if c is not None:
                    vals.append(pf3_reduce(c)[metric.default_scalar])
            except Exception as e:  # noqa: BLE001
                logger.debug("pf3@sev fail: %s", e)
        return float(np.mean(vals)) if vals else None

    elif readout_name == "bf3":
        bf3_curve = metric.require_curve()
        bf3_reduce = metric.require_reduce()
        vals = []
        for s in samples:
            img = (s.image if severity == 0
                   else IM.corrupt_image(s.image, mode, seed=0, severity=severity))
            try:
                c = bf3_curve(wrapper, img, s.question)
                vals.append(bf3_reduce(c)["final_entropy"])
            except Exception as e:  # noqa: BLE001
                logger.debug("bf3@sev fail: %s", e)
        return float(np.mean(vals)) if vals else None
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
        for sev in severities:
            val = _metric_at_severity(wrapper, samples, fam, sev, readout, cfg)
            ys.append(val if val is not None else float("nan"))
            logger.info("[%s] CF-2 %s sev=%.3f %s=%.4f",
                        model_tag, fam, sev, readout_label, ys[-1])
        feats = curve_features(severities, ys)
        out_fams[fam] = {"severities": severities, "curve": ys, "features": feats}

    return {"model": model_tag, "readout": readout_label, "families": out_fams}
