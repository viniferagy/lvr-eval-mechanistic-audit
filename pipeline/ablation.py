"""
pipeline/ablation.py
====================
BF-1: Latent Ablation —— 与 BF-3 / PF-3 联动版。

机制: 在 decoder layer i 挂 forward hook 扰动其 hidden state, 然后在 hook
仍生效时, 用 metrics registry 重新计算 BF-3 entropy curve 与 PF-3 attention
KL curve。对比“无消融 baseline” -> 得到“消融第 i 层后内部指标怎么变”。

ablation mode
-------------
identity : h <- 输入 (移除该层对 residual 的增量, 最干净的因果消融, 推荐)
zero     : h <- 0
mean     : h <- mean(h)
noise    : h <- h + N(0, s)

读出标量(每层一个 Δ):
  BF-3: early_to_late_drop  (confidence sharpening)
  PF-3: mean_kl             (modality dependence)
同时保留每个消融配置下重算出的完整 curve, 供作图。
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .data import ProbeSample
from .metrics import resolve_readout

logger = logging.getLogger("lvr_eval.ablation")


# --------------------------------------------------------------------------- #
#  hook
# --------------------------------------------------------------------------- #
def _make_hook(mode: str, noise_std: float = 0.1):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        if mode == "zero":
            h_new = torch.zeros_like(h)
        elif mode == "identity":
            h_new = args[0]
        elif mode == "mean":
            h_new = h.mean(dim=(0, 1), keepdim=True).expand_as(h).clone()
        elif mode == "noise":
            h_new = h + torch.randn_like(h) * noise_std
        else:
            raise ValueError(f"未知 ablation mode: {mode}")
        return h_new if rest is None else (h_new,) + rest
    return hook


@contextmanager
def ablate_layer(wrapper, layer_idx: int, mode: str, noise_std: float = 0.1):
    """layer_idx<0 = 不消融(baseline)。"""
    handle = None
    if layer_idx >= 0:
        handle = wrapper.layers[layer_idx].register_forward_hook(_make_hook(mode, noise_std))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


# --------------------------------------------------------------------------- #
#  在当前(可能被消融的)模型上, 算 probe set 的平均 BF-3 / PF-3 curve
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_internal(wrapper, samples: list[ProbeSample], cfg: dict) -> dict:
    bf1 = cfg["bf1"]
    do_bf3 = bf1.get("readout_bf3", True)
    do_pf3 = bf1.get("readout_pf3", True)
    pf3_mode = cfg["pf3"]["corruption_mode"]
    pf3_seeds = cfg["pf3"]["num_seeds"]
    bf3_metric = resolve_readout("bf3")
    pf3_metric = resolve_readout("pf3")
    bf3_curve = bf3_metric.require_curve()
    bf3_reduce = bf3_metric.require_reduce()
    pf3_curve = pf3_metric.require_curve()
    pf3_reduce = pf3_metric.require_reduce()

    bf3_curves, pf3_curves = [], []
    for s in samples:
        if do_bf3:
            try:
                bf3_curves.append(bf3_curve(wrapper, s.image, s.question))
            except Exception as e:  # noqa: BLE001
                logger.debug("bf3 sample fail: %s", e)
        if do_pf3:
            try:
                c, _ = pf3_curve(wrapper, s.image, s.question,
                                 corruption_mode=pf3_mode, num_seeds=pf3_seeds)
                if c is not None:
                    pf3_curves.append(c)
            except Exception as e:  # noqa: BLE001
                logger.debug("pf3 sample fail: %s", e)

    out: dict = {}
    if bf3_curves:
        m = np.mean(np.stack(bf3_curves), axis=0)
        out["bf3_curve"] = m.tolist()
        out["bf3"] = bf3_reduce(m)
    if pf3_curves:
        m = np.mean(np.stack(pf3_curves), axis=0)
        out["pf3_curve"] = m.tolist()
        out["pf3"] = pf3_reduce(m)
    return out


# --------------------------------------------------------------------------- #
#  完整 sweep
# --------------------------------------------------------------------------- #
def run_ablation_sweep(wrapper, samples: list[ProbeSample],
                       cfg: dict, model_tag: str) -> dict:
    bf1 = cfg["bf1"]
    mode = bf1["mode"]
    noise_std = bf1.get("noise_std", 0.1)
    bf3_key = bf1.get("bf3_scalar", "early_to_late_drop")
    pf3_key = bf1.get("pf3_scalar", "mean_kl")

    logger.info("[%s] BF-1 baseline (no ablation) ...", model_tag)
    with ablate_layer(wrapper, -1, mode):
        baseline = probe_internal(wrapper, samples, cfg)
    logger.info("[%s] baseline bf3=%s pf3=%s",
                model_tag, baseline.get("bf3"), baseline.get("pf3"))

    layer_ids = bf1.get("layers") or list(range(wrapper.n_layers))
    results = []
    for li in layer_ids:
        with ablate_layer(wrapper, li, mode, noise_std):
            m = probe_internal(wrapper, samples, cfg)
        rec = {"layer": li, "bf3": m.get("bf3"), "pf3": m.get("pf3"),
               "bf3_curve": m.get("bf3_curve"), "pf3_curve": m.get("pf3_curve"),
               "delta": {}}
        if "bf3" in baseline and "bf3" in m:
            rec["delta"]["bf3"] = baseline["bf3"][bf3_key] - m["bf3"][bf3_key]
        if "pf3" in baseline and "pf3" in m:
            rec["delta"]["pf3"] = baseline["pf3"][pf3_key] - m["pf3"][pf3_key]
        results.append(rec)
        logger.info("[%s] layer %02d/%d  Δbf3(%s)=%s  Δpf3(%s)=%s",
                    model_tag, li, wrapper.n_layers, bf3_key,
                    rec["delta"].get("bf3"), pf3_key, rec["delta"].get("pf3"))

    return {"model": model_tag, "mode": mode,
            "n_layers": wrapper.n_layers,
            "bf3_scalar": bf3_key, "pf3_scalar": pf3_key,
            "baseline": baseline, "layers": results}
