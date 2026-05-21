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
def _make_hook(mode: str, noise_std: float = 0.1, token_span=None):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        target_slice = slice(None) if token_span is None else token_span.as_slice()
        h_new = h.clone()
        if mode == "zero":
            replacement = torch.zeros_like(h[:, target_slice, :])
        elif mode == "identity":
            replacement = args[0][:, target_slice, :]
        elif mode == "mean":
            replacement = (
                h[:, target_slice, :]
                .mean(dim=(0, 1), keepdim=True)
                .expand_as(h[:, target_slice, :])
                .clone()
            )
        elif mode == "noise":
            replacement = h[:, target_slice, :] + torch.randn_like(h[:, target_slice, :]) * noise_std
        else:
            raise ValueError(f"未知 ablation mode: {mode}")
        h_new[:, target_slice, :] = replacement
        return h_new if rest is None else (h_new,) + rest
    return hook


@contextmanager
def ablate_layer(wrapper, layer_idx: int, mode: str, noise_std: float = 0.1,
                 token_span=None):
    """layer_idx<0 = 不消融(baseline)。"""
    handle = None
    if layer_idx >= 0:
        handle = wrapper.layers[layer_idx].register_forward_hook(
            _make_hook(mode, noise_std, token_span=token_span)
        )
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
    bf3_reduce = bf3_metric.require_reduce()
    pf3_reduce = pf3_metric.require_reduce()

    bf3_curves, pf3_curves = [], []
    span_records = []
    counters = {"n_total": 0, "n_success": 0, "n_skipped": 0,
                "skip_reasons": {}, "query_target_counts": {}}

    def record_success(meta: dict):
        counters["n_success"] += 1
        kind = meta.get("query_target_kind", "unknown")
        counters["query_target_counts"][kind] = (
            counters["query_target_counts"].get(kind, 0) + 1
        )
        span_records.append({
            "query_target_kind": kind,
            "query_span": meta.get("query_span"),
            "image_span": meta.get("image_span"),
            "adapter_notes": meta.get("adapter_notes", {}),
        })

    def record_skip(reason: str):
        counters["n_skipped"] += 1
        counters["skip_reasons"][reason] = counters["skip_reasons"].get(reason, 0) + 1

    for s in samples:
        counters["n_total"] += 1
        if do_bf3:
            try:
                from . import internal_metrics as IM

                meta = IM.bf3_curve_with_meta_from_sample(wrapper, s)
                bf3_curves.append(meta["curve"])
                record_success(meta)
            except Exception as e:  # noqa: BLE001
                record_skip(f"bf3:{type(e).__name__}")
                logger.debug("bf3 sample fail: %s", e)
        if do_pf3:
            try:
                from . import internal_metrics as IM

                meta = IM.pf3_curve_with_meta_from_sample(
                    wrapper, s,
                    corruption_mode=pf3_mode, num_seeds=pf3_seeds,
                )
                if meta["curve"] is not None:
                    pf3_curves.append(meta["curve"])
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
    out["span_metadata"] = {
        **counters,
        "valid_rate": counters["n_success"] / max(counters["n_total"], 1),
        "examples": span_records[:5],
    }
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
            "ablation_target": "whole_layer_legacy",
            "bf3_scalar": bf3_key, "pf3_scalar": pf3_key,
            "baseline": baseline, "layers": results}


@torch.no_grad()
def run_targeted_ablation_sweep(wrapper, samples: list[ProbeSample],
                                cfg: dict, model_tag: str) -> dict:
    """Targeted BF-1: ablate adapter query span and read BF-3/PF-3."""
    from . import internal_metrics as IM

    bf1 = cfg["bf1"]
    mode = bf1["mode"]
    noise_std = bf1.get("noise_std", 0.1)
    do_pf3 = bool(bf1.get("readout_pf3", True))
    bf3_key = bf1.get("bf3_scalar", "early_to_late_drop")
    pf3_key = bf1.get("pf3_scalar", "mean_kl")
    pf3_mode = cfg.get("pf3", {}).get("corruption_mode", "mask_50pct")
    pf3_seeds = int(cfg.get("pf3", {}).get("num_seeds", 3))
    layer_ids = bf1.get("layers") or list(range(wrapper.n_layers))

    prepared = []
    baseline_pf3_curves = []
    skip_reasons: dict[str, int] = {}
    query_target_counts: dict[str, int] = {}
    for s in samples:
        try:
            inputs = wrapper.build_inputs_from_sample(s)
            out = wrapper.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            spans = wrapper.adapter.get_spans(wrapper, inputs, out)
            query_span = spans.preferred_query_span()
            curve = IM.bf3_curve_from_inputs(wrapper, inputs, query_span, outputs=out)
            meta = IM._span_payload(spans, query_span)  # internal JSON-safe helper
            prepared.append({
                "id": s.id,
                "sample": s,
                "inputs": inputs,
                "query_span": query_span,
                "baseline_curve": curve,
                "span_meta": meta,
            })
            kind = meta.get("query_target_kind", "unknown")
            query_target_counts[kind] = query_target_counts.get(kind, 0) + 1
        except Exception as exc:  # noqa: BLE001
            reason = type(exc).__name__
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            logger.debug("targeted bf1 prepare fail: %s", exc)
            continue

        if do_pf3:
            try:
                pf3_meta = IM.pf3_curve_with_meta_from_sample(
                    wrapper,
                    s,
                    corruption_mode=pf3_mode,
                    num_seeds=pf3_seeds,
                )
                if pf3_meta["curve"] is not None:
                    baseline_pf3_curves.append(pf3_meta["curve"])
            except Exception as exc:  # noqa: BLE001
                logger.debug("targeted bf1 baseline pf3 fail: %s", exc)

    if prepared:
        baseline_curve = np.mean(
            np.stack([rec["baseline_curve"] for rec in prepared]),
            axis=0,
        )
        baseline = {
            "bf3_curve": baseline_curve.tolist(),
            "bf3": IM.bf3_reduce(baseline_curve),
            "span_metadata": {
                "n_total": len(samples),
                "n_success": len(prepared),
                "n_skipped": len(samples) - len(prepared),
                "skip_reasons": skip_reasons,
                "query_target_counts": query_target_counts,
                "valid_rate": len(prepared) / max(len(samples), 1),
                "examples": [rec["span_meta"] for rec in prepared[:5]],
            },
        }
        if baseline_pf3_curves:
            baseline_pf3_curve = np.mean(np.stack(baseline_pf3_curves), axis=0)
            baseline["pf3_curve"] = baseline_pf3_curve.tolist()
            baseline["pf3"] = IM.pf3_reduce(baseline_pf3_curve)
    else:
        baseline = {
            "span_metadata": {
                "n_total": len(samples),
                "n_success": 0,
                "n_skipped": len(samples),
                "skip_reasons": skip_reasons,
                "query_target_counts": query_target_counts,
                "valid_rate": 0.0,
                "examples": [],
            },
        }

    results = []
    for li in layer_ids:
        bf3_curves = []
        pf3_curves = []
        for rec in prepared:
            try:
                with ablate_layer(
                    wrapper,
                    li,
                    mode,
                    noise_std,
                    token_span=rec["query_span"],
                ):
                    out = wrapper.model(
                        **rec["inputs"],
                        output_hidden_states=True,
                        output_attentions=False,
                        return_dict=True,
                    )
                bf3_curves.append(
                    IM.bf3_curve_from_inputs(
                        wrapper,
                        rec["inputs"],
                        rec["query_span"],
                        outputs=out,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("targeted bf1 layer %s bf3 fail: %s", li, exc)

            if do_pf3:
                try:
                    with ablate_layer(
                        wrapper,
                        li,
                        mode,
                        noise_std,
                        token_span=rec["query_span"],
                    ):
                        pf3_meta = IM.pf3_curve_with_meta_from_sample(
                            wrapper,
                            rec["sample"],
                            corruption_mode=pf3_mode,
                            num_seeds=pf3_seeds,
                        )
                    if pf3_meta["curve"] is not None:
                        pf3_curves.append(pf3_meta["curve"])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("targeted bf1 layer %s pf3 fail: %s", li, exc)

        rec_out = {
            "layer": li,
            "bf3": None,
            "pf3": None,
            "bf3_curve": None,
            "pf3_curve": None,
            "delta": {},
        }
        if bf3_curves and baseline.get("bf3"):
            m = np.mean(np.stack(bf3_curves), axis=0)
            rec_out["bf3_curve"] = m.tolist()
            rec_out["bf3"] = IM.bf3_reduce(m)
            rec_out["delta"]["bf3"] = baseline["bf3"][bf3_key] - rec_out["bf3"][bf3_key]
        if pf3_curves and baseline.get("pf3"):
            m = np.mean(np.stack(pf3_curves), axis=0)
            rec_out["pf3_curve"] = m.tolist()
            rec_out["pf3"] = IM.pf3_reduce(m)
            rec_out["delta"]["pf3"] = baseline["pf3"][pf3_key] - rec_out["pf3"][pf3_key]
        results.append(rec_out)
        logger.info("[%s] targeted layer %02d/%d Δbf3(%s)=%s Δpf3(%s)=%s",
                    model_tag, li, wrapper.n_layers, bf3_key,
                    rec_out["delta"].get("bf3"), pf3_key,
                    rec_out["delta"].get("pf3"))

    return {
        "model": model_tag,
        "mode": mode,
        "n_layers": wrapper.n_layers,
        "ablation_target": "adapter_preferred_query_span",
        "readout": "bf3_pf3" if do_pf3 else "bf3",
        "bf3_scalar": bf3_key,
        "pf3_scalar": pf3_key,
        "baseline": baseline,
        "layers": results,
    }
