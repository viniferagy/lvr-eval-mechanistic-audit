"""BF-Conf calibrated confidence progression."""
from __future__ import annotations

import numpy as np

from ... import internal_metrics as IM
from . import trace_latent as TL
from .bf_patch_answer_transfer import _answer_token_id, _model_forward
from ..base import MetricSpec


METRIC_ID = "bf_conf_calibrated_progression"
LEGACY_NAME = "bf_conf"
DEFAULT_SCALAR = "gold_logit_slope"


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "scalars": [
            "early_to_late_drop",
            "final_entropy",
            "mean_entropy",
            "gold_logit_slope",
            "text_only_control_rate",
        ],
        "status": "runnable_v0",
        "trace_latent_optional": True,
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("bf_conf", {}) or {}


def _safe_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def _answer_logits_by_layer(wrapper, inputs, query_span, answer_token_id: int) -> list[float]:
    if answer_token_id is None:
        return []
    last_pos = query_span.end - 1
    captured: list[object | None] = [None] * wrapper.n_layers
    handles = []

    def make_hook(idx: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden[0, last_pos, :].detach().cpu()
            return output
        return hook

    for idx, layer in enumerate(wrapper.layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))
    try:
        _model_forward(wrapper, inputs)
    finally:
        for handle in handles:
            handle.remove()

    values = []
    for hidden in captured:
        if hidden is None:
            continue
        h = IM._to_model_device(hidden, wrapper)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        normed = wrapper.final_norm(h) if wrapper.final_norm is not None else h
        logits = wrapper.lm_head(normed)
        values.append(float(logits[0, int(answer_token_id)].detach().float().cpu().item()))
    return values


def _text_only_control(wrapper, sample, enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False, "available": False, "reason": "disabled"}
    if not hasattr(wrapper.adapter, "build_inputs"):
        return {"enabled": True, "available": False, "reason": "adapter_no_build_inputs"}
    try:
        inputs = wrapper.adapter.build_inputs(wrapper, sample.image, sample.question)
        spans = wrapper.adapter.get_spans(wrapper, inputs, None)
        curve = IM.bf3_curve_from_inputs(wrapper, inputs, spans.preferred_query_span())
        return {
            "enabled": True,
            "available": True,
            "reduction": IM.bf3_reduce(np.asarray(curve, dtype=float)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "available": False, "reason": repr(exc)}


def _aggregate(samples: list[dict]) -> dict | None:
    valid = [s for s in samples if isinstance(s.get("reduction"), dict)]
    if not valid:
        return None
    keys = sorted({key for s in valid for key, value in s["reduction"].items() if value is not None})
    out = {}
    for key in keys:
        vals = [float(s["reduction"][key]) for s in valid if s["reduction"].get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    out["n"] = len(valid)
    controls = [s.get("text_only_control", {}) for s in valid]
    out["text_only_control_rate"] = (
        sum(1 for c in controls if c.get("available")) / len(controls)
        if controls else None
    )
    return out


def _run_trace_latent(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    per_sample = []
    for sample in samples:
        try:
            trace = TL.generate_trace(wrapper, sample, cfg, label="bf_conf")
            answer_logits = TL.latent_answer_logit_curve(wrapper, trace, sample.answer)
            if not answer_logits and sample.counterfactual_answer is not None:
                answer_logits = TL.latent_answer_logit_curve(wrapper, trace, sample.counterfactual_answer)
            curve = answer_logits or TL.latent_norm_curve(trace)
            reduction = {
                "early_to_late_drop": (
                    float(curve[0] - curve[-1]) if len(curve) >= 2 else None
                ),
                "final_entropy": None,
                "mean_entropy": float(np.mean(curve)) if curve else None,
                "gold_logit_slope": TL.safe_slope(answer_logits),
            }
            per_sample.append({
                "id": sample.id,
                "curve": [float(v) for v in curve],
                "answer_token_id": TL.first_token_id(wrapper, sample.answer),
                "gold_logit_by_layer": [float(v) for v in answer_logits],
                "reduction": reduction,
                "query_target_kind": "generation_trace_latent_state",
                "query_span": None,
                "image_span": None,
                "trace_quality": trace.get("trace_quality"),
                "n_lvr_mode_steps": int(trace.get("n_lvr_mode_steps") or 0),
                "n_hidden_feedback_steps": int(trace.get("n_hidden_feedback_steps") or 0),
                "n_captured_latent_states": int(trace.get("n_captured_latent_states") or 0),
                "captured_state_shapes": TL.state_shape_metadata(trace),
                "text_only_control": {
                    "enabled": False,
                    "available": False,
                    "reason": "trace_latent_generation_path",
                },
            })
        except Exception as exc:  # noqa: BLE001
            per_sample.append({"id": sample.id, "error": repr(exc)})
    aggregate_curve = (
        np.mean(np.stack([np.asarray(s["curve"], dtype=float) for s in per_sample if s.get("curve")]), axis=0)
        if any(s.get("curve") for s in per_sample)
        else None
    )
    return {
        "model": model_tag,
        "schema": build_schema(),
        "curve": aggregate_curve.tolist() if aggregate_curve is not None else None,
        "samples": per_sample,
        "reduction": _aggregate(per_sample),
        "config": {"text_only_control": False, "trace_latent": True},
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    if TL.enabled(cfg, METRIC_ID):
        return _run_trace_latent(wrapper, samples, cfg, model_tag)

    local = _cfg(cfg)
    text_only_enabled = bool(local.get("text_only_control", True))
    per_sample = []
    for sample in samples:
        try:
            image, image_meta = IM.prepare_image_for_audit(wrapper, sample.image)
            prepared = sample.__class__(**{**sample.__dict__, "image": image})
            inputs = wrapper.build_inputs_from_sample(prepared)
            spans = wrapper.adapter.get_spans(wrapper, inputs, None)
            query_span = spans.preferred_query_span()
            curve = IM.bf3_curve_from_inputs(wrapper, inputs, query_span)
            reduction = IM.bf3_reduce(np.asarray(curve, dtype=float))
            answer_token_id = _answer_token_id(wrapper, sample.answer)
            answer_logits = _answer_logits_by_layer(wrapper, inputs, query_span, answer_token_id)
            gold_slope = _safe_slope(answer_logits)
            reduction["gold_logit_slope"] = gold_slope
            per_sample.append({
                "id": sample.id,
                "curve": np.asarray(curve, dtype=float).tolist(),
                "answer_token_id": answer_token_id,
                "gold_logit_by_layer": answer_logits,
                "reduction": reduction,
                "query_target_kind": query_span.kind,
                "query_span": [query_span.start, query_span.end],
                "image_span": [spans.image_tokens.start, spans.image_tokens.end],
                "image_preprocess": image_meta,
                "text_only_control": _text_only_control(wrapper, prepared, text_only_enabled),
            })
        except Exception as exc:  # noqa: BLE001
            per_sample.append({"id": sample.id, "error": repr(exc)})

    aggregate_curve = (
        np.mean(np.stack([np.asarray(s["curve"], dtype=float) for s in per_sample if s.get("curve")]), axis=0)
        if any(s.get("curve") for s in per_sample)
        else None
    )
    return {
        "model": model_tag,
        "schema": build_schema(),
        "curve": aggregate_curve.tolist() if aggregate_curve is not None else None,
        "samples": per_sample,
        "reduction": _aggregate(per_sample),
        "config": {"text_only_control": text_only_enabled},
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="BF-Conf Calibrated Progression",
    kind="internal_curve",
    default_scalar=DEFAULT_SCALAR,
    run_fn=run,
)
