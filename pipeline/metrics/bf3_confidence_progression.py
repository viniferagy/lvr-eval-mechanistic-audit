"""BF-3: confidence progression over decoder layers."""
from __future__ import annotations

import numpy as np

from .. import internal_metrics as IM
from .base import MetricSpec


METRIC_ID = "bf3_confidence_progression"
LEGACY_NAME = "bf3"
DEFAULT_SCALAR = "early_to_late_drop"


def curve(wrapper, image, question: str) -> np.ndarray:
    return IM.bf3_curve(wrapper, image, question)


def reduce_curve(values) -> dict[str, float]:
    return IM.bf3_reduce(np.asarray(values))


def scalar_from_curve(values, scalar: str = DEFAULT_SCALAR) -> float:
    return reduce_curve(values)[scalar]


def _empty_span_stats(n_total: int) -> dict:
    return {
        "n_total": int(n_total),
        "n_success": 0,
        "n_skipped": 0,
        "skip_reasons": {},
        "query_target_counts": {},
        "valid_rate": 0.0,
        "examples": [],
    }


def _record_success(stats: dict, meta: dict):
    stats["n_success"] += 1
    kind = meta.get("query_target_kind", "unknown")
    stats["query_target_counts"][kind] = stats["query_target_counts"].get(kind, 0) + 1
    if len(stats["examples"]) < 5:
        stats["examples"].append({
            "query_target_kind": kind,
            "query_span": meta.get("query_span"),
            "image_span": meta.get("image_span"),
            "adapter_notes": meta.get("adapter_notes", {}),
        })


def _record_skip(stats: dict, reason: str):
    stats["n_skipped"] += 1
    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    curves = []
    per_sample = []
    span_stats = _empty_span_stats(len(samples))

    for s in samples:
        try:
            meta = IM.bf3_curve_with_meta_from_sample(wrapper, s)
            curve_values = np.asarray(meta["curve"], dtype=float)
            curves.append(curve_values)
            _record_success(span_stats, meta)
            per_sample.append({
                "id": s.id,
                "curve": curve_values.tolist(),
                "reduction": reduce_curve(curve_values),
                "query_target_kind": meta.get("query_target_kind"),
                "query_span": meta.get("query_span"),
                "image_span": meta.get("image_span"),
                "adapter_notes": meta.get("adapter_notes", {}),
            })
        except Exception as exc:  # noqa: BLE001
            reason = type(exc).__name__
            _record_skip(span_stats, reason)
            per_sample.append({"id": s.id, "error": repr(exc)})

    span_stats["valid_rate"] = span_stats["n_success"] / max(span_stats["n_total"], 1)
    aggregate_curve = np.mean(np.stack(curves), axis=0) if curves else None
    return {
        "model": model_tag,
        "curve": aggregate_curve.tolist() if aggregate_curve is not None else None,
        "reduction": reduce_curve(aggregate_curve) if aggregate_curve is not None else None,
        "samples": per_sample,
        "span_metadata": span_stats,
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="BF-3 Confidence Progression",
    kind="internal_curve",
    curve_fn=curve,
    reduce_fn=reduce_curve,
    default_scalar=DEFAULT_SCALAR,
    run_fn=run,
)
