"""Metric registry and first-class LVR metric entrypoints."""
from __future__ import annotations

from .base import MetricSpec
from .output_accuracy import exact_match
from .registry import (
    get_metric,
    list_metrics,
    list_runnable_metrics,
    normalize_metric_id,
    resolve_readout,
    run_metric,
)

__all__ = [
    "MetricSpec",
    "exact_match",
    "get_metric",
    "list_metrics",
    "list_runnable_metrics",
    "normalize_metric_id",
    "resolve_readout",
    "run_metric",
]
