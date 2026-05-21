"""PF-3: intact-vs-corrupted attention distance over decoder layers."""
from __future__ import annotations

import numpy as np

from .. import internal_metrics as IM
from .base import MetricSpec


METRIC_ID = "pf3_attention_distance"
LEGACY_NAME = "pf3"
DEFAULT_SCALAR = "mean_kl"


def curve(wrapper, image, question: str, *,
          corruption_mode: str = "mask_50pct",
          num_seeds: int = 3,
          severity: float | None = None) -> tuple[np.ndarray | None, int]:
    return IM.pf3_curve(
        wrapper,
        image,
        question,
        corruption_mode=corruption_mode,
        num_seeds=num_seeds,
        severity=severity,
    )


def reduce_curve(values) -> dict[str, float]:
    return IM.pf3_reduce(np.asarray(values))


def scalar_from_curve(values, scalar: str = DEFAULT_SCALAR) -> float:
    return reduce_curve(values)[scalar]


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="PF-3 Attention Distance",
    kind="internal_curve",
    curve_fn=curve,
    reduce_fn=reduce_curve,
    default_scalar=DEFAULT_SCALAR,
)

