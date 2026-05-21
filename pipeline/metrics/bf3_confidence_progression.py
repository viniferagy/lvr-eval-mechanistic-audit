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


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="BF-3 Confidence Progression",
    kind="internal_curve",
    curve_fn=curve,
    reduce_fn=reduce_curve,
    default_scalar=DEFAULT_SCALAR,
)

