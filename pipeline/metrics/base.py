"""Shared metric registry types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


CurveFn = Callable[..., object]
ReduceFn = Callable[[object], dict[str, float]]
RunFn = Callable[..., dict]


@dataclass(frozen=True)
class MetricSpec:
    """Descriptor for one first-class LVR metric."""

    metric_id: str
    legacy_name: str
    title: str
    kind: str
    curve_fn: CurveFn | None = None
    reduce_fn: ReduceFn | None = None
    default_scalar: str | None = None
    run_fn: RunFn | None = None

    def require_curve(self) -> CurveFn:
        if self.curve_fn is None:
            raise ValueError(f"{self.metric_id} does not expose a curve function")
        return self.curve_fn

    def require_reduce(self) -> ReduceFn:
        if self.reduce_fn is None:
            raise ValueError(f"{self.metric_id} does not expose a reduce function")
        return self.reduce_fn

    def require_run(self) -> RunFn:
        if self.run_fn is None:
            raise ValueError(f"{self.metric_id} does not expose a run function")
        return self.run_fn

