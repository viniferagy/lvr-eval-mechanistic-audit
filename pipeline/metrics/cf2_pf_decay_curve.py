"""CF-2: corruption severity decay curve metric entrypoint."""
from __future__ import annotations

from .base import MetricSpec


METRIC_ID = "cf2_pf_decay_curve"
LEGACY_NAME = "cf2"


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    from ..degradation import run_decay_sweep

    return run_decay_sweep(wrapper, samples, cfg, model_tag)


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="CF-2 PF Decay Curve",
    kind="sweep",
    run_fn=run,
)

