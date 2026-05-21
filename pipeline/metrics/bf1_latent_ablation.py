"""BF-1: targeted span ablation metric entrypoint."""
from __future__ import annotations

from .base import MetricSpec


METRIC_ID = "bf1_latent_ablation"
LEGACY_NAME = "bf1"


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    from ..ablation import run_targeted_ablation_sweep

    return run_targeted_ablation_sweep(wrapper, samples, cfg, model_tag)


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="BF-1 Targeted Span Ablation",
    kind="sweep",
    run_fn=run,
)
