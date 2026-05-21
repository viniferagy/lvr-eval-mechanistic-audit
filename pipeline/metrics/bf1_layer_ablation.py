"""BF-1 legacy whole-layer ablation sweep metric entrypoint."""
from __future__ import annotations

from .base import MetricSpec


METRIC_ID = "bf1_layer_ablation"
LEGACY_NAME = "bf1_layer"


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    from ..ablation import run_ablation_sweep

    return run_ablation_sweep(wrapper, samples, cfg, model_tag)


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="BF-1 Legacy Whole-Layer Ablation",
    kind="sweep",
    run_fn=run,
)
