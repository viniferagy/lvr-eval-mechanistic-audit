"""First-class metric registry."""
from __future__ import annotations

from .bf1_layer_ablation import SPEC as BF1_LAYER_SPEC
from .base import MetricSpec
from .bf1_latent_ablation import SPEC as BF1_SPEC
from .bf3_confidence_progression import SPEC as BF3_SPEC
from .cf2_pf_decay_curve import SPEC as CF2_SPEC
from .legacy import LEGACY_SPECS
from .lvr_generation_trace import SPEC as LVR_TRACE_SPEC
from .pf3_attention_distance import SPEC as PF3_SPEC
from .v2 import V2_SPECS


_SPECS = {
    BF3_SPEC.metric_id: BF3_SPEC,
    PF3_SPEC.metric_id: PF3_SPEC,
    BF1_SPEC.metric_id: BF1_SPEC,
    BF1_LAYER_SPEC.metric_id: BF1_LAYER_SPEC,
    CF2_SPEC.metric_id: CF2_SPEC,
    LVR_TRACE_SPEC.metric_id: LVR_TRACE_SPEC,
}
_SPECS.update({spec.metric_id: spec for spec in LEGACY_SPECS})
_SPECS.update({spec.metric_id: spec for spec in V2_SPECS})

_ALIASES = {
    spec.legacy_name: spec.metric_id
    for spec in _SPECS.values()
}
_ALIASES.update({
    "bf3_confidence_progression": BF3_SPEC.metric_id,
    "pf3_attention_distance": PF3_SPEC.metric_id,
    "bf1_latent_ablation": BF1_SPEC.metric_id,
    "bf1_targeted_ablation": BF1_SPEC.metric_id,
    "bf1_layer_ablation": BF1_LAYER_SPEC.metric_id,
    "cf2_pf_decay_curve": CF2_SPEC.metric_id,
    "lvr_generation_trace": LVR_TRACE_SPEC.metric_id,
    "bf3_legacy": "bf3_confidence_progression_legacy",
    "pf3_legacy": "pf3_attention_distance_legacy",
    "bf1_legacy": "bf1_latent_ablation_legacy",
    "bf1_layer_legacy": "bf1_layer_ablation_legacy",
    "cf2_legacy": "cf2_pf_decay_curve_legacy",
    "lvr_trace_legacy": "lvr_generation_trace_legacy",
    "pf_a": "pf_a_corruption_selectivity",
    "bf_patch": "bf_patch_answer_transfer",
    "bf_swap": "bf_swap_latent_replacement",
    "bf_conf": "bf_conf_calibrated_progression",
    "cf_stage": "cf_stage_decay",
    "pf_b": "pf_b_patch_alignment",
})


def normalize_metric_id(name: str) -> str:
    key = str(name).strip()
    try:
        return _ALIASES[key]
    except KeyError:
        if key in _SPECS:
            return key
        raise KeyError(f"unknown metric: {name}") from None


def get_metric(name: str) -> MetricSpec:
    return _SPECS[normalize_metric_id(name)]


def list_metrics() -> list[MetricSpec]:
    return list(_SPECS.values())


def list_runnable_metrics() -> list[MetricSpec]:
    return [spec for spec in _SPECS.values() if spec.run_fn is not None]


def resolve_readout(name: str) -> MetricSpec:
    spec = get_metric(name)
    if spec.kind != "internal_curve":
        raise ValueError(f"{name} is not an internal curve readout")
    return spec


def run_metric(name: str, wrapper, samples, cfg: dict, model_tag: str) -> dict:
    return get_metric(name).require_run()(wrapper, samples, cfg, model_tag)
