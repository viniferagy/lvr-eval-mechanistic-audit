"""Legacy metric specs frozen as regression references."""
from __future__ import annotations

from .bf1_latent_ablation import SPEC as BF1_LEGACY_SPEC
from .bf1_layer_ablation import SPEC as BF1_LAYER_LEGACY_SPEC
from .bf3_confidence_progression import SPEC as BF3_LEGACY_SPEC
from .cf2_pf_decay_curve import SPEC as CF2_LEGACY_SPEC
from .lvr_generation_trace import SPEC as LVR_TRACE_LEGACY_SPEC
from .pf3_attention_distance import SPEC as PF3_LEGACY_SPEC

LEGACY_SPECS = [
    BF3_LEGACY_SPEC,
    PF3_LEGACY_SPEC,
    BF1_LEGACY_SPEC,
    BF1_LAYER_LEGACY_SPEC,
    CF2_LEGACY_SPEC,
    LVR_TRACE_LEGACY_SPEC,
]

__all__ = [
    "BF1_LAYER_LEGACY_SPEC",
    "BF1_LEGACY_SPEC",
    "BF3_LEGACY_SPEC",
    "CF2_LEGACY_SPEC",
    "LEGACY_SPECS",
    "LVR_TRACE_LEGACY_SPEC",
    "PF3_LEGACY_SPEC",
]
