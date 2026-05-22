"""Legacy alias for BF-1 whole-layer ablation."""
from __future__ import annotations

from dataclasses import replace

from ..bf1_layer_ablation import SPEC as BASE_SPEC

SPEC = replace(
    BASE_SPEC,
    metric_id="bf1_layer_ablation_legacy",
    legacy_name="bf1_layer_legacy",
    title="BF-1 Legacy Whole-Layer Ablation (Legacy)",
)
