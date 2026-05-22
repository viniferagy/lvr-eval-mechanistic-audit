"""Legacy alias for BF-1 targeted span ablation."""
from __future__ import annotations

from dataclasses import replace

from ..bf1_latent_ablation import SPEC as BASE_SPEC

SPEC = replace(
    BASE_SPEC,
    metric_id="bf1_latent_ablation_legacy",
    legacy_name="bf1_legacy",
    title="BF-1 Targeted Span Ablation (Legacy)",
)
