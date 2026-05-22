"""Legacy alias for PF-3 attention distance."""
from __future__ import annotations

from dataclasses import replace

from ..pf3_attention_distance import SPEC as BASE_SPEC

SPEC = replace(
    BASE_SPEC,
    metric_id="pf3_attention_distance_legacy",
    legacy_name="pf3_legacy",
    title="PF-3 Attention Distance (Legacy)",
)
