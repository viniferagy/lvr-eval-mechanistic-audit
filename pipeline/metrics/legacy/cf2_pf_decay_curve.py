"""Legacy alias for CF-2 PF decay curve."""
from __future__ import annotations

from dataclasses import replace

from ..cf2_pf_decay_curve import SPEC as BASE_SPEC

SPEC = replace(
    BASE_SPEC,
    metric_id="cf2_pf_decay_curve_legacy",
    legacy_name="cf2_legacy",
    title="CF-2 PF Decay Curve (Legacy)",
)
