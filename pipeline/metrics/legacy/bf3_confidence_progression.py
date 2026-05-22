"""Legacy alias for BF-3 confidence progression."""
from __future__ import annotations

from dataclasses import replace

from ..bf3_confidence_progression import SPEC as BASE_SPEC

SPEC = replace(
    BASE_SPEC,
    metric_id="bf3_confidence_progression_legacy",
    legacy_name="bf3_legacy",
    title="BF-3 Confidence Progression (Legacy)",
)
