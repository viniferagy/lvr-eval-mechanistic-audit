"""Legacy alias for approximate LVR generation trace."""
from __future__ import annotations

from dataclasses import replace

from ..lvr_generation_trace import SPEC as BASE_SPEC

SPEC = replace(
    BASE_SPEC,
    metric_id="lvr_generation_trace_legacy",
    legacy_name="lvr_trace_legacy",
    title="LVR Generation Trace (Legacy Approximate)",
)
