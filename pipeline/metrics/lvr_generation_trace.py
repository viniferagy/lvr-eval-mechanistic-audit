"""LVR inference-time generation trace metric."""
from __future__ import annotations

from .base import MetricSpec


METRIC_ID = "lvr_generation_trace"
LEGACY_NAME = "lvr_trace"


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    audit_cfg = cfg.get("audit", {})
    out = []
    for s in samples:
        try:
            trace = wrapper.adapter.generate_with_trace(
                wrapper,
                s.image,
                s.question,
                decoding_strategy=audit_cfg.get("lvr_decoding_strategy", "steps"),
                lvr_steps=audit_cfg.get("lvr_steps", 16),
            )
            out.append({
                "id": s.id,
                "generated_text": trace.get("generated_text"),
                "lvr_generated_positions": trace.get("lvr_generated_positions"),
                "lvr_token_positions": trace.get("lvr_token_positions"),
                "lvr_latent_end_positions": trace.get("lvr_latent_end_positions"),
                "lvr_block_spans": trace.get("lvr_block_spans"),
                "unexpected_lvr_inner_positions": trace.get("unexpected_lvr_inner_positions"),
                "trace_quality": trace.get("trace_quality"),
                "notes": trace.get("notes"),
            })
        except Exception as exc:  # noqa: BLE001
            out.append({"id": s.id, "error": repr(exc)})
    return {"model": model_tag, "samples": out}


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="LVR Generation Trace",
    kind="trace",
    run_fn=run,
)
