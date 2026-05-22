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
                "legacy_trace_quality": trace.get("legacy_trace_quality"),
                "sampled_layers": trace.get("sampled_layers"),
                "mode": trace.get("mode"),
                "last_hidden_state_shape": trace.get("last_hidden_state_shape"),
                "n_lvr_mode_steps": trace.get("n_lvr_mode_steps"),
                "n_hidden_feedback_steps": trace.get("n_hidden_feedback_steps"),
                "n_embed_calls": trace.get("n_embed_calls"),
                "lm_head_called": trace.get("lm_head_called"),
                "lm_head_call_count": trace.get("lm_head_call_count"),
                "module_paths": trace.get("module_paths"),
                "missing_modules": trace.get("missing_modules"),
                "n_events": trace.get("n_events"),
                "trace_v2_error": trace.get("trace_v2_error"),
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
