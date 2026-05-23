"""W3 LVR generation-time latent-state answer transfer."""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

import numpy as np

from ..base import MetricSpec


METRIC_ID = "lvr_latent_patch_answer_transfer"
LEGACY_NAME = "lvr_latent_patch"
CANDIDATES = ("original", "modified")


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "scalars": [
            "latent_answer_transfer_rate",
            "latent_margin_shift",
            "n_paired",
            "n_success",
            "n_patch_applied",
            "n_with_lvr_mode",
            "n_with_captured_state",
        ],
        "patch_policy": {
            "patch_steps": ["last"],
            "patch_tensor": "output_last_position_hidden_state",
            "intervention_site": "forward_pre.last_position_hidden_state",
        },
        "status": "runnable_w3_gate",
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("lvr_latent_patch", {}) or {}


def _source_sample(sample):
    return replace(sample, image=sample.counterfactual_image, answer=sample.counterfactual_answer)


def _norm_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"[^a-z]+", " ", text)
    return " ".join(text.split())


def parse_candidate(text: Any) -> str | None:
    words = _norm_text(text).split()
    for word in words:
        if word in CANDIDATES:
            return word
    return None


def _candidate_token_id(wrapper, answer: Any) -> int | None:
    text = str(answer or "").strip()
    if not text:
        return None
    tokenizer = getattr(wrapper.processor, "tokenizer", wrapper.processor)
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if ids and ids.__class__.__name__ == "Encoding":
        ids = getattr(ids, "ids", [])
    return int(ids[0]) if ids else None


def _last_score_margin(trace: dict, wrapper, source_answer: Any, target_answer: Any) -> float | None:
    source_id = _candidate_token_id(wrapper, source_answer)
    target_id = _candidate_token_id(wrapper, target_answer)
    scores = trace.get("scores")
    if source_id is None or target_id is None or not scores:
        return None
    try:
        logits = scores[-1][0].detach().float()
        return float((logits[int(source_id)] - logits[int(target_id)]).cpu().item())
    except Exception:  # noqa: BLE001
        return None


def _select_patch_states(trace: dict, patch_steps: list[str]) -> list:
    states = [
        item for item in trace.get("_captured_states", [])
        if item.get("kind") == "output_last_position_hidden_state" and item.get("tensor") is not None
    ]
    if not states:
        return []
    if "all" in patch_steps:
        selected = states
    elif "last" in patch_steps:
        selected = [states[-1]]
    else:
        step_ids = {int(step) for step in patch_steps}
        selected = [item for item in states if int(item.get("step_index", -1)) in step_ids]
    return [item["tensor"] for item in selected]


def _trace_kwargs(local: dict, audit_cfg: dict, *, patch_states=None, patch_steps=None) -> dict:
    lvr_steps = int(local.get("lvr_steps", audit_cfg.get("lvr_steps", 2)))
    return {
        "decoding_strategy": str(local.get("decoding_strategy", audit_cfg.get("lvr_decoding_strategy", "steps"))),
        "lvr_steps": lvr_steps,
        "max_new_tokens": int(local.get("max_new_tokens", 128)),
        "output_attentions": False,
        "output_hidden_states": False,
        "trace_capture": {
            "capture_tensors": True,
            "max_captured_tensors": int(local.get("max_captured_tensors", max(4, lvr_steps + 2))),
            "patch_states": patch_states,
            "patch_steps": patch_steps,
        },
    }


def run_one_pair(wrapper, sample, cfg: dict) -> dict:
    local = _cfg(cfg)
    audit_cfg = cfg.get("audit", {}) or {}
    patch_steps = [str(v) for v in local.get("patch_steps", ["last"])]
    if sample.counterfactual_image is None or sample.counterfactual_answer is None:
        raise ValueError("sample missing counterfactual image/answer")

    source = _source_sample(sample)
    target = sample
    source_trace = wrapper.adapter.generate_with_trace(
        wrapper,
        source.image,
        source.question,
        **_trace_kwargs(local, audit_cfg),
    )
    patch_states = _select_patch_states(source_trace, patch_steps)
    if not patch_states:
        raise RuntimeError("source trace captured no patchable latent states")

    clean_trace = wrapper.adapter.generate_with_trace(
        wrapper,
        target.image,
        target.question,
        **_trace_kwargs(local, audit_cfg),
    )
    patched_trace = wrapper.adapter.generate_with_trace(
        wrapper,
        target.image,
        target.question,
        **_trace_kwargs(local, audit_cfg, patch_states=patch_states, patch_steps=None),
    )

    source_answer = parse_candidate(source.answer)
    target_answer = parse_candidate(target.answer)
    clean_answer = parse_candidate(clean_trace.get("generated_text"))
    patched_answer = parse_candidate(patched_trace.get("generated_text"))
    clean_margin = _last_score_margin(clean_trace, wrapper, source.answer, target.answer)
    patched_margin = _last_score_margin(patched_trace, wrapper, source.answer, target.answer)
    margin_shift = (
        patched_margin - clean_margin
        if patched_margin is not None and clean_margin is not None
        else None
    )
    n_patch_applied = int(patched_trace.get("n_patch_applied") or 0)
    n_captured = int(source_trace.get("n_captured_latent_states") or 0)

    return {
        "id": sample.id,
        "paired_id": sample.paired_id,
        "source_answer": source_answer or str(source.answer),
        "target_answer": target_answer or str(target.answer),
        "clean_generated_text": clean_trace.get("generated_text"),
        "patched_generated_text": patched_trace.get("generated_text"),
        "clean_answer": clean_answer,
        "patched_answer": patched_answer,
        "answer_transferred": (
            patched_answer is not None and source_answer is not None and patched_answer == source_answer
        ),
        "clean_margin": clean_margin,
        "patched_margin": patched_margin,
        "latent_margin_shift": margin_shift,
        "trace_quality": patched_trace.get("trace_quality"),
        "source_trace_quality": source_trace.get("trace_quality"),
        "missing_modules": patched_trace.get("missing_modules"),
        "trace_v2_error": patched_trace.get("trace_v2_error"),
        "n_lvr_mode_steps": int(patched_trace.get("n_lvr_mode_steps") or 0),
        "n_hidden_feedback_steps": int(patched_trace.get("n_hidden_feedback_steps") or 0),
        "n_captured_latent_states": n_captured,
        "n_patch_applied": n_patch_applied,
        "captured_state_shapes": [item.get("shape") for item in source_trace.get("captured_state_metadata", [])],
        "patched_captured_state_shapes": [
            item.get("shape") for item in patched_trace.get("captured_state_metadata", [])
        ],
        "patch_steps": patch_steps,
        "reduction": {
            "latent_answer_transfer_rate": float(
                patched_answer is not None and source_answer is not None and patched_answer == source_answer
            ),
            "latent_margin_shift": margin_shift,
        },
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def reduce_records(records: list[dict], n_paired: int) -> dict | None:
    valid = [r for r in records if r.get("error") is None]
    if not valid and n_paired <= 0:
        return None
    transfers = [
        float(bool(r.get("answer_transferred")))
        for r in valid
        if r.get("answer_transferred") is not None
    ]
    shifts = [
        float(r["latent_margin_shift"])
        for r in valid
        if r.get("latent_margin_shift") is not None
    ]
    return {
        "latent_answer_transfer_rate": _mean(transfers),
        "latent_margin_shift": _mean(shifts),
        "n_paired": int(n_paired),
        "n_success": len(valid),
        "n_error": n_paired - len(valid),
        "n_patch_applied": sum(1 for r in valid if int(r.get("n_patch_applied") or 0) >= 1),
        "n_with_lvr_mode": sum(1 for r in valid if int(r.get("n_lvr_mode_steps") or 0) >= 1),
        "n_with_captured_state": sum(1 for r in valid if int(r.get("n_captured_latent_states") or 0) >= 1),
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    local = _cfg(cfg)
    paired = [
        sample for sample in samples
        if sample.counterfactual_image is not None and sample.counterfactual_answer is not None
    ]
    max_pairs = local.get("max_pairs")
    if max_pairs is not None:
        paired = paired[:int(max_pairs)]

    records = []
    for sample in paired:
        try:
            records.append(run_one_pair(wrapper, sample, cfg))
        except Exception as exc:  # noqa: BLE001
            records.append({
                "id": sample.id,
                "paired_id": sample.paired_id,
                "source_answer": str(sample.counterfactual_answer),
                "target_answer": str(sample.answer),
                "error": repr(exc),
            })

    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "max_pairs": max_pairs,
            "patch_steps": [str(v) for v in local.get("patch_steps", ["last"])],
            "patch_tensor": "output_last_position_hidden_state",
            "intervention_site": "forward_pre.last_position_hidden_state",
            "answer_candidates": list(CANDIDATES),
        },
        "samples": records,
        "reduction": reduce_records(records, len(paired)),
        "n_paired": len(paired),
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="LVR Latent Patch Answer Transfer",
    kind="internal_curve",
    run_fn=run,
)
