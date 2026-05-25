"""Shared utilities for generation-trace latent metrics.

These helpers keep the W14/W15 trace-latent path explicit and opt-in. The
existing v2 metrics still default to their W7/W9 query-span implementations;
configs that set ``trace_latent.enabled=true`` switch supported LVR metrics to
real generation-time hidden-feedback states captured by the traced adapter.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - smoke envs may not import torch.
    torch = None


CANDIDATES = ("original", "modified")
TRACE_QUALITY = "instrumented_sparse_v0"


class TraceLatentPolicyViolation(RuntimeError):
    pass


def cfg(cfg_root: dict) -> dict:
    return cfg_root.get("trace_latent", {}) or {}


def enabled(cfg_root: dict, metric_id: str) -> bool:
    local = cfg(cfg_root)
    if not bool(local.get("enabled", False)):
        return False
    apply_to = local.get("apply_to", "all")
    if apply_to == "all":
        return True
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    return metric_id in {str(item) for item in (apply_to or [])}


def required(cfg_root: dict) -> bool:
    local = cfg(cfg_root)
    trace_v2 = cfg_root.get("trace_v2", {}) or {}
    return bool(local.get("required", trace_v2.get("required", False)))


def forbid_fallback(cfg_root: dict) -> bool:
    local = cfg(cfg_root)
    trace_v2 = cfg_root.get("trace_v2", {}) or {}
    return bool(local.get("forbid_fallback", trace_v2.get("forbid_fallback", required(cfg_root))))


def assert_trace_ok(trace: dict, cfg_root: dict, label: str) -> None:
    if not required(cfg_root):
        return
    quality = trace.get("trace_quality")
    if quality != TRACE_QUALITY:
        raise TraceLatentPolicyViolation(
            f"{label} trace_quality must be {TRACE_QUALITY} under trace_latent.required; got {quality!r}"
        )
    if forbid_fallback(cfg_root) and trace.get("trace_v2_error"):
        raise TraceLatentPolicyViolation(f"{label} trace fallback/error forbidden: {trace.get('trace_v2_error')!r}")
    if forbid_fallback(cfg_root) and trace.get("missing_modules"):
        raise TraceLatentPolicyViolation(f"{label} trace missing modules forbidden: {trace.get('missing_modules')!r}")


def _trace_kwargs(cfg_root: dict, *, patch_states=None, patch_steps=None) -> dict:
    local = cfg(cfg_root)
    audit = cfg_root.get("audit", {}) or {}
    lvr_steps = int(local.get("lvr_steps", audit.get("lvr_steps", 8)))
    return {
        "decoding_strategy": str(local.get("decoding_strategy", audit.get("lvr_decoding_strategy", "steps"))),
        "lvr_steps": lvr_steps,
        "max_new_tokens": int(local.get("max_new_tokens", 96)),
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": True,
        "trace_capture": {
            "capture_tensors": True,
            "max_captured_tensors": int(local.get("max_captured_tensors", max(4, lvr_steps + 2))),
            "patch_states": patch_states,
            "patch_steps": patch_steps,
            "patch_shape_policy": str(local.get("patch_shape_policy", "strict")),
        },
    }


def generate_trace(wrapper, sample, cfg_root: dict, *, image=None, question: str | None = None,
                   patch_states=None, patch_steps=None, label: str = "trace") -> dict:
    if not hasattr(wrapper.adapter, "generate_with_trace"):
        raise TraceLatentPolicyViolation(f"{label} adapter does not expose generate_with_trace")
    trace = wrapper.adapter.generate_with_trace(
        wrapper,
        sample.image if image is None else image,
        sample.question if question is None else question,
        **_trace_kwargs(cfg_root, patch_states=patch_states, patch_steps=patch_steps),
    )
    assert_trace_ok(trace, cfg_root, label)
    return trace


def captured_state_items(trace: dict, *, kind: str = "output_last_position_hidden_state") -> list[dict]:
    return [
        item for item in trace.get("_captured_states", [])
        if item.get("kind") == kind and item.get("tensor") is not None
    ]


def state_shape_metadata(trace: dict) -> list:
    return [item.get("shape") for item in trace.get("captured_state_metadata", [])]


def selected_state_items(trace: dict, steps: list[str] | tuple[str, ...] | None = None) -> list[dict]:
    states = captured_state_items(trace)
    if not states:
        return []
    steps = [str(step) for step in (steps or ["all"])]
    if "all" in steps:
        return states
    if "last" in steps:
        return [states[-1]]
    requested = {int(step) for step in steps}
    return [item for item in states if int(item.get("step_index", -1)) in requested]


def _item_matrix(item: dict):
    tensor = item.get("tensor")
    if torch is not None and torch.is_tensor(tensor):
        arr = tensor.detach().float().cpu().numpy()
    else:
        arr = np.asarray(tensor, dtype=float)
    if arr.size == 0:
        return None
    arr = arr.reshape(-1, arr.shape[-1])
    return arr[-1]


def state_matrix(trace: dict, steps: list[str] | tuple[str, ...] | None = None) -> np.ndarray | None:
    rows = []
    for item in selected_state_items(trace, steps):
        row = _item_matrix(item)
        if row is not None and np.all(np.isfinite(row)):
            rows.append(row.astype(float))
    if not rows:
        return None
    return np.stack(rows, axis=0)


def state_distance(trace_a: dict, trace_b: dict, *,
                   steps: list[str] | tuple[str, ...] | None = None) -> dict:
    a = state_matrix(trace_a, steps)
    b = state_matrix(trace_b, steps)
    if a is None or b is None:
        return {
            "mean_l2": None,
            "mean_cosine_distance": None,
            "n_aligned_states": 0,
        }
    n = min(a.shape[0], b.shape[0])
    h = min(a.shape[1], b.shape[1])
    a = a[:n, :h]
    b = b[:n, :h]
    diff = a - b
    l2 = np.linalg.norm(diff, axis=1) / max(np.sqrt(float(h)), 1.0)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide(np.sum(a * b, axis=1), denom, out=np.zeros(n, dtype=float), where=denom > 0)
    return {
        "mean_l2": float(np.mean(l2)),
        "mean_cosine_distance": float(np.mean(1.0 - cosine)),
        "source_mean_norm": float(np.mean(np.linalg.norm(a, axis=1))),
        "target_mean_norm": float(np.mean(np.linalg.norm(b, axis=1))),
        "n_aligned_states": int(n),
        "hidden_size": int(h),
    }


def latent_delta(trace_a: dict, trace_b: dict, *,
                 steps: list[str] | tuple[str, ...] | None = None,
                 distance: str = "l2") -> float | None:
    dist = state_distance(trace_a, trace_b, steps=steps)
    key = "mean_cosine_distance" if distance == "cosine" else "mean_l2"
    value = dist.get(key)
    return float(value) if value is not None else None


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


def token_ids(wrapper, answer: Any) -> list[int]:
    text = str(answer or "").strip()
    if not text:
        return []
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
    return [int(token_id) for token_id in (ids or [])]


def first_token_id(wrapper, answer: Any) -> int | None:
    ids = token_ids(wrapper, answer)
    return ids[0] if ids else None


def score_margin(trace: dict, wrapper, source_answer: Any, target_answer: Any) -> float | None:
    source_id = first_token_id(wrapper, source_answer)
    target_id = first_token_id(wrapper, target_answer)
    scores = trace.get("scores")
    if source_id is None or target_id is None or not scores:
        return None
    try:
        logits = scores[-1][0].detach().float()
        value = float((logits[int(source_id)] - logits[int(target_id)]).cpu().item())
        return value if np.isfinite(value) else None
    except Exception:  # noqa: BLE001
        return None


def latent_logit_margin(wrapper, trace: dict, source_answer: Any, target_answer: Any) -> float | None:
    """Continuous answer margin from the last captured latent hidden state.

    Generation APIs do not always expose reliable constrained-answer scores.
    This fallback uses the same logit-lens path as ``latent_answer_logit_curve``
    but compares source and target answer token logits at the final captured
    generation-time latent state.
    """
    if torch is None:
        return None
    source_id = first_token_id(wrapper, source_answer)
    target_id = first_token_id(wrapper, target_answer)
    if source_id is None or target_id is None:
        return None
    states = captured_state_items(trace)
    if not states:
        return None
    tensor = states[-1].get("tensor")
    if not torch.is_tensor(tensor):
        return None
    try:
        h = tensor.detach().to(wrapper.model.device)
        head_weight = getattr(wrapper.lm_head, "weight", None)
        if torch.is_tensor(head_weight):
            h = h.to(dtype=head_weight.dtype)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if h.dim() > 2:
            h = h.reshape(-1, h.shape[-1])
        h = h[-1:, :]
        normed = wrapper.final_norm(h) if wrapper.final_norm is not None else h
        logits = wrapper.lm_head(normed)[0].detach().float()
        value = float((logits[int(source_id)] - logits[int(target_id)]).cpu().item())
        return value if np.isfinite(value) else None
    except Exception:  # noqa: BLE001
        return None


def parsed_margin(parsed_answer: str | None, source_answer: str | None, target_answer: str | None) -> float | None:
    if parsed_answer is None or source_answer is None or target_answer is None:
        return None
    if parsed_answer == source_answer:
        return 1.0
    if parsed_answer == target_answer:
        return -1.0
    return 0.0


def latent_answer_logit_curve(wrapper, trace: dict, answer: Any) -> list[float]:
    if torch is None:
        return []
    answer_id = first_token_id(wrapper, answer)
    if answer_id is None:
        return []
    values: list[float] = []
    for item in captured_state_items(trace):
        tensor = item.get("tensor")
        if not torch.is_tensor(tensor):
            continue
        h = tensor.detach().to(wrapper.model.device)
        head_weight = getattr(wrapper.lm_head, "weight", None)
        if torch.is_tensor(head_weight):
            h = h.to(dtype=head_weight.dtype)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if h.dim() > 2:
            h = h.reshape(-1, h.shape[-1])
        h = h[-1:, :]
        normed = wrapper.final_norm(h) if wrapper.final_norm is not None else h
        logits = wrapper.lm_head(normed)
        values.append(float(logits[0, int(answer_id)].detach().float().cpu().item()))
    return values


def latent_norm_curve(trace: dict) -> list[float]:
    values = []
    for item in captured_state_items(trace):
        row = _item_matrix(item)
        if row is not None:
            values.append(float(np.linalg.norm(row)))
    return values


def safe_slope(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if len(vals) < 2:
        return None
    x = np.arange(len(vals), dtype=float)
    return float(np.polyfit(x, np.asarray(vals, dtype=float), deg=1)[0])


def source_sample(sample):
    return replace(sample, image=sample.counterfactual_image, answer=sample.counterfactual_answer)


def patch_pair(wrapper, sample, cfg_root: dict, *,
               patch_steps: list[str] | tuple[str, ...] | None = None) -> dict:
    if sample.counterfactual_image is None or sample.counterfactual_answer is None:
        raise ValueError("sample missing counterfactual image/answer")
    local = cfg(cfg_root)
    patch_steps = [str(v) for v in (patch_steps or local.get("patch_steps", ["last"]))]
    source = source_sample(sample)
    target = sample

    source_trace = generate_trace(wrapper, source, cfg_root, label="source")
    clean_trace = generate_trace(wrapper, target, cfg_root, label="clean")
    selected = selected_state_items(source_trace, patch_steps)
    if not selected:
        raise RuntimeError("source trace captured no selected patchable latent states")

    if "each" in patch_steps:
        step_results = []
        for item in selected:
            step_index = int(item.get("step_index", len(step_results)))
            patched_trace = generate_trace(
                wrapper,
                target,
                cfg_root,
                patch_states=[item["tensor"]],
                patch_steps=[step_index],
                label=f"patched_step_{step_index}",
            )
            step_results.append(_patch_record(
                patched_trace=patched_trace,
                clean_trace=clean_trace,
                wrapper=wrapper,
                sample=sample,
                source=source,
                step_index=step_index,
                source_state_shape=item.get("shape"),
            ))
        patched_summary = sorted(step_results, key=lambda row: int(row.get("step_index") or 0))[-1]
    else:
        step_ids = [int(item["step_index"]) for item in selected if item.get("step_index") is not None]
        patched_trace = generate_trace(
            wrapper,
            target,
            cfg_root,
            patch_states=[item["tensor"] for item in selected],
            patch_steps=step_ids or None,
            label="patched",
        )
        patched_summary = _patch_record(
            patched_trace=patched_trace,
            clean_trace=clean_trace,
            wrapper=wrapper,
            sample=sample,
            source=source,
            step_index=step_ids[-1] if step_ids else None,
            source_state_shape=selected[-1].get("shape"),
        )
        step_results = [patched_summary]

    source_answer = parse_candidate(source.answer)
    target_answer = parse_candidate(target.answer)
    if source_answer is None:
        source_answer = str(source.answer)
    if target_answer is None:
        target_answer = str(target.answer)
    return {
        "id": sample.id,
        "paired_id": sample.paired_id,
        "source_answer": source_answer,
        "target_answer": target_answer,
        "source_answer_token_ids": token_ids(wrapper, source.answer),
        "target_answer_token_ids": token_ids(wrapper, target.answer),
        "clean_generated_text": clean_trace.get("generated_text"),
        "patched_generated_text": patched_summary.get("patched_generated_text"),
        "clean_answer": patched_summary.get("clean_answer"),
        "patched_answer": patched_summary.get("patched_answer"),
        "answer_transferred": patched_summary.get("answer_transferred"),
        "clean_margin": patched_summary.get("clean_margin"),
        "patched_margin": patched_summary.get("patched_margin"),
        "logprob_margin_shift": patched_summary.get("latent_margin_shift"),
        "latent_margin_shift": patched_summary.get("latent_margin_shift"),
        "effective_margin_shift": patched_summary.get("effective_margin_shift"),
        "latent_logit_margin_shift": patched_summary.get("latent_logit_margin_shift"),
        "margin_source": patched_summary.get("margin_source"),
        "trace_quality": patched_summary.get("trace_quality"),
        "source_trace_quality": source_trace.get("trace_quality"),
        "missing_modules": patched_summary.get("missing_modules"),
        "trace_v2_error": patched_summary.get("trace_v2_error"),
        "n_lvr_mode_steps": int(patched_summary.get("n_lvr_mode_steps") or 0),
        "n_hidden_feedback_steps": int(patched_summary.get("n_hidden_feedback_steps") or 0),
        "n_captured_latent_states": int(source_trace.get("n_captured_latent_states") or 0),
        "n_patch_applied": sum(1 for item in step_results if int(item.get("n_patch_applied") or 0) >= 1),
        "n_step_patches_attempted": len(step_results),
        "captured_state_shapes": state_shape_metadata(source_trace),
        "patched_captured_state_shapes": patched_summary.get("patched_captured_state_shapes") or [],
        "patch_steps": patch_steps,
        "step_results": step_results,
        "trace_latent_mode": True,
        "reduction": {
            "logprob_margin_shift": patched_summary.get("latent_margin_shift"),
            "effective_margin_shift": patched_summary.get("effective_margin_shift"),
            "latent_logit_margin_shift": patched_summary.get("latent_logit_margin_shift"),
            "answer_transfer_rate": (
                float(bool(patched_summary.get("answer_transferred")))
                if patched_summary.get("answer_transferred") is not None else None
            ),
            "answer_transfer": (
                float(bool(patched_summary.get("answer_transferred")))
                if patched_summary.get("answer_transferred") is not None else None
            ),
        },
    }


def _patch_record(*, patched_trace: dict, clean_trace: dict, wrapper, sample, source,
                  step_index: int | None, source_state_shape=None) -> dict:
    parsed_source = parse_candidate(source.answer)
    parsed_patched = parse_candidate(patched_trace.get("generated_text"))
    parsed_clean = parse_candidate(clean_trace.get("generated_text"))
    source_answer_for_margin = parsed_source or source.answer
    target_answer_for_margin = parse_candidate(sample.answer) or sample.answer
    clean_margin = score_margin(
        clean_trace,
        wrapper,
        source_answer_for_margin,
        target_answer_for_margin,
    )
    patched_margin = score_margin(
        patched_trace,
        wrapper,
        source_answer_for_margin,
        target_answer_for_margin,
    )
    latent_clean_margin = None
    latent_patched_margin = None
    margin_source = "generation_scores" if clean_margin is not None and patched_margin is not None else None
    if margin_source is None:
        latent_clean_margin = latent_logit_margin(
            wrapper,
            clean_trace,
            source_answer_for_margin,
            target_answer_for_margin,
        )
        latent_patched_margin = latent_logit_margin(
            wrapper,
            patched_trace,
            source_answer_for_margin,
            target_answer_for_margin,
        )
        if latent_clean_margin is not None and latent_patched_margin is not None:
            clean_margin = latent_clean_margin
            patched_margin = latent_patched_margin
            margin_source = "latent_logit_lens"
    if margin_source is None:
        clean_margin = parsed_margin(parsed_clean, parsed_source, parse_candidate(sample.answer))
        patched_margin = parsed_margin(parsed_patched, parsed_source, parse_candidate(sample.answer))
        margin_source = "parsed_answer_fallback"
    margin_shift = (
        patched_margin - clean_margin
        if patched_margin is not None and clean_margin is not None
        else None
    )
    return {
        "step_index": step_index,
        "source_state_shape": source_state_shape,
        "patched_generated_text": patched_trace.get("generated_text"),
        "clean_answer": parsed_clean,
        "patched_answer": parsed_patched,
        "answer_transferred": (
            parsed_patched is not None and parsed_source is not None and parsed_patched == parsed_source
        ),
        "clean_margin": clean_margin,
        "patched_margin": patched_margin,
        "latent_margin_shift": margin_shift,
        "effective_margin_shift": margin_shift,
        "latent_logit_clean_margin": latent_clean_margin,
        "latent_logit_patched_margin": latent_patched_margin,
        "latent_logit_margin_shift": (
            latent_patched_margin - latent_clean_margin
            if latent_patched_margin is not None and latent_clean_margin is not None
            else None
        ),
        "margin_source": margin_source,
        "trace_quality": patched_trace.get("trace_quality"),
        "missing_modules": patched_trace.get("missing_modules"),
        "trace_v2_error": patched_trace.get("trace_v2_error"),
        "n_lvr_mode_steps": int(patched_trace.get("n_lvr_mode_steps") or 0),
        "n_hidden_feedback_steps": int(patched_trace.get("n_hidden_feedback_steps") or 0),
        "n_patch_applied": int(patched_trace.get("n_patch_applied") or 0),
        "patched_captured_state_shapes": state_shape_metadata(patched_trace),
    }
