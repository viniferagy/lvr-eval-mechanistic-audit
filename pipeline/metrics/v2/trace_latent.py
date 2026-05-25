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
GENERATION_SCORE_MARGIN_SOURCE = "generation_scores_aligned_first_token"
CONTINUOUS_MARGIN_SOURCES = {"generation_scores", GENERATION_SCORE_MARGIN_SOURCE, "latent_logit_lens"}


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


def _shape_list(value: Any) -> list[int] | None:
    if torch is not None and torch.is_tensor(value):
        return [int(dim) for dim in value.shape]
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001
        return None
    if arr.shape == ():
        return []
    return [int(dim) for dim in arr.shape]


def _normalize_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if torch is not None and torch.is_tensor(value):
        value = value.detach().flatten().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif isinstance(value, tuple):
        value = list(value)
    elif not isinstance(value, list):
        try:
            value = list(value)
        except TypeError:
            value = [value]
    if value and isinstance(value[0], list):
        value = value[0]
    out = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:  # noqa: BLE001
            continue
    return out


def generated_token_ids(trace: dict) -> list[int]:
    """Return generated token ids after the prompt, when the adapter exposes them."""
    explicit = trace.get("generated_ids")
    if explicit is not None:
        return _normalize_id_list(explicit)

    sequences = trace.get("sequences")
    if sequences is None:
        return []
    try:
        if torch is not None and torch.is_tensor(sequences):
            seq = sequences[0] if sequences.dim() > 1 else sequences
        else:
            seq = sequences[0] if sequences and isinstance(sequences[0], (list, tuple)) else sequences
        prompt_len = trace.get("prompt_len")
        if prompt_len is None:
            inputs = trace.get("inputs") or {}
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            if torch is not None and torch.is_tensor(input_ids):
                prompt_len = int(input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0])
            elif input_ids is not None:
                arr = np.asarray(input_ids)
                prompt_len = int(arr.shape[1] if arr.ndim > 1 else arr.shape[0])
        prompt_len = int(prompt_len or 0)
        if torch is not None and torch.is_tensor(seq):
            return [int(v) for v in seq[prompt_len:].detach().cpu().tolist()]
        return _normalize_id_list(seq[prompt_len:])
    except Exception:  # noqa: BLE001
        return []


def _decode_token(wrapper, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    tokenizer = getattr(wrapper.processor, "tokenizer", wrapper.processor)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except TypeError:
        try:
            return tokenizer.decode([int(token_id)])
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


def _score_tensor_at(scores: Any, index: int):
    score = scores[index]
    if torch is not None and torch.is_tensor(score):
        tensor = score.detach().float()
        if tensor.dim() == 2:
            tensor = tensor[0]
        elif tensor.dim() != 1:
            return None, _shape_list(score)
        return tensor, _shape_list(score)
    arr = np.asarray(score, dtype=float)
    shape = [int(dim) for dim in arr.shape]
    if arr.ndim == 2:
        arr = arr[0]
    elif arr.ndim != 1:
        return None, shape
    return arr, shape


def _score_value(logits: Any, token_id: int) -> float | None:
    try:
        if torch is not None and torch.is_tensor(logits):
            if int(token_id) < 0 or int(token_id) >= int(logits.shape[-1]):
                return None
            value = float(logits[int(token_id)].detach().cpu().item())
        else:
            if int(token_id) < 0 or int(token_id) >= int(len(logits)):
                return None
            value = float(logits[int(token_id)])
    except Exception:  # noqa: BLE001
        return None
    return value if np.isfinite(value) else None


def _score_diag_fail(diag: dict, reason: str, **updates) -> dict:
    diag.update(updates)
    diag["status"] = "fail"
    diag["reason"] = reason
    diag["margin"] = None
    return diag


def score_margin_with_diagnostics(
    trace: dict,
    wrapper,
    source_answer: Any,
    target_answer: Any,
    *,
    label: str | None = None,
) -> dict:
    """Aligned generation-score answer margin with machine-readable failure reasons.

    This intentionally avoids the old ``scores[-1]`` assumption. Generation
    scores are used only when a single-token source/target candidate appears in
    the generated token stream and the matching score index is present.
    Multi-token candidate sequence scoring remains a separate future path, so
    those cases are diagnosed and passed to the latent-logit fallback.
    """
    source_ids = token_ids(wrapper, source_answer)
    target_ids = token_ids(wrapper, target_answer)
    scores = trace.get("scores")
    generated_ids = generated_token_ids(trace)
    decoded_tokens = trace.get("decoded_generated_tokens")
    if decoded_tokens is None:
        decoded_tokens = [_decode_token(wrapper, token_id) for token_id in generated_ids[:32]]

    diag = {
        "label": label,
        "source": "generation_scores",
        "status": "fail",
        "reason": None,
        "margin": None,
        "scores_len": None,
        "generated_ids_len": len(generated_ids),
        "source_token_ids": source_ids,
        "target_token_ids": target_ids,
        "source_token_len": len(source_ids),
        "target_token_len": len(target_ids),
        "score_shape": None,
        "used_score_index": None,
        "used_generated_token_index": None,
        "used_generated_token_id": None,
        "used_generated_token_text": None,
        "candidate_seen": None,
        "score_generated_length_mismatch": None,
        "adapter_score_dropped": False,
        "decoded_generated_tokens_preview": decoded_tokens[:16] if isinstance(decoded_tokens, list) else None,
    }

    if not source_ids or not target_ids:
        return _score_diag_fail(diag, "candidate_token_missing")
    if len(source_ids) != 1 or len(target_ids) != 1:
        return _score_diag_fail(diag, "multi_token_candidate")
    if scores is None:
        return _score_diag_fail(
            diag,
            "missing_scores",
            adapter_score_dropped=bool(trace.get("_captured_states") or trace.get("captured_state_metadata")),
        )
    try:
        scores_len = len(scores)
    except TypeError:
        return _score_diag_fail(diag, "bad_score_shape", score_shape=_shape_list(scores))
    diag["scores_len"] = int(scores_len)
    diag["score_generated_length_mismatch"] = bool(generated_ids and scores_len != len(generated_ids))
    if scores_len <= 0:
        return _score_diag_fail(diag, "empty_scores")
    if not generated_ids:
        return _score_diag_fail(diag, "missing_generated_ids")

    source_id = int(source_ids[0])
    target_id = int(target_ids[0])
    candidate_index = None
    candidate_seen = None
    for idx, token_id in enumerate(generated_ids):
        if int(token_id) == source_id:
            candidate_index = idx
            candidate_seen = "source"
            break
        if int(token_id) == target_id:
            candidate_index = idx
            candidate_seen = "target"
            break
    if candidate_index is None:
        return _score_diag_fail(diag, "decision_index_mismatch")
    if candidate_index >= scores_len:
        return _score_diag_fail(
            diag,
            "score_generated_length_mismatch",
            used_generated_token_index=int(candidate_index),
            used_generated_token_id=int(generated_ids[candidate_index]),
            used_generated_token_text=_decode_token(wrapper, int(generated_ids[candidate_index])),
            candidate_seen=candidate_seen,
        )

    logits, score_shape = _score_tensor_at(scores, candidate_index)
    diag.update({
        "score_shape": score_shape,
        "used_score_index": int(candidate_index),
        "used_generated_token_index": int(candidate_index),
        "used_generated_token_id": int(generated_ids[candidate_index]),
        "used_generated_token_text": _decode_token(wrapper, int(generated_ids[candidate_index])),
        "candidate_seen": candidate_seen,
    })
    if logits is None:
        return _score_diag_fail(diag, "bad_score_shape")

    source_value = _score_value(logits, source_id)
    target_value = _score_value(logits, target_id)
    if source_value is None or target_value is None:
        return _score_diag_fail(diag, "nonfinite_score")
    margin = source_value - target_value
    if not np.isfinite(margin):
        return _score_diag_fail(diag, "nonfinite_score")
    diag.update({
        "status": "pass",
        "reason": "aligned_candidate_token",
        "source": GENERATION_SCORE_MARGIN_SOURCE,
        "margin": float(margin),
        "source_score": float(source_value),
        "target_score": float(target_value),
    })
    return diag


def score_margin(trace: dict, wrapper, source_answer: Any, target_answer: Any) -> float | None:
    return score_margin_with_diagnostics(trace, wrapper, source_answer, target_answer).get("margin")


def score_diagnostic_summary(records: list[dict]) -> dict:
    total = 0
    clean_ok = 0
    patched_ok = 0
    either_ok = 0
    both_ok = 0
    missing_diag = 0
    reason_counts: dict[str, int] = {}
    clean_reason_counts: dict[str, int] = {}
    patched_reason_counts: dict[str, int] = {}
    multi_token = 0
    length_mismatch = 0
    for record in records:
        if record.get("error") is not None:
            continue
        clean = record.get("clean_score_diagnostic")
        patched = record.get("patched_score_diagnostic")
        if not isinstance(clean, dict) or not isinstance(patched, dict):
            missing_diag += 1
            continue
        total += 1
        clean_pass = clean.get("status") == "pass"
        patched_pass = patched.get("status") == "pass"
        clean_ok += int(clean_pass)
        patched_ok += int(patched_pass)
        either_ok += int(clean_pass or patched_pass)
        both_ok += int(clean_pass and patched_pass)
        for prefix, diag, bucket in (
            ("clean", clean, clean_reason_counts),
            ("patched", patched, patched_reason_counts),
        ):
            if diag.get("status") == "pass":
                continue
            reason = str(diag.get("reason") or "unknown")
            bucket[reason] = bucket.get(reason, 0) + 1
            key = f"{prefix}:{reason}"
            reason_counts[key] = reason_counts.get(key, 0) + 1
        if clean.get("reason") == "multi_token_candidate" or patched.get("reason") == "multi_token_candidate":
            multi_token += 1
        if clean.get("score_generated_length_mismatch") or patched.get("score_generated_length_mismatch"):
            length_mismatch += 1
    return {
        "diagnostic_records": total,
        "missing_diagnostic_records": missing_diag,
        "clean_generation_scores_ok": clean_ok,
        "patched_generation_scores_ok": patched_ok,
        "either_generation_scores_ok": either_ok,
        "both_generation_scores_ok": both_ok,
        "clean_generation_scores_ok_rate": float(clean_ok) / float(total) if total else None,
        "patched_generation_scores_ok_rate": float(patched_ok) / float(total) if total else None,
        "both_generation_scores_ok_rate": float(both_ok) / float(total) if total else None,
        "generation_score_failure_reason_counts": reason_counts,
        "clean_generation_score_failure_reason_counts": clean_reason_counts,
        "patched_generation_score_failure_reason_counts": patched_reason_counts,
        "multi_token_candidate_ratio": float(multi_token) / float(total) if total else None,
        "score_generated_length_mismatch_ratio": float(length_mismatch) / float(total) if total else None,
    }


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
        "clean_margin_source": patched_summary.get("clean_margin_source"),
        "patched_margin_source": patched_summary.get("patched_margin_source"),
        "clean_generation_score_margin": patched_summary.get("clean_generation_score_margin"),
        "patched_generation_score_margin": patched_summary.get("patched_generation_score_margin"),
        "generation_score_margin_shift": patched_summary.get("generation_score_margin_shift"),
        "clean_score_diagnostic": patched_summary.get("clean_score_diagnostic"),
        "patched_score_diagnostic": patched_summary.get("patched_score_diagnostic"),
        "score_failure_reasons": patched_summary.get("score_failure_reasons"),
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
    clean_score_diag = score_margin_with_diagnostics(
        clean_trace,
        wrapper,
        source_answer_for_margin,
        target_answer_for_margin,
        label="clean",
    )
    patched_score_diag = score_margin_with_diagnostics(
        patched_trace,
        wrapper,
        source_answer_for_margin,
        target_answer_for_margin,
        label="patched",
    )
    clean_margin = clean_score_diag.get("margin")
    patched_margin = patched_score_diag.get("margin")
    latent_clean_margin = None
    latent_patched_margin = None
    generation_score_margin_shift = (
        patched_margin - clean_margin
        if clean_score_diag.get("status") == "pass"
        and patched_score_diag.get("status") == "pass"
        and patched_margin is not None
        and clean_margin is not None
        else None
    )
    margin_source = (
        GENERATION_SCORE_MARGIN_SOURCE
        if generation_score_margin_shift is not None
        else None
    )
    clean_margin_source = margin_source
    patched_margin_source = margin_source
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
            clean_margin_source = "latent_logit_lens"
            patched_margin_source = "latent_logit_lens"
    if margin_source is None:
        clean_margin = parsed_margin(parsed_clean, parsed_source, parse_candidate(sample.answer))
        patched_margin = parsed_margin(parsed_patched, parsed_source, parse_candidate(sample.answer))
        margin_source = "parsed_answer_fallback"
        clean_margin_source = "parsed_answer_fallback"
        patched_margin_source = "parsed_answer_fallback"
    score_failure_reasons = []
    for prefix, diag in (("clean", clean_score_diag), ("patched", patched_score_diag)):
        if diag.get("status") != "pass":
            score_failure_reasons.append(f"{prefix}:{diag.get('reason') or 'unknown'}")
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
        "clean_margin_source": clean_margin_source,
        "patched_margin_source": patched_margin_source,
        "clean_generation_score_margin": clean_score_diag.get("margin"),
        "patched_generation_score_margin": patched_score_diag.get("margin"),
        "generation_score_margin_shift": generation_score_margin_shift,
        "clean_score_diagnostic": clean_score_diag,
        "patched_score_diagnostic": patched_score_diag,
        "score_failure_reasons": score_failure_reasons,
        "trace_quality": patched_trace.get("trace_quality"),
        "missing_modules": patched_trace.get("missing_modules"),
        "trace_v2_error": patched_trace.get("trace_v2_error"),
        "n_lvr_mode_steps": int(patched_trace.get("n_lvr_mode_steps") or 0),
        "n_hidden_feedback_steps": int(patched_trace.get("n_hidden_feedback_steps") or 0),
        "n_patch_applied": int(patched_trace.get("n_patch_applied") or 0),
        "patched_captured_state_shapes": state_shape_metadata(patched_trace),
    }
