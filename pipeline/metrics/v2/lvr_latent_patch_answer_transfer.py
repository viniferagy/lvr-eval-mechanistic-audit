"""W3/W4 LVR generation-time latent-state answer transfer."""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

import numpy as np

from ..base import MetricSpec


METRIC_ID = "lvr_latent_patch_answer_transfer"
LEGACY_NAME = "lvr_latent_patch"
CANDIDATES = ("original", "modified")


class TracePolicyViolation(RuntimeError):
    pass


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "scalars": [
            "latent_answer_transfer_rate",
            "latent_margin_shift",
            "best_step_transfer_rate",
            "best_step_index",
            "step_transfer_auc",
            "last_step_transfer_rate",
            "n_steps_evaluated",
            "n_paired",
            "n_success",
            "n_patch_applied",
            "n_with_lvr_mode",
            "n_with_captured_state",
        ],
        "patch_policy": {
            "patch_steps": ["last", "each"],
            "patch_tensor": "output_last_position_hidden_state",
            "intervention_site": "forward_pre.last_position_hidden_state",
        },
        "status": "runnable_w4_step_localization",
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("lvr_latent_patch", {}) or {}


def _trace_policy(cfg: dict) -> tuple[bool, bool]:
    trace_cfg = cfg.get("trace_v2", {}) or {}
    required = bool(trace_cfg.get("required", False))
    forbid_fallback = bool(trace_cfg.get("forbid_fallback", required))
    return required, forbid_fallback


def _assert_trace_ok(trace: dict, cfg: dict, label: str) -> None:
    required, forbid_fallback = _trace_policy(cfg)
    if not required:
        return
    quality = trace.get("trace_quality")
    if quality != "instrumented_sparse_v0":
        raise TracePolicyViolation(
            f"{label} trace_quality must be instrumented_sparse_v0 under trace_v2.required; got {quality!r}"
        )
    if forbid_fallback and trace.get("trace_v2_error"):
        raise TracePolicyViolation(f"{label} trace fallback/error forbidden: {trace.get('trace_v2_error')!r}")
    if forbid_fallback and trace.get("missing_modules"):
        raise TracePolicyViolation(f"{label} trace missing modules forbidden: {trace.get('missing_modules')!r}")


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


def _captured_output_states(trace: dict) -> list[dict]:
    return [
        item for item in trace.get("_captured_states", [])
        if item.get("kind") == "output_last_position_hidden_state" and item.get("tensor") is not None
    ]


def _select_patch_state_items(trace: dict, patch_steps: list[str]) -> list[dict]:
    states = _captured_output_states(trace)
    if not states:
        return []
    if "all" in patch_steps:
        selected = states
    elif "last" in patch_steps:
        selected = [states[-1]]
    elif "each" in patch_steps:
        selected = states
    else:
        step_ids = {int(step) for step in patch_steps}
        selected = [item for item in states if int(item.get("step_index", -1)) in step_ids]
    return selected


def _select_patch_states(trace: dict, patch_steps: list[str]) -> list:
    return [item["tensor"] for item in _select_patch_state_items(trace, patch_steps)]


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
            "patch_shape_policy": str(local.get("patch_shape_policy", "strict")),
        },
    }


def _patch_result(
    *,
    patched_trace: dict,
    clean_trace: dict,
    wrapper,
    source_answer_text: Any,
    target_answer_text: Any,
    source_answer: str | None,
    clean_answer: str | None,
    step_index: int | None,
    source_state_shape=None,
) -> dict:
    patched_answer = parse_candidate(patched_trace.get("generated_text"))
    clean_margin = _last_score_margin(clean_trace, wrapper, source_answer_text, target_answer_text)
    patched_margin = _last_score_margin(patched_trace, wrapper, source_answer_text, target_answer_text)
    margin_shift = (
        patched_margin - clean_margin
        if patched_margin is not None and clean_margin is not None
        else None
    )
    return {
        "step_index": step_index,
        "source_state_shape": source_state_shape,
        "patched_generated_text": patched_trace.get("generated_text"),
        "patched_answer": patched_answer,
        "clean_answer": clean_answer,
        "answer_transferred": (
            patched_answer is not None and source_answer is not None and patched_answer == source_answer
        ),
        "clean_margin": clean_margin,
        "patched_margin": patched_margin,
        "latent_margin_shift": margin_shift,
        "trace_quality": patched_trace.get("trace_quality"),
        "missing_modules": patched_trace.get("missing_modules"),
        "trace_v2_error": patched_trace.get("trace_v2_error"),
        "n_lvr_mode_steps": int(patched_trace.get("n_lvr_mode_steps") or 0),
        "n_hidden_feedback_steps": int(patched_trace.get("n_hidden_feedback_steps") or 0),
        "n_patch_applied": int(patched_trace.get("n_patch_applied") or 0),
        "patched_captured_state_shapes": [
            item.get("shape") for item in patched_trace.get("captured_state_metadata", [])
        ],
    }


def _step_transfer_auc(step_results: list[dict]) -> float | None:
    points = [
        (int(r["step_index"]), float(bool(r.get("answer_transferred"))))
        for r in step_results
        if r.get("step_index") is not None and r.get("answer_transferred") is not None
    ]
    if not points:
        return None
    points = sorted(points)
    y = np.asarray([value for _idx, value in points], dtype=float)
    if y.size == 1:
        return float(y[0])
    x = np.asarray([idx for idx, _value in points], dtype=float)
    denom = float(x[-1] - x[0])
    if denom <= 0:
        return float(np.mean(y))
    return float(np.trapz(y, x=x) / denom)


def _record_step_summary(step_results: list[dict]) -> dict:
    if not step_results:
        return {}
    ordered = sorted(
        [r for r in step_results if r.get("step_index") is not None],
        key=lambda item: int(item["step_index"]),
    )
    if not ordered:
        ordered = list(step_results)
    transfer_values = [
        float(bool(r.get("answer_transferred")))
        for r in ordered
        if r.get("answer_transferred") is not None
    ]
    shift_values = [
        float(r["latent_margin_shift"])
        for r in ordered
        if r.get("latent_margin_shift") is not None
    ]
    best = max(
        ordered,
        key=lambda item: (float(bool(item.get("answer_transferred"))), -int(item.get("step_index") or 0)),
    )
    last = ordered[-1]
    return {
        "latent_answer_transfer_rate": float(bool(last.get("answer_transferred"))),
        "latent_margin_shift": last.get("latent_margin_shift"),
        "best_step_transfer_rate": float(bool(best.get("answer_transferred"))),
        "best_step_index": best.get("step_index"),
        "step_transfer_auc": _step_transfer_auc(ordered),
        "last_step_transfer_rate": float(bool(last.get("answer_transferred"))),
        "mean_step_transfer_rate": _mean(transfer_values),
        "mean_step_margin_shift": _mean(shift_values),
        "n_steps_evaluated": len(ordered),
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
    _assert_trace_ok(source_trace, cfg, "source")

    clean_trace = wrapper.adapter.generate_with_trace(
        wrapper,
        target.image,
        target.question,
        **_trace_kwargs(local, audit_cfg),
    )
    _assert_trace_ok(clean_trace, cfg, "clean")

    source_answer = parse_candidate(source.answer)
    target_answer = parse_candidate(target.answer)
    clean_answer = parse_candidate(clean_trace.get("generated_text"))
    n_captured = int(source_trace.get("n_captured_latent_states") or 0)

    step_results = []
    if "each" in patch_steps:
        state_items = _captured_output_states(source_trace)
        if not state_items:
            raise RuntimeError("source trace captured no per-step latent states")
        for item in state_items:
            step_index = int(item.get("step_index", len(step_results)))
            patched_trace = wrapper.adapter.generate_with_trace(
                wrapper,
                target.image,
                target.question,
                **_trace_kwargs(
                    local,
                    audit_cfg,
                    patch_states=[item["tensor"]],
                    patch_steps=[step_index],
                ),
            )
            _assert_trace_ok(patched_trace, cfg, f"patched_step_{step_index}")
            step_results.append(_patch_result(
                patched_trace=patched_trace,
                clean_trace=clean_trace,
                wrapper=wrapper,
                source_answer_text=source.answer,
                target_answer_text=target.answer,
                source_answer=source_answer,
                clean_answer=clean_answer,
                step_index=step_index,
                source_state_shape=item.get("shape"),
            ))
        patched_summary = sorted(step_results, key=lambda item: int(item.get("step_index") or 0))[-1]
    else:
        selected_items = _select_patch_state_items(source_trace, patch_steps)
        if not selected_items:
            raise RuntimeError("source trace captured no selected patchable latent states")
        selected_step_ids = [
            int(item["step_index"])
            for item in selected_items
            if item.get("step_index") is not None
        ]
        patched_trace = wrapper.adapter.generate_with_trace(
            wrapper,
            target.image,
            target.question,
            **_trace_kwargs(
                local,
                audit_cfg,
                patch_states=[item["tensor"] for item in selected_items],
                patch_steps=selected_step_ids or None,
            ),
        )
        _assert_trace_ok(patched_trace, cfg, "patched")
        patched_summary = _patch_result(
            patched_trace=patched_trace,
            clean_trace=clean_trace,
            wrapper=wrapper,
            source_answer_text=source.answer,
            target_answer_text=target.answer,
            source_answer=source_answer,
            clean_answer=clean_answer,
            step_index=(
                int(selected_items[-1].get("step_index"))
                if selected_items[-1].get("step_index") is not None
                else None
            ),
            source_state_shape=selected_items[-1].get("shape"),
        )
        step_results = [patched_summary]

    step_summary = _record_step_summary(step_results)

    return {
        "id": sample.id,
        "paired_id": sample.paired_id,
        "source_answer": source_answer or str(source.answer),
        "target_answer": target_answer or str(target.answer),
        "clean_generated_text": clean_trace.get("generated_text"),
        "patched_generated_text": patched_summary.get("patched_generated_text"),
        "clean_answer": clean_answer,
        "patched_answer": patched_summary.get("patched_answer"),
        "answer_transferred": (
            patched_summary.get("patched_answer") is not None
            and source_answer is not None
            and patched_summary.get("patched_answer") == source_answer
        ),
        "clean_margin": patched_summary.get("clean_margin"),
        "patched_margin": patched_summary.get("patched_margin"),
        "latent_margin_shift": patched_summary.get("latent_margin_shift"),
        "trace_quality": patched_summary.get("trace_quality"),
        "source_trace_quality": source_trace.get("trace_quality"),
        "missing_modules": patched_summary.get("missing_modules"),
        "trace_v2_error": patched_summary.get("trace_v2_error"),
        "n_lvr_mode_steps": int(patched_summary.get("n_lvr_mode_steps") or 0),
        "n_hidden_feedback_steps": int(patched_summary.get("n_hidden_feedback_steps") or 0),
        "n_captured_latent_states": n_captured,
        "n_patch_applied": sum(1 for item in step_results if int(item.get("n_patch_applied") or 0) >= 1),
        "n_step_patches_attempted": len(step_results),
        "captured_state_shapes": [item.get("shape") for item in source_trace.get("captured_state_metadata", [])],
        "patched_captured_state_shapes": patched_summary.get("patched_captured_state_shapes") or [],
        "patch_steps": patch_steps,
        "step_results": step_results,
        "reduction": {
            "latent_answer_transfer_rate": float(
                patched_summary.get("patched_answer") is not None
                and source_answer is not None
                and patched_summary.get("patched_answer") == source_answer
            ),
            "latent_margin_shift": patched_summary.get("latent_margin_shift"),
            **step_summary,
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
    step_rows = [
        step
        for record in valid
        for step in (record.get("step_results") or [])
        if step.get("step_index") is not None
    ]
    per_step = []
    for step_index in sorted({int(step["step_index"]) for step in step_rows}):
        rows = [step for step in step_rows if int(step["step_index"]) == step_index]
        step_transfers = [
            float(bool(row.get("answer_transferred")))
            for row in rows
            if row.get("answer_transferred") is not None
        ]
        step_shifts = [
            float(row["latent_margin_shift"])
            for row in rows
            if row.get("latent_margin_shift") is not None
        ]
        per_step.append({
            "step_index": int(step_index),
            "n": len(rows),
            "transfer_rate": _mean(step_transfers),
            "mean_margin_shift": _mean(step_shifts),
        })
    valid_step_rates = [
        row for row in per_step
        if row.get("transfer_rate") is not None
    ]
    best_step = None
    if valid_step_rates:
        best_step = max(valid_step_rates, key=lambda row: (float(row["transfer_rate"]), -int(row["step_index"])))
    last_step = valid_step_rates[-1] if valid_step_rates else None
    if len(valid_step_rates) == 1:
        step_auc = float(valid_step_rates[0]["transfer_rate"])
    elif len(valid_step_rates) > 1:
        x = np.asarray([float(row["step_index"]) for row in valid_step_rates], dtype=float)
        y = np.asarray([float(row["transfer_rate"]) for row in valid_step_rates], dtype=float)
        denom = float(x[-1] - x[0])
        step_auc = float(np.trapz(y, x=x) / denom) if denom > 0 else float(np.mean(y))
    else:
        step_auc = None
    return {
        "latent_answer_transfer_rate": _mean(transfers),
        "latent_margin_shift": _mean(shifts),
        "best_step_transfer_rate": best_step.get("transfer_rate") if best_step else None,
        "best_step_index": best_step.get("step_index") if best_step else None,
        "step_transfer_auc": step_auc,
        "last_step_transfer_rate": last_step.get("transfer_rate") if last_step else None,
        "n_steps_evaluated": len(per_step),
        "per_step": per_step,
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
        except TracePolicyViolation:
            raise
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
            "patch_shape_policy": str(local.get("patch_shape_policy", "strict")),
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
