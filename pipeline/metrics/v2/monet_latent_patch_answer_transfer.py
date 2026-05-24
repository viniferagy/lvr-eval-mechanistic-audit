"""W13 Monet Transformers latent-state answer transfer.

This metric audits Monet's official Transformers latent-mode path:

1. run source/counterfactual input with ``latent_mode=True``;
2. capture ``ce_patch_pos`` and ``ce_patch_vec`` hidden-feedback tensors;
3. score the target/clean input after injecting source tensors at Monet latent
   positions;
4. report whether the constrained answer is pushed toward the source answer.

It is a real hidden-state replacement gate, but not the modified-vLLM
scheduler-native generation loop used for Monet paper inference.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

import numpy as np
import torch

from ..base import MetricSpec


METRIC_ID = "monet_latent_patch_answer_transfer"
LEGACY_NAME = "monet_latent_patch"
CANDIDATES = ("original", "modified")


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "scalars": [
            "latent_answer_transfer_rate",
            "latent_margin_shift",
            "n_paired",
            "n_success",
            "n_error",
            "n_patch_applied",
            "n_with_captured_state",
        ],
        "patch_policy": {
            "source": "counterfactual_image_question_answer",
            "target": "clean_image_question_answer",
            "capture_path": "Monet Transformers latent_mode ce_patch_vec",
            "intervention_site": "ce_patch_pos latent token embeddings",
        },
        "status": "runnable_w13_monet_transformers_latent_gate",
        "boundary": "not_vllm_scheduler_native_generation",
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("monet_latent_patch", {}) or {}


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


def _source_sample(sample):
    return replace(sample, image=sample.counterfactual_image, answer=sample.counterfactual_answer)


def _selected_latents(capture: dict, patch_steps: list[str], *, shape_policy: str) -> tuple[list[int], list[torch.Tensor], list[int]]:
    positions = list((capture.get("ce_patch_pos") or [[]])[0])
    vectors = (capture.get("ce_patch_vec") or [None])[0]
    if not torch.is_tensor(vectors) or vectors.ndim != 2:
        raise RuntimeError("source Monet capture missing 2D ce_patch_vec tensor")
    if len(positions) != int(vectors.shape[0]):
        raise RuntimeError("source Monet ce_patch_pos/ce_patch_vec length mismatch")
    if not positions:
        raise RuntimeError("source Monet capture has no latent positions")

    if "all" in patch_steps:
        idxs = list(range(len(positions)))
    elif "last" in patch_steps:
        idxs = [len(positions) - 1]
    else:
        requested = {int(step) for step in patch_steps}
        idxs = [idx for idx in range(len(positions)) if idx in requested or int(positions[idx]) in requested]
    if not idxs:
        raise RuntimeError(f"no Monet latent states selected for patch_steps={patch_steps}")

    selected_pos = [int(positions[idx]) for idx in idxs]
    selected_vec = vectors[idxs, :].detach()
    if shape_policy == "strict" and selected_vec.shape[0] != len(selected_pos):
        raise RuntimeError("strict patch shape policy rejected selected ce_patch_vec/pos mismatch")
    return selected_pos, [selected_vec], idxs


def _score_margin(scores: dict[str, float], source_answer: str | None, target_answer: str | None) -> float | None:
    if source_answer is None or target_answer is None:
        return None
    if source_answer not in scores or target_answer not in scores:
        return None
    return float(scores[source_answer] - scores[target_answer])


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run_one_pair(wrapper, sample, cfg: dict) -> dict:
    local = _cfg(cfg)
    latent_size = int(local.get("latent_size", 10))
    patch_steps = [str(v) for v in local.get("patch_steps", ["all"])]
    shape_policy = str(local.get("patch_shape_policy", "strict"))
    if sample.counterfactual_image is None or sample.counterfactual_answer is None:
        raise ValueError("sample missing counterfactual image/answer")
    if wrapper.arch not in {"monet_qwen2_5_vl", "monet", "monet_qwen"}:
        raise ValueError(f"{METRIC_ID} requires Monet adapter, got arch={wrapper.arch}")
    if not hasattr(wrapper.adapter, "capture_latent_state"):
        raise ValueError("Monet adapter does not expose capture_latent_state")

    source = _source_sample(sample)
    target = sample
    source_answer = parse_candidate(source.answer)
    target_answer = parse_candidate(target.answer)
    if source_answer is None or target_answer is None:
        raise ValueError(f"SPD constrained answers must parse as {CANDIDATES}: {source.answer!r}, {target.answer!r}")

    source_capture = wrapper.adapter.capture_latent_state(
        wrapper,
        source.image,
        source.question,
        latent_size=latent_size,
    )
    target_capture = wrapper.adapter.capture_latent_state(
        wrapper,
        target.image,
        target.question,
        latent_size=latent_size,
    )
    source_patch_pos, patch_vec, patch_indices = _selected_latents(source_capture, patch_steps, shape_policy=shape_policy)
    target_positions = [int(v) for v in (target_capture.get("ce_patch_pos") or [[]])[0]]
    if shape_policy == "strict":
        if len(target_positions) != len((source_capture.get("ce_patch_pos") or [[]])[0]):
            raise RuntimeError(
                "strict patch shape policy rejected source/target latent position count mismatch: "
                f"{len((source_capture.get('ce_patch_pos') or [[]])[0])} vs {len(target_positions)}"
            )
        if patch_vec[0].shape[-1] != int(target_capture.get("hidden_size") or -1):
            raise RuntimeError(
                "strict patch shape policy rejected hidden size mismatch: "
                f"{patch_vec[0].shape[-1]} vs {target_capture.get('hidden_size')}"
            )
    if any(idx >= len(target_positions) for idx in patch_indices):
        raise RuntimeError("selected Monet patch index exceeds target latent positions")
    target_patch_pos = [int(target_positions[idx]) for idx in patch_indices]

    clean_scoring = wrapper.adapter.score_candidates_with_latents(
        wrapper,
        target.image,
        target.question,
        latent_size=latent_size,
        ce_patch_pos=target_capture["ce_patch_pos"],
        ce_patch_vec=target_capture["ce_patch_vec"],
        candidates=CANDIDATES,
    )
    patched_scoring = wrapper.adapter.score_candidates_with_latents(
        wrapper,
        target.image,
        target.question,
        latent_size=latent_size,
        ce_patch_pos=[target_patch_pos],
        ce_patch_vec=patch_vec,
        candidates=CANDIDATES,
    )
    clean_answer = clean_scoring["predicted_answer"]
    patched_answer = patched_scoring["predicted_answer"]
    clean_margin = _score_margin(clean_scoring["scores"], source_answer, target_answer)
    patched_margin = _score_margin(patched_scoring["scores"], source_answer, target_answer)
    margin_shift = (
        patched_margin - clean_margin
        if patched_margin is not None and clean_margin is not None
        else None
    )
    transferred = patched_answer == source_answer

    return {
        "id": sample.id,
        "paired_id": sample.paired_id,
        "source_answer": source_answer,
        "target_answer": target_answer,
        "clean_answer": clean_answer,
        "patched_answer": patched_answer,
        "answer_transferred": bool(transferred),
        "clean_scores": clean_scoring["scores"],
        "patched_scores": patched_scoring["scores"],
        "candidate_token_ids": clean_scoring["token_ids"],
        "clean_margin": clean_margin,
        "patched_margin": patched_margin,
        "latent_margin_shift": margin_shift,
        "trace_quality": source_capture.get("trace_quality"),
        "source_trace_quality": source_capture.get("trace_quality"),
        "target_trace_quality": target_capture.get("trace_quality"),
        "latent_mode_path": source_capture.get("latent_mode_path"),
        "vllm_scheduler_native": False,
        "n_monet_latent_positions": int(source_capture.get("n_monet_latent_positions") or 0),
        "n_captured_latent_states": int(source_capture.get("n_captured_latent_states") or 0),
        "n_patch_applied": len(target_patch_pos),
        "captured_state_shapes": source_capture.get("captured_state_shapes") or [],
        "target_captured_state_shapes": target_capture.get("captured_state_shapes") or [],
        "patch_state_shapes": [list(patch_vec[0].shape)],
        "source_latent_positions": source_capture.get("latent_positions"),
        "target_latent_positions": target_capture.get("latent_positions"),
        "source_patch_positions": source_patch_pos,
        "patch_positions": target_patch_pos,
        "patch_steps": patch_steps,
        "reduction": {
            "latent_answer_transfer_rate": float(bool(transferred)),
            "latent_margin_shift": margin_shift,
        },
    }


def reduce_records(records: list[dict], n_paired: int) -> dict | None:
    valid = [r for r in records if r.get("error") is None]
    transfers = [
        float(bool(record.get("answer_transferred")))
        for record in valid
        if record.get("answer_transferred") is not None
    ]
    shifts = [
        float(record["latent_margin_shift"])
        for record in valid
        if record.get("latent_margin_shift") is not None
    ]
    return {
        "latent_answer_transfer_rate": _mean(transfers),
        "latent_margin_shift": _mean(shifts),
        "n_paired": int(n_paired),
        "n_success": len(valid),
        "n_error": int(n_paired) - len(valid),
        "n_patch_applied": sum(int(record.get("n_patch_applied") or 0) >= 1 for record in valid),
        "n_with_captured_state": sum(int(record.get("n_captured_latent_states") or 0) >= 1 for record in valid),
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
            "latent_size": int(local.get("latent_size", 10)),
            "patch_steps": [str(v) for v in local.get("patch_steps", ["all"])],
            "patch_shape_policy": str(local.get("patch_shape_policy", "strict")),
            "answer_candidates": list(CANDIDATES),
            "latent_mode_path": "transformers_ce_patch_vec",
            "vllm_scheduler_native": False,
        },
        "samples": records,
        "reduction": reduce_records(records, len(paired)),
        "n_paired": len(paired),
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="Monet Latent Patch Answer Transfer",
    kind="internal_curve",
    run_fn=run,
)
