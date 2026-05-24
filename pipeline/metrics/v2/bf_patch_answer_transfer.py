"""BF-Patch answer transfer via paired hidden-state patching."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import Any

import numpy as np

from ...adapters.spans import TokenSpan
from ..base import MetricSpec

try:
    import torch
    _no_grad = torch.no_grad
except ImportError:
    torch = None

    def _no_grad():
        def deco(fn):
            return fn
        return deco


METRIC_ID = "bf_patch_answer_transfer"
LEGACY_NAME = "bf_patch"
DEFAULT_LAYERS = [0, 7, 14, 21, 27]
DEFAULT_POSITION_BUCKETS = ["image", "query", "latent"]


def patch_grid(layers=None, position_buckets=None) -> list[dict]:
    layers = list(DEFAULT_LAYERS if layers is None else layers)
    buckets = list(DEFAULT_POSITION_BUCKETS if position_buckets is None else position_buckets)
    return [{"layer": int(layer), "position_bucket": bucket} for layer in layers for bucket in buckets]


def answer_transfer_rate(records: list[dict]) -> float | None:
    valid = [r for r in records if r.get("answer_transferred") is not None]
    if not valid:
        return None
    return sum(1 for r in valid if bool(r["answer_transferred"])) / len(valid)


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "grid": patch_grid(),
        "scalars": ["logprob_margin_shift", "answer_transfer_rate", "n_paired"],
        "status": "runnable_v0",
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("bf_patch", {}) or {}


def _to_int_list(values, default) -> list[int]:
    return [int(v) for v in (default if values is None else values)]


def _to_str_list(values, default) -> list[str]:
    return [str(v) for v in (default if values is None else values)]


def _decode_token(wrapper, token_id: int) -> str:
    tokenizer = getattr(wrapper.processor, "tokenizer", wrapper.processor)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode([int(token_id)]).strip()


def _answer_token_id(wrapper, answer: Any) -> int | None:
    if answer is None:
        return None
    text = str(answer).strip()
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
    if not ids:
        return None
    return int(ids[0])


def _answer_token_ids(wrapper, answer: Any) -> list[int]:
    if answer is None:
        return []
    text = str(answer).strip()
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


def _token_logit(logits, token_id: int | None) -> float | None:
    if token_id is None:
        return None
    return float(logits[int(token_id)].detach().float().cpu().item())


def _answer_margin(logits, source_token_id: int | None, target_token_id: int | None) -> float | None:
    source_logit = _token_logit(logits, source_token_id)
    target_logit = _token_logit(logits, target_token_id)
    if source_logit is None or target_logit is None:
        return None
    return source_logit - target_logit


def _argmax_answer(logits, wrapper, candidate_ids: list[int | None]) -> tuple[str | None, int | None]:
    valid_ids = [int(v) for v in candidate_ids if v is not None]
    if valid_ids:
        candidate_logits = logits[valid_ids].detach().float()
        token_id = valid_ids[int(candidate_logits.argmax().item())]
    else:
        token_id = int(logits.detach().float().argmax().item())
    return _decode_token(wrapper, token_id), token_id


def _append_answer_tokens(inputs, answer_token_ids: list[int]):
    if torch is None or not answer_token_ids:
        return inputs
    out = dict(inputs)
    input_ids = out.get("input_ids")
    if input_ids is None:
        return inputs
    answer = torch.tensor([answer_token_ids], dtype=input_ids.dtype, device=input_ids.device)
    out["input_ids"] = torch.cat([input_ids, answer], dim=1)
    attention_mask = out.get("attention_mask")
    if attention_mask is not None:
        out["attention_mask"] = torch.cat([attention_mask, torch.ones_like(answer)], dim=1)
    return out


def _sequence_logprob(wrapper, inputs, spans, answer_token_ids: list[int], *,
                      patch_layer: int | None = None,
                      patch_slice: slice | None = None,
                      replacement=None) -> float | None:
    if torch is None or not answer_token_ids:
        return None
    scored_inputs = _append_answer_tokens(inputs, answer_token_ids)
    prompt_len = int(inputs["input_ids"].shape[1])
    context = (
        _patch_layer_hidden(wrapper, patch_layer, patch_slice, replacement)
        if patch_layer is not None and patch_slice is not None and replacement is not None
        else nullcontext()
    )
    with context:
        output = _model_forward(wrapper, scored_inputs)
    logits = output.logits[0].float()
    total = 0.0
    for idx, token_id in enumerate(answer_token_ids):
        pos = prompt_len + idx - 1
        if pos < 0 or pos >= logits.shape[0]:
            return None
        total += float(torch.log_softmax(logits[pos], dim=-1)[int(token_id)].detach().cpu().item())
    return total


def _candidate_logprobs(wrapper, inputs, spans, source_ids: list[int], target_ids: list[int], *,
                        patch_layer: int | None = None,
                        patch_slice: slice | None = None,
                        replacement=None) -> dict:
    source_lp = _sequence_logprob(
        wrapper,
        inputs,
        spans,
        source_ids,
        patch_layer=patch_layer,
        patch_slice=patch_slice,
        replacement=replacement,
    )
    target_lp = _sequence_logprob(
        wrapper,
        inputs,
        spans,
        target_ids,
        patch_layer=patch_layer,
        patch_slice=patch_slice,
        replacement=replacement,
    )
    margin = source_lp - target_lp if source_lp is not None and target_lp is not None else None
    return {"source_logprob": source_lp, "target_logprob": target_lp, "margin": margin}


def _span_for_bucket(spans, bucket: str) -> TokenSpan:
    if bucket == "image":
        return spans.image_tokens
    if bucket == "query":
        return spans.preferred_query_span()
    if bucket == "latent":
        if spans.latent_tokens is not None:
            return spans.latent_tokens
        if spans.lvr_placeholder_tokens is not None:
            return spans.lvr_placeholder_tokens
        return spans.preferred_query_span()
    raise ValueError(f"unknown BF-Patch position bucket: {bucket}")


def _aligned_slices(source_span: TokenSpan, target_span: TokenSpan) -> tuple[slice, slice, int]:
    length = min(source_span.length, target_span.length)
    if length <= 0:
        raise ValueError(f"empty patch span: source={source_span} target={target_span}")
    return (
        slice(source_span.start, source_span.start + length),
        slice(target_span.start, target_span.start + length),
        int(length),
    )


def _model_forward(wrapper, inputs):
    try:
        return wrapper.model(
            **inputs,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
            use_cache=False,
        )
    except TypeError:
        return wrapper.model(
            **inputs,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )


def _answer_probe_logits(output, spans):
    pos = spans.answer_probe_pos
    if pos is None:
        pos = spans.preferred_query_span().end - 1
    return output.logits[0, int(pos), :]


@contextmanager
def _capture_layer_hidden(wrapper, layer_idx: int, token_slice: slice):
    captured = {"hidden": None}

    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden[:, token_slice, :].detach()
        return output

    handle = wrapper.layers[layer_idx].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def _patch_layer_hidden(wrapper, layer_idx: int, token_slice: slice, replacement):
    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        patched = hidden.clone()
        repl = replacement.to(device=hidden.device, dtype=hidden.dtype)
        length = min(patched[:, token_slice, :].shape[1], repl.shape[1])
        patched[:, slice(token_slice.start, token_slice.start + length), :] = repl[:, :length, :]
        return patched if rest is None else (patched,) + rest

    handle = wrapper.layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _sample_with_counterfactual(sample):
    return replace(
        sample,
        image=sample.counterfactual_image,
        answer=sample.counterfactual_answer,
    )


@_no_grad()
def patch_one_pair(wrapper, sample, *, layer: int, position_bucket: str) -> dict:
    if sample.counterfactual_image is None or sample.counterfactual_answer is None:
        raise ValueError("sample is missing counterfactual_image or counterfactual_answer")
    if layer < 0 or layer >= wrapper.n_layers:
        raise ValueError(f"layer {layer} outside model layer range 0..{wrapper.n_layers - 1}")

    source = _sample_with_counterfactual(sample)
    target = sample
    source_inputs = wrapper.build_inputs_from_sample(source)
    target_inputs = wrapper.build_inputs_from_sample(target)
    source_spans = wrapper.adapter.get_spans(wrapper, source_inputs, None)
    target_spans = wrapper.adapter.get_spans(wrapper, target_inputs, None)
    source_span = _span_for_bucket(source_spans, position_bucket)
    target_span = _span_for_bucket(target_spans, position_bucket)
    source_slice, target_slice, patch_length = _aligned_slices(source_span, target_span)

    source_token_id = _answer_token_id(wrapper, source.answer)
    target_token_id = _answer_token_id(wrapper, target.answer)
    source_answer_token_ids = _answer_token_ids(wrapper, source.answer)
    target_answer_token_ids = _answer_token_ids(wrapper, target.answer)

    with _capture_layer_hidden(wrapper, layer, source_slice) as captured:
        _model_forward(wrapper, source_inputs)
    replacement = captured["hidden"]
    if replacement is None:
        raise RuntimeError(f"BF-Patch did not capture source hidden state at layer {layer}")

    clean_scores = _candidate_logprobs(
        wrapper,
        target_inputs,
        target_spans,
        source_answer_token_ids,
        target_answer_token_ids,
    )
    target_clean = _model_forward(wrapper, target_inputs)
    clean_logits = _answer_probe_logits(target_clean, target_spans)
    clean_token_margin = _answer_margin(clean_logits, source_token_id, target_token_id)
    patched_scores = _candidate_logprobs(
        wrapper,
        target_inputs,
        target_spans,
        source_answer_token_ids,
        target_answer_token_ids,
        patch_layer=layer,
        patch_slice=target_slice,
        replacement=replacement,
    )
    with _patch_layer_hidden(wrapper, layer, target_slice, replacement):
        patched_out = _model_forward(wrapper, target_inputs)
    patched_logits = _answer_probe_logits(patched_out, target_spans)
    clean_margin = clean_scores["margin"]
    patched_margin = patched_scores["margin"]
    patched_answer, patched_token_id = _argmax_answer(
        patched_logits,
        wrapper,
        [source_token_id, target_token_id],
    )
    diagnostic_token_margin = _answer_margin(patched_logits, source_token_id, target_token_id)
    answer_transferred = patched_scores["margin"] is not None and patched_scores["margin"] > 0
    logprob_margin_shift = (
        patched_scores["margin"] - clean_scores["margin"]
        if patched_scores["margin"] is not None and clean_scores["margin"] is not None
        else None
    )

    return {
        "id": sample.id,
        "paired_id": sample.paired_id,
        "random_pair_source_id": (sample.task_metadata or {}).get("random_pair_source_id"),
        "random_pair_source_paired_id": (sample.task_metadata or {}).get("random_pair_source_paired_id"),
        "layer": int(layer),
        "position_bucket": position_bucket,
        "source_answer": str(source.answer),
        "target_answer": str(target.answer),
        "patched_answer": patched_answer,
        "answer_transferred": answer_transferred,
        "patched_token_id": patched_token_id,
        "source_token_id": source_token_id,
        "target_token_id": target_token_id,
        "source_answer_token_ids": source_answer_token_ids,
        "target_answer_token_ids": target_answer_token_ids,
        "clean_source_logprob": clean_scores["source_logprob"],
        "clean_target_logprob": clean_scores["target_logprob"],
        "patched_source_logprob": patched_scores["source_logprob"],
        "patched_target_logprob": patched_scores["target_logprob"],
        "clean_margin": clean_margin,
        "patched_margin": patched_margin,
        "logprob_margin_shift": logprob_margin_shift,
        "logit_margin_shift": (
            diagnostic_token_margin - clean_token_margin
            if diagnostic_token_margin is not None and clean_token_margin is not None
            else None
        ),
        "clean_token_margin": clean_token_margin,
        "diagnostic_token_margin": diagnostic_token_margin,
        "patch_length": patch_length,
        "source_span": [source_span.start, source_span.end],
        "target_span": [target_span.start, target_span.end],
    }


def _cell_summary(cell: dict, records: list[dict]) -> dict:
    valid_shifts = [
        float(r["logprob_margin_shift"])
        for r in records
        if r.get("logprob_margin_shift") is not None
    ]
    return {
        **cell,
        "logprob_margin_shift": float(np.mean(valid_shifts)) if valid_shifts else None,
        "logit_margin_shift": float(np.mean([
            float(r["logit_margin_shift"])
            for r in records
            if r.get("logit_margin_shift") is not None
        ])) if any(r.get("logit_margin_shift") is not None for r in records) else None,
        "answer_transfer_rate": answer_transfer_rate(records),
        "n_paired": len(records),
        "n_success": sum(1 for r in records if r.get("error") is None),
        "n_error": sum(1 for r in records if r.get("error") is not None),
        "records": records,
    }


def reduce_cells(cells: list[dict]) -> dict | None:
    shifts = [float(c["logprob_margin_shift"]) for c in cells if c.get("logprob_margin_shift") is not None]
    transfers = [
        float(c["answer_transfer_rate"])
        for c in cells
        if c.get("answer_transfer_rate") is not None
    ]
    if not shifts and not transfers:
        return None
    return {
        "logprob_margin_shift": float(np.mean(shifts)) if shifts else None,
        "logit_margin_shift": float(np.mean([
            float(c["logit_margin_shift"]) for c in cells if c.get("logit_margin_shift") is not None
        ])) if any(c.get("logit_margin_shift") is not None for c in cells) else None,
        "answer_transfer_rate": float(np.mean(transfers)) if transfers else None,
        "n_paired": int(max((c.get("n_paired", 0) for c in cells), default=0)),
        "n_cells": len(cells),
    }


def _flatten_sample_records(cells: list[dict]) -> list[dict]:
    samples = []
    for cell in cells:
        for record in cell.get("records", []):
            if record.get("error") is not None:
                continue
            transferred = record.get("answer_transferred")
            samples.append({
                "id": record.get("id"),
                "paired_id": record.get("paired_id"),
                "layer": cell.get("layer"),
                "position_bucket": cell.get("position_bucket"),
                "reduction": {
                    "logprob_margin_shift": record.get("logprob_margin_shift"),
                    "logit_margin_shift": record.get("logit_margin_shift"),
                    "answer_transfer": (
                        float(bool(transferred)) if transferred is not None else None
                    ),
                },
            })
    return samples


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    bf_cfg = _cfg(cfg)
    layers = _to_int_list(bf_cfg.get("layers"), DEFAULT_LAYERS)
    buckets = _to_str_list(bf_cfg.get("position_buckets"), DEFAULT_POSITION_BUCKETS)
    paired = [
        sample for sample in samples
        if sample.counterfactual_image is not None and sample.counterfactual_answer is not None
    ]
    max_pairs = bf_cfg.get("max_pairs")
    if max_pairs is not None:
        paired = paired[:int(max_pairs)]

    cells = []
    for cell in patch_grid(layers, buckets):
        records = []
        for sample in paired:
            try:
                records.append(patch_one_pair(
                    wrapper,
                    sample,
                    layer=cell["layer"],
                    position_bucket=cell["position_bucket"],
                ))
            except Exception as exc:  # noqa: BLE001
                records.append({
                    "id": sample.id,
                    "paired_id": sample.paired_id,
                    "layer": cell["layer"],
                    "position_bucket": cell["position_bucket"],
                    "source_answer": str(sample.counterfactual_answer),
                    "target_answer": str(sample.answer),
                    "patched_answer": None,
                    "logprob_margin_shift": None,
                    "logit_margin_shift": None,
                    "error": repr(exc),
                })
        cells.append(_cell_summary(cell, records))

    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "layers": layers,
            "position_buckets": buckets,
            "max_pairs": max_pairs,
            "source": "counterfactual",
            "target": "clean",
        },
        "cells": cells,
        "samples": _flatten_sample_records(cells),
        "reduction": reduce_cells(cells),
        "n_paired": len(paired),
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="BF-Patch Answer Transfer",
    kind="internal_curve",
    run_fn=run,
)
