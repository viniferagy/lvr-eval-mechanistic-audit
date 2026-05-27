"""Offline diagnostics for BF-Patch and BF-Swap usage artifacts.

The core BF metrics intentionally keep a small set of primary scalars.  This
module expands the saved per-record artifacts into diagnostic summaries that
answer why answer transfer and continuous margins sometimes disagree.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from pipeline.metrics.v2.bf_patch_answer_transfer import continuous_margin_shift_from_record


BF_METRICS = {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}
BOUNDARY_BINS = (
    ("near", 0.0, 0.1),
    ("medium", 0.1, 0.5),
    ("far", 0.5, float("inf")),
)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _close(a: float | None, b: float | None, *, atol: float = 1e-8) -> bool:
    return a is not None and b is not None and abs(a - b) <= atol


def _has_candidate_sequence_logprobs(record: dict) -> bool:
    return (
        record.get("clean_source_logprob") is not None
        and record.get("clean_target_logprob") is not None
        and record.get("patched_source_logprob") is not None
        and record.get("patched_target_logprob") is not None
    )


def margin_source_for_record(record: dict, shift: float | None) -> str:
    explicit = record.get("margin_source")
    if explicit:
        return str(explicit)
    patched_source = record.get("patched_margin_source")
    clean_source = record.get("clean_margin_source")
    if patched_source or clean_source:
        return str(patched_source or clean_source)
    if shift is None:
        legacy = finite_float(record.get("logprob_margin_shift"))
        if legacy is not None:
            return "parsed_answer_fallback_legacy"
        return "no_continuous_margin"
    if _has_candidate_sequence_logprobs(record):
        return "candidate_sequence_logprob_legacy"
    if _close(shift, finite_float(record.get("generation_score_margin_shift"))):
        return "generation_scores"
    if _close(shift, finite_float(record.get("latent_logit_margin_shift"))):
        return "latent_logit_lens"
    return "continuous_margin_unknown"


def boundary_bin(clean_margin: float | None) -> str | None:
    if clean_margin is None:
        return None
    abs_margin = abs(clean_margin)
    for label, low, high in BOUNDARY_BINS:
        if low <= abs_margin < high:
            return label
    return None


def record_rows_from_envelope(envelope: dict, *, include_controls: bool = True) -> list[dict]:
    metric_id = str(envelope.get("metric_id") or "")
    if metric_id not in BF_METRICS:
        return []
    payload = envelope.get("payload") or {}
    if not isinstance(payload, dict):
        return []

    model = str(envelope.get("model") or payload.get("model") or "unknown")
    task = str(
        envelope.get("task")
        or payload.get("task")
        or (payload.get("config") or {}).get("task")
        or "unknown"
    )
    source_run_dir = envelope.get("source_run_dir") or payload.get("source_run_dir")
    source_run_root = envelope.get("source_run_root") or payload.get("source_run_root")
    source_file = envelope.get("source_file")

    groups: list[tuple[str, list[dict]]] = [("primary", list(payload.get("cells") or []))]
    if include_controls:
        groups.append(("control", list(payload.get("control_cells") or [])))

    rows: list[dict] = []
    for group, cells in groups:
        for cell_idx, cell in enumerate(cells):
            control = cell.get("control") if group == "control" else None
            for record_idx, record in enumerate(cell.get("records") or []):
                if not isinstance(record, dict) or record.get("error") is not None:
                    continue
                shift = continuous_margin_shift_from_record(record)
                clean_margin = finite_float(record.get("clean_margin"))
                patched_margin = finite_float(record.get("patched_margin"))
                legacy_shift = finite_float(record.get("logprob_margin_shift"))
                effective_shift = finite_float(record.get("effective_margin_shift"))
                if effective_shift is None:
                    effective_shift = finite_float(record.get("latent_margin_shift"))
                transferred = bool_or_none(record.get("answer_transferred"))
                source_token_ids = record.get("source_answer_token_ids") or []
                target_token_ids = record.get("target_answer_token_ids") or []
                try:
                    source_token_len = len(source_token_ids)
                except TypeError:
                    source_token_len = None
                try:
                    target_token_len = len(target_token_ids)
                except TypeError:
                    target_token_len = None
                rows.append({
                    "metric_id": metric_id,
                    "model": model,
                    "task": task,
                    "source_run_dir": source_run_dir,
                    "source_run_root": source_run_root,
                    "source_file": source_file,
                    "group": group,
                    "control": str(control) if control is not None else "",
                    "cell_index": cell_idx,
                    "record_index": record_idx,
                    "layer": cell.get("layer", record.get("layer")),
                    "position_bucket": cell.get("position_bucket", record.get("position_bucket")),
                    "id": record.get("id"),
                    "paired_id": record.get("paired_id"),
                    "clean_margin": clean_margin,
                    "patched_margin": patched_margin,
                    "abs_clean_margin": abs(clean_margin) if clean_margin is not None else None,
                    "boundary_bin": boundary_bin(clean_margin),
                    "continuous_margin_shift": shift,
                    "legacy_logprob_margin_shift": legacy_shift,
                    "effective_margin_shift": effective_shift,
                    "latent_logit_margin_shift": finite_float(record.get("latent_logit_margin_shift")),
                    "generation_score_margin_shift": finite_float(record.get("generation_score_margin_shift")),
                    "margin_source": margin_source_for_record(record, shift),
                    "clean_margin_source": record.get("clean_margin_source"),
                    "patched_margin_source": record.get("patched_margin_source"),
                    "answer_transferred": transferred,
                    "direction_correct": (shift > 0.0) if shift is not None else None,
                    "source_answer": record.get("source_answer"),
                    "target_answer": record.get("target_answer"),
                    "source_answer_token_len": source_token_len,
                    "target_answer_token_len": target_token_len,
                    "has_continuous_margin": shift is not None,
                    "has_effective_only_margin": shift is None and effective_shift is not None,
                })
    return rows


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarize_rows(rows: Iterable[dict]) -> dict:
    items = list(rows)
    shifts = [
        float(row["continuous_margin_shift"])
        for row in items
        if row.get("continuous_margin_shift") is not None
    ]
    transfers = [
        1.0 if row.get("answer_transferred") else 0.0
        for row in items
        if row.get("answer_transferred") is not None
    ]
    directions = [1.0 if shift > 0.0 else 0.0 for shift in shifts]
    abs_shifts = [abs(shift) for shift in shifts]
    return {
        "n_records": len(items),
        "n_shift": len(shifts),
        "n_missing_shift": len(items) - len(shifts),
        "continuous_margin_coverage": (len(shifts) / len(items)) if items else None,
        "mean_signed_shift": _mean(shifts),
        "mean_abs_shift": _mean(abs_shifts),
        "median_shift": _median(shifts),
        "directional_accuracy": _mean(directions),
        "answer_transfer_rate": _mean(transfers),
        "n_transfer": len(transfers),
    }


def agreement_counts(rows: Iterable[dict]) -> dict:
    counts = {
        "transfer_yes_direction_correct": 0,
        "transfer_yes_direction_wrong_or_zero": 0,
        "transfer_no_direction_correct": 0,
        "transfer_no_direction_wrong_or_zero": 0,
        "n_agreement_rows": 0,
    }
    for row in rows:
        transferred = row.get("answer_transferred")
        shift = row.get("continuous_margin_shift")
        if transferred is None or shift is None:
            continue
        correct = float(shift) > 0.0
        counts["n_agreement_rows"] += 1
        if transferred and correct:
            counts["transfer_yes_direction_correct"] += 1
        elif transferred:
            counts["transfer_yes_direction_wrong_or_zero"] += 1
        elif correct:
            counts["transfer_no_direction_correct"] += 1
        else:
            counts["transfer_no_direction_wrong_or_zero"] += 1
    return counts


def grouped_summaries(rows: Iterable[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            value = "unknown"
        groups[str(value)].append(row)
    out = []
    for value in sorted(groups):
        out.append({key: value, **summarize_rows(groups[value])})
    return out


def boundary_summaries(rows: Iterable[dict]) -> list[dict]:
    items = list(rows)
    out = []
    for label, _low, _high in BOUNDARY_BINS:
        out.append({
            "boundary_bin": label,
            **summarize_rows([row for row in items if row.get("boundary_bin") == label]),
        })
    missing = [row for row in items if row.get("boundary_bin") is None]
    if missing:
        out.append({"boundary_bin": "unknown", **summarize_rows(missing)})
    return out


def cell_summaries(rows: Iterable[dict]) -> list[dict]:
    groups: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("metric_id"),
            row.get("task"),
            row.get("model"),
            row.get("source_run_root"),
            row.get("group"),
            row.get("control") or "",
            row.get("layer"),
            row.get("position_bucket"),
        )
        groups[key].append(row)
    out = []
    for key in sorted(groups, key=lambda item: tuple(str(v) for v in item)):
        metric_id, task, model, source_run_root, group, control, layer, bucket = key
        out.append({
            "metric_id": metric_id,
            "task": task,
            "model": model,
            "source_run_root": source_run_root,
            "group": group,
            "control": control,
            "layer": layer,
            "position_bucket": bucket,
            **summarize_rows(groups[key]),
        })
    return out


def control_normalized_summaries(rows: Iterable[dict]) -> list[dict]:
    items = [row for row in rows if row.get("metric_id") == "bf_swap_latent_replacement"]
    if not items:
        return []
    primary: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    controls: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for row in items:
        key = (
            row.get("task"),
            row.get("model"),
            row.get("source_run_root"),
            row.get("layer"),
            row.get("position_bucket"),
        )
        if row.get("group") == "primary":
            primary[key].append(row)
        elif row.get("group") == "control":
            controls[(*key, row.get("control") or "unknown")].append(row)

    out = []
    for key in sorted(primary, key=lambda item: tuple(str(v) for v in item)):
        task, model, source_run_root, layer, bucket = key
        primary_summary = summarize_rows(primary[key])
        for control_key, control_rows in sorted(controls.items(), key=lambda item: tuple(str(v) for v in item[0])):
            if control_key[:5] != key:
                continue
            control = control_key[5]
            control_summary = summarize_rows(control_rows)
            primary_signed = primary_summary.get("mean_signed_shift")
            control_signed = control_summary.get("mean_signed_shift")
            primary_abs = primary_summary.get("mean_abs_shift")
            control_abs = control_summary.get("mean_abs_shift")
            abs_ratio = None
            if primary_abs is not None and control_abs is not None:
                abs_ratio = primary_abs / (abs(control_abs) + 1e-12)
            out.append({
                "metric_id": "bf_swap_latent_replacement",
                "task": task,
                "model": model,
                "source_run_root": source_run_root,
                "layer": layer,
                "position_bucket": bucket,
                "control": control,
                "primary_n_shift": primary_summary.get("n_shift"),
                "control_n_shift": control_summary.get("n_shift"),
                "primary_mean_signed_shift": primary_signed,
                "control_mean_signed_shift": control_signed,
                "signed_shift_minus_control": (
                    primary_signed - control_signed
                    if primary_signed is not None and control_signed is not None
                    else None
                ),
                "primary_mean_abs_shift": primary_abs,
                "control_mean_abs_shift": control_abs,
                "abs_shift_minus_control": (
                    primary_abs - control_abs
                    if primary_abs is not None and control_abs is not None
                    else None
                ),
                "abs_specificity_ratio": abs_ratio,
                "primary_answer_transfer_rate": primary_summary.get("answer_transfer_rate"),
                "control_answer_transfer_rate": control_summary.get("answer_transfer_rate"),
            })
    return out


def summarize_envelope(envelope: dict) -> dict | None:
    metric_id = str(envelope.get("metric_id") or "")
    if metric_id not in BF_METRICS:
        return None
    all_rows = record_rows_from_envelope(envelope, include_controls=True)
    primary_rows = [row for row in all_rows if row.get("group") == "primary"]
    if not primary_rows:
        return None
    base = {
        "metric_id": metric_id,
        "task": primary_rows[0].get("task"),
        "model": primary_rows[0].get("model"),
        "source_run_dir": primary_rows[0].get("source_run_dir"),
        "source_run_root": primary_rows[0].get("source_run_root"),
        "source_file": primary_rows[0].get("source_file"),
    }
    return {
        **base,
        **summarize_rows(primary_rows),
        "agreement": agreement_counts(primary_rows),
        "boundary_bins": boundary_summaries(primary_rows),
        "margin_sources": grouped_summaries(primary_rows, "margin_source"),
        "cell_summaries": cell_summaries(primary_rows),
        "control_normalized": control_normalized_summaries(all_rows),
    }


def build_diagnostics(envelopes: Iterable[dict]) -> dict:
    envelope_list = [env for env in envelopes if str(env.get("metric_id") or "") in BF_METRICS]
    record_rows: list[dict] = []
    summaries = []
    for envelope in envelope_list:
        record_rows.extend(record_rows_from_envelope(envelope, include_controls=True))
        summary = summarize_envelope(envelope)
        if summary is not None:
            summaries.append(summary)
    primary_rows = [row for row in record_rows if row.get("group") == "primary"]
    return {
        "version": "bf_usage_diagnostics_v0",
        "n_metric_envelopes": len(envelope_list),
        "n_record_rows": len(record_rows),
        "overall_primary": summarize_rows(primary_rows),
        "summaries": summaries,
        "record_rows": record_rows,
        "cell_rows": cell_summaries(record_rows),
        "boundary_rows": [
            {
                "metric_id": summary["metric_id"],
                "task": summary["task"],
                "model": summary["model"],
                "source_run_root": summary.get("source_run_root"),
                **row,
            }
            for summary in summaries
            for row in summary["boundary_bins"]
        ],
        "margin_source_rows": [
            {
                "metric_id": summary["metric_id"],
                "task": summary["task"],
                "model": summary["model"],
                "source_run_root": summary.get("source_run_root"),
                **row,
            }
            for summary in summaries
            for row in summary["margin_sources"]
        ],
        "control_normalized_rows": [
            row
            for summary in summaries
            for row in summary["control_normalized"]
        ],
    }
