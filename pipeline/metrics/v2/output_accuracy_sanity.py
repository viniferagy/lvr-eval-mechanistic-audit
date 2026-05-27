"""Output-level accuracy sanity metric."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np

from ..base import MetricSpec


METRIC_ID = "output_accuracy_sanity"
LEGACY_NAME = "output_acc"
logger = logging.getLogger("lvr_eval.metrics.output_accuracy_sanity")


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "scalars": ["accuracy", "n", "n_error"],
        "status": "runnable_sanity_v0",
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("output_accuracy", {}) or {}


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _answer_values(answer: Any) -> list[str]:
    if isinstance(answer, dict):
        vals = []
        for key in ("answer", "label", "text", "value"):
            if answer.get(key) is not None:
                vals.append(str(answer[key]))
        aliases = answer.get("aliases")
        if isinstance(aliases, (list, tuple, set)):
            vals.extend(str(item) for item in aliases if item is not None)
        return vals or [str(answer)]
    if isinstance(answer, (list, tuple, set)):
        return [str(item) for item in answer]
    return [str(answer)]


def _choice_letter(value: Any) -> str | None:
    text = str(value or "").strip()
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return text.upper()
    return None


def _prediction_choice_letters(prediction: Any) -> set[str]:
    """Extract explicit multiple-choice letters from a model response.

    A gold answer of "A" must not be matched by ordinary substring containment:
    nearly every English sentence contains the letter "a". Count only explicit
    answer markers such as "A", "A.", "(A)", "Answer: A", or "option A".
    """
    text = str(prediction or "").strip()
    if not text:
        return set()
    matches: set[str] = set()
    for pattern in (
        r"(?i)(?:^|\b)(?:answer|option|choice|final answer)\s*(?:is|:)?\s*\(?([A-Z])\)?(?:\.|\b)",
        r"(?i)^\s*\(?([A-Z])\)?(?:\.|\)|:|\s|$)",
    ):
        for match in re.finditer(pattern, text):
            matches.add(match.group(1).upper())
    return matches


def _numbers_from_text(value: Any) -> list[float]:
    text = str(value or "")
    out = []
    for match in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", text):
        try:
            out.append(float(match))
        except ValueError:
            continue
    return out


def _numeric_answer_spec(answer: Any, cfg: dict | None = None) -> dict | None:
    cfg = cfg or {}
    if isinstance(answer, dict):
        raw = None
        for key in ("numeric_value", "value", "number", "answer"):
            if answer.get(key) is not None:
                raw = answer.get(key)
                break
        if raw is None:
            return None
        values = _numbers_from_text(raw)
        if not values:
            return None
        value = values[0]
        tolerance = answer.get("tolerance")
        abs_tolerance = answer.get("abs_tolerance", tolerance)
        rel_tolerance = answer.get("rel_tolerance")
        try:
            abs_tol = float(abs_tolerance) if abs_tolerance is not None else float(cfg.get("numeric_abs_tolerance", 1e-3))
        except (TypeError, ValueError):
            abs_tol = float(cfg.get("numeric_abs_tolerance", 1e-3))
        try:
            rel_tol = float(rel_tolerance) if rel_tolerance is not None else float(cfg.get("numeric_rel_tolerance", 0.0))
        except (TypeError, ValueError):
            rel_tol = float(cfg.get("numeric_rel_tolerance", 0.0))
        return {"value": value, "abs_tolerance": abs_tol, "rel_tolerance": rel_tol}
    if isinstance(answer, (int, float)) and np.isfinite(float(answer)):
        return {
            "value": float(answer),
            "abs_tolerance": float((cfg or {}).get("numeric_abs_tolerance", 1e-3)),
            "rel_tolerance": float((cfg or {}).get("numeric_rel_tolerance", 0.0)),
        }
    return None


def _numeric_hit(prediction: Any, answer: Any, cfg: dict | None = None) -> bool:
    spec = _numeric_answer_spec(answer, cfg)
    if spec is None:
        return False
    pred_values = _numbers_from_text(prediction)
    if not pred_values:
        return False
    target = float(spec["value"])
    abs_tol = max(0.0, float(spec.get("abs_tolerance") or 0.0))
    rel_tol = max(0.0, float(spec.get("rel_tolerance") or 0.0))
    tol = max(abs_tol, rel_tol * abs(target))
    return any(abs(float(value) - target) <= tol for value in pred_values)


def answer_hit(prediction: Any, answer: Any, cfg: dict | None = None) -> bool:
    if _numeric_hit(prediction, answer, cfg):
        return True
    pred = normalize_answer(prediction)
    if not pred:
        return False
    for gold in _answer_values(answer):
        letter = _choice_letter(gold)
        if letter is not None:
            if letter in _prediction_choice_letters(prediction):
                return True
            continue
        norm = normalize_answer(gold)
        if norm and (norm == pred or norm in pred):
            return True
    return False


def reduce_records(records: list[dict]) -> dict:
    valid = [record for record in records if record.get("error") is None]
    hits = [float(bool(record.get("hit"))) for record in valid if record.get("hit") is not None]
    return {
        "accuracy": float(np.mean(hits)) if hits else None,
        "n": len(valid),
        "n_total": len(records),
        "n_error": len(records) - len(valid),
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    local = _cfg(cfg)
    batch_size = int(local.get("batch_size", 1))
    max_new_tokens = int(local.get("max_new_tokens", 64))
    progress_every = int(local.get("progress_every", 25))
    records = []
    started = time.monotonic()
    n_total = len(samples)
    for start in range(0, len(samples), max(batch_size, 1)):
        batch = samples[start:start + max(batch_size, 1)]
        try:
            preds = wrapper.generate(
                [sample.image for sample in batch],
                [sample.question for sample in batch],
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            for sample in batch:
                records.append({
                    "id": sample.id,
                    "answer": sample.answer,
                    "error": repr(exc),
                })
            continue
        for sample, pred in zip(batch, preds):
            hit = answer_hit(pred, sample.answer, local)
            records.append({
                "id": sample.id,
                "paired_id": sample.paired_id,
                "prediction": pred,
                "normalized_prediction": normalize_answer(pred),
                "answer": sample.answer,
                "normalized_answer": [normalize_answer(v) for v in _answer_values(sample.answer)],
                "hit": hit,
                "source_dataset": sample.source_dataset,
                "task_metadata": sample.task_metadata or {},
                "reduction": {"accuracy": float(hit)},
            })
        processed = min(start + len(batch), n_total)
        if progress_every > 0 and (processed == n_total or processed % progress_every == 0):
            elapsed = max(time.monotonic() - started, 1e-9)
            n_error = sum(1 for record in records if record.get("error") is not None)
            n_hit = sum(1 for record in records if record.get("error") is None and bool(record.get("hit")))
            logger.info(
                "[progress] output_accuracy_sanity model=%s processed=%d/%d "
                "hits=%d errors=%d rate=%.2f samples/s elapsed=%.1fs",
                model_tag,
                processed,
                n_total,
                n_hit,
                n_error,
                processed / elapsed,
                elapsed,
            )
    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "progress_every": progress_every,
            "source": "model_generate_exact_or_contains_match",
        },
        "samples": records,
        "reduction": reduce_records(records),
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="Output Accuracy Sanity",
    kind="output_sanity",
    run_fn=run,
)
