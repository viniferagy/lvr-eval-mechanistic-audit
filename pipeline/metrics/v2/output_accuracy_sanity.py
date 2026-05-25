"""Output-level accuracy sanity metric."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..base import MetricSpec


METRIC_ID = "output_accuracy_sanity"
LEGACY_NAME = "output_acc"


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
        return vals or [str(answer)]
    if isinstance(answer, (list, tuple, set)):
        return [str(item) for item in answer]
    return [str(answer)]


def answer_hit(prediction: Any, answer: Any) -> bool:
    pred = normalize_answer(prediction)
    if not pred:
        return False
    for gold in _answer_values(answer):
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
        "n_error": len(records) - len(valid),
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    local = _cfg(cfg)
    batch_size = int(local.get("batch_size", 1))
    max_new_tokens = int(local.get("max_new_tokens", 64))
    records = []
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
            hit = answer_hit(pred, sample.answer)
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
    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
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
