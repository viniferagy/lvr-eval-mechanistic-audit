"""BLINK loader for fine-grained perception audit samples."""
from __future__ import annotations

import json
import logging
from typing import Any

from .data import ProbeSample, _field_value, _load_image_value

logger = logging.getLogger("lvr_eval.data.blink")


DEFAULT_BLINK_FIELD_MAP = {
    "id": ["id", "sample_id", "uid", "question_id"],
    "image": ["image", "image_path", "image_1", "img"],
    "question": ["question", "prompt", "query", "instruction"],
    "answer": ["answer", "label", "target", "correct_answer"],
    "bboxes": ["bboxes", "bbox", "region_bboxes"],
    "choices": ["choices", "options", "answer_choices"],
    "rationale": ["rationale", "explanation"],
}


def _field_candidates(cfg: dict, logical_name: str) -> list[str]:
    field_map = dict(DEFAULT_BLINK_FIELD_MAP)
    field_map.update(cfg.get("field_map") or {})
    configured = field_map[logical_name]
    if isinstance(configured, str):
        return [configured]
    return list(configured)


def _value(record: dict, cfg: dict, logical_name: str, default=None, required: bool = False):
    local_cfg = {**cfg, "field_map": {logical_name: _field_candidates(cfg, logical_name)}}
    return _field_value(record, local_cfg, logical_name, default=default, required=required)


def _load_records(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)
    if isinstance(records, dict):
        for key in ("data", "records", "samples", "train", "validation", "test"):
            if isinstance(records.get(key), list):
                records = records[key]
                break
    if not isinstance(records, list):
        raise ValueError(f"BLINK input must be a JSON list or JSONL: {path}")
    return [record for record in records if isinstance(record, dict)]


def _question_with_choices(question: str, choices: Any) -> str:
    if not choices:
        return question
    if isinstance(choices, dict):
        parts = [f"{key}. {value}" for key, value in choices.items()]
    else:
        parts = [str(item) for item in choices]
    if not parts:
        return question
    return f"{question}\n\nChoices:\n" + "\n".join(parts) + "\n\nAnswer with the correct option or answer text."


def load_blink(cfg: dict) -> list[ProbeSample]:
    path = cfg.get("json_path") or cfg.get("jsonl_path") or cfg.get("path")
    if not path:
        raise KeyError("blink loader requires json_path/jsonl_path/path")
    image_root = cfg.get("image_root", "")
    skip_missing = bool(cfg.get("skip_missing_images", True))

    out: list[ProbeSample] = []
    missing_image_count = 0
    weak_oracle_count = 0
    for i, record in enumerate(_load_records(path)):
        try:
            image_value = _value(record, cfg, "image", required=True)
            image = _load_image_value(image_value, image_root)
        except Exception:  # noqa: BLE001
            missing_image_count += 1
            if skip_missing:
                continue
            raise

        choices = _value(record, cfg, "choices")
        question = _question_with_choices(str(_value(record, cfg, "question", default="")), choices)
        sample_id = str(_value(record, cfg, "id", default=f"blink_{i:06d}"))
        bboxes = _value(record, cfg, "bboxes")
        if not bboxes:
            weak_oracle_count += 1

        out.append(ProbeSample(
            id=sample_id,
            image=image,
            question=question,
            answer=_value(record, cfg, "answer"),
            raw=record,
            image_path=str(image_value),
            bboxes=bboxes,
            rationale=_value(record, cfg, "rationale"),
            paired_id=str(record.get("paired_id", sample_id)),
            source_dataset=record.get("dataset", "blink"),
            task_metadata={
                "source_type": "blink",
                "choices": choices,
                "weak_oracle": not bool(bboxes),
                "category": record.get("category") or record.get("task") or record.get("subtask"),
            },
        ))

    logger.info(
        "loaded BLINK: %s -> %d usable samples missing_image_count=%d weak_oracle_count=%d",
        path,
        len(out),
        missing_image_count,
        weak_oracle_count,
    )
    return out
