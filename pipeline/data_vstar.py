"""V*Bench loader for high-resolution detail-localization audit samples."""
from __future__ import annotations

import json
import logging
from typing import Any

from .data import ProbeSample, _field_value, _load_image_value

logger = logging.getLogger("lvr_eval.data.vstar")


DEFAULT_VSTAR_FIELD_MAP = {
    "id": ["id", "sample_id", "uid", "question_id"],
    "image": ["image", "image_path", "img"],
    "question": ["question", "prompt", "query", "instruction"],
    "answer": ["answer", "label", "target", "correct_answer"],
    "bboxes": ["bbox", "bboxes", "answer_bbox", "region_bboxes"],
    "choices": ["choices", "options", "answer_choices"],
    "category": ["category", "split", "subtask", "task"],
    "rationale": ["rationale", "explanation"],
}


def _field_candidates(cfg: dict, logical_name: str) -> list[str]:
    field_map = dict(DEFAULT_VSTAR_FIELD_MAP)
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
        raise ValueError(f"V*Bench input must be a JSON list or JSONL: {path}")
    return [record for record in records if isinstance(record, dict)]


def _question_with_choices(question: str, choices: Any) -> str:
    if not choices:
        return question
    if isinstance(choices, dict):
        parts = [f"{key}. {value}" for key, value in choices.items()]
    else:
        letters = ["A", "B", "C", "D", "E", "F"]
        parts = [
            f"{letters[idx]}. {item}" if idx < len(letters) else str(item)
            for idx, item in enumerate(choices)
        ]
    if not parts:
        return question
    return f"{question}\n\nChoices:\n" + "\n".join(parts) + "\n\nAnswer with the correct option or answer text."


def normalize_bboxes(value: Any) -> list:
    """Normalize V*Bench bbox values to a list of four-coordinate boxes."""
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("bbox", "bboxes", "answer_bbox", "region_bboxes"):
            if key in value:
                return normalize_bboxes(value[key])
        if all(k in value for k in ("x0", "y0", "x1", "y1")):
            return [[value["x0"], value["y0"], value["x1"], value["y1"]]]
        if all(k in value for k in ("x", "y", "w", "h")):
            x, y, w, h = value["x"], value["y"], value["w"], value["h"]
            return [[x, y, x + w, y + h]]
        return []
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if len(vals) == 4 and all(isinstance(v, (int, float)) for v in vals):
            return [vals]
        out = []
        for item in vals:
            out.extend(normalize_bboxes(item))
        return out
    return []


def load_vstar(cfg: dict) -> list[ProbeSample]:
    path = cfg.get("json_path") or cfg.get("jsonl_path") or cfg.get("path")
    if not path:
        raise KeyError("vstar loader requires json_path/jsonl_path/path")
    image_root = cfg.get("image_root", "")
    skip_missing = bool(cfg.get("skip_missing_images", True))
    require_bbox = bool(cfg.get("require_bbox", True))

    out: list[ProbeSample] = []
    missing_image_count = 0
    missing_bbox_count = 0
    high_res_count = 0
    for i, record in enumerate(_load_records(path)):
        try:
            image_value = _value(record, cfg, "image", required=True)
            image = _load_image_value(image_value, image_root)
        except Exception:  # noqa: BLE001
            missing_image_count += 1
            if skip_missing:
                continue
            raise

        bboxes = normalize_bboxes(_value(record, cfg, "bboxes"))
        if not bboxes:
            missing_bbox_count += 1
            if require_bbox:
                raise ValueError(f"V*Bench sample {record.get('id', i)!r} missing bbox")
        choices = _value(record, cfg, "choices")
        question = _question_with_choices(str(_value(record, cfg, "question", default="")), choices)
        sample_id = str(_value(record, cfg, "id", default=f"vstar_{i:06d}"))
        high_resolution = bool(record.get("high_resolution", max(image.size) >= 1500))
        high_res_count += int(high_resolution)
        metadata = dict(record.get("task_metadata") or {})
        metadata.update({
            "source_type": "vstar",
            "choices": choices,
            "category": _value(record, cfg, "category"),
            "high_resolution": high_resolution,
            "bbox_required": require_bbox,
            "bbox_count": len(bboxes),
            "bbox_format": record.get("bbox_format", "xyxy_absolute"),
        })
        if record.get("answer_text") is not None:
            metadata["answer_text"] = record.get("answer_text")

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
            source_dataset=record.get("dataset", "vstar"),
            task_metadata=metadata,
        ))

    logger.info(
        "loaded V*Bench: %s -> %d usable samples missing_image_count=%d missing_bbox_count=%d high_res_count=%d",
        path,
        len(out),
        missing_image_count,
        missing_bbox_count,
        high_res_count,
    )
    return out
