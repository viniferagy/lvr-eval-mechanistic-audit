"""SPD-Faith paired-counterfactual loader."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from .data import ProbeSample, _field_value, _load_image_value

logger = logging.getLogger("lvr_eval.data.spd_faith")


DEFAULT_SPD_FIELD_MAP = {
    "id": ["id", "sample_id", "uid", "pair_id"],
    "paired_id": ["paired_id", "pair_id", "id"],
    "image": ["image", "image_clean", "clean_image", "image_a"],
    "counterfactual_image": [
        "counterfactual_image",
        "image_counterfactual",
        "cf_image",
        "image_b",
    ],
    "question": ["question", "query", "prompt"],
    "answer": ["answer", "gold_answer_clean", "answer_clean", "label"],
    "counterfactual_answer": [
        "counterfactual_answer",
        "gold_answer_counterfactual",
        "answer_counterfactual",
        "cf_answer",
    ],
    "rationale": ["rationale", "cot", "explanation"],
    "region_mask": ["region_mask", "mask", "diff_mask"],
    "bboxes": ["bboxes", "bbox", "boxes"],
}


def _field_candidates(cfg: dict, logical_name: str) -> list[str]:
    field_map = dict(DEFAULT_SPD_FIELD_MAP)
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
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "records", "samples"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"SPD-Faith input must be a JSON list or JSONL: {path}")
    return data


def load_spd_faith(cfg: dict) -> list[ProbeSample]:
    path = cfg.get("json_path") or cfg.get("jsonl_path") or cfg.get("path")
    if not path:
        raise KeyError("spd_faith loader requires json_path/jsonl_path/path")
    image_root = cfg.get("image_root", "")
    skip_missing = bool(cfg.get("skip_missing_images", True))
    require_region_or_bbox = bool(cfg.get("require_region_or_bbox", False))

    out: list[ProbeSample] = []
    missing_image_count = 0
    missing_region_count = 0
    for i, record in enumerate(_load_records(path)):
        if not isinstance(record, dict):
            continue
        if require_region_or_bbox and not (
            _value(record, cfg, "region_mask") is not None or _value(record, cfg, "bboxes")
        ):
            missing_region_count += 1
            continue
        try:
            image_value = _value(record, cfg, "image", required=True)
            cf_image_value = _value(record, cfg, "counterfactual_image", required=True)
            image = _load_image_value(image_value, image_root)
            cf_image = _load_image_value(cf_image_value, image_root)
        except Exception:  # noqa: BLE001
            missing_image_count += 1
            if skip_missing:
                continue
            raise

        sample_id = _value(record, cfg, "id", default=f"spd_{i}")
        out.append(ProbeSample(
            id=str(sample_id),
            image=image,
            question=str(_value(record, cfg, "question", default="")),
            answer=_value(record, cfg, "answer"),
            raw=record,
            image_path=str(image_value),
            counterfactual_image=cf_image,
            counterfactual_answer=_value(record, cfg, "counterfactual_answer"),
            rationale=_value(record, cfg, "rationale"),
            region_mask=_value(record, cfg, "region_mask"),
            bboxes=_value(record, cfg, "bboxes"),
            paired_id=str(_value(record, cfg, "paired_id", default=sample_id)),
            source_dataset=record.get("dataset", "spd_faith"),
            task_metadata={
                "source_type": "spd_faith",
                "counterfactual_image_path": str(cf_image_value),
            },
        ))

    logger.info(
        "loaded SPD-Faith: %s -> %d usable pairs missing_image_count=%d",
        path,
        len(out),
        missing_image_count,
    )
    if require_region_or_bbox:
        logger.info("SPD-Faith region/bbox filter skipped %d records", missing_region_count)
    return out
