"""MazePlanning loader for step-level visual reasoning samples."""
from __future__ import annotations

import json
import logging
from typing import Any

from .data import ProbeSample, _field_value, _load_image_value

logger = logging.getLogger("lvr_eval.data.maze")


DEFAULT_MAZE_FIELD_MAP = {
    "id": ["id", "sample_id", "uid", "maze_id"],
    "image": ["image", "maze_image", "image_path"],
    "question": ["question", "query", "prompt", "instruction"],
    "answer": ["answer", "path", "solution", "target"],
    "rationale": ["rationale", "trace", "reasoning_trace"],
}


def _field_candidates(cfg: dict, logical_name: str) -> list[str]:
    field_map = dict(DEFAULT_MAZE_FIELD_MAP)
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
        for key in ("data", "records", "samples", "test"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Maze input must be a JSON list or JSONL: {path}")
    return data


def load_maze(cfg: dict) -> list[ProbeSample]:
    path = cfg.get("json_path") or cfg.get("jsonl_path") or cfg.get("path")
    if not path:
        raise KeyError("maze loader requires json_path/jsonl_path/path")
    image_root = cfg.get("image_root", "")
    skip_missing = bool(cfg.get("skip_missing_images", True))

    out: list[ProbeSample] = []
    missing_image_count = 0
    for i, record in enumerate(_load_records(path)):
        if not isinstance(record, dict):
            continue
        try:
            image_value = _value(record, cfg, "image", required=True)
            image = _load_image_value(image_value, image_root)
        except Exception:  # noqa: BLE001
            missing_image_count += 1
            if skip_missing:
                continue
            raise

        sample_id = _value(record, cfg, "id", default=f"maze_{i}")
        task_metadata = {
            "source_type": "maze",
            "difficulty": record.get("difficulty"),
            "grid_size": record.get("grid_size") or record.get("size"),
            "steps": record.get("steps"),
            "start": record.get("start"),
            "goal": record.get("goal"),
        }
        out.append(ProbeSample(
            id=str(sample_id),
            image=image,
            question=str(_value(record, cfg, "question", default="")),
            answer=_value(record, cfg, "answer"),
            raw=record,
            image_path=str(image_value),
            rationale=_value(record, cfg, "rationale"),
            paired_id=str(record.get("paired_id", sample_id)),
            source_dataset=record.get("dataset", "maze"),
            task_metadata={k: v for k, v in task_metadata.items() if v is not None},
        ))

    logger.info(
        "loaded Maze: %s -> %d usable samples missing_image_count=%d",
        path,
        len(out),
        missing_image_count,
    )
    return out
