"""
pipeline/data.py
================
Probe 数据集加载。复用 BF-3 的 eval split 以保持指标可比。

统一产出 ProbeSample 列表:
    id, image(PIL.Image, RGB), question(str), answer(str/任意 ground-truth)

退化(CF-2)只作用在 image 上，因此这里保证 image 始终是干净的 PIL.Image。
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger("lvr_eval.data")


DEFAULT_FIELD_MAP = {
    "image": ["image"],
    "question": ["question", "query", "prompt"],
    "answer": ["answer", "label", "target"],
    "id": ["id", "sample_id", "uid"],
}


@dataclass
class ProbeSample:
    id: str
    image: Image.Image
    question: str
    answer: Any


def _open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _nested_get(record: dict, path: str):
    current = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _field_candidates(cfg: dict, logical_name: str) -> list[str]:
    configured = (cfg.get("field_map") or {}).get(logical_name)
    if configured is None:
        configured = DEFAULT_FIELD_MAP[logical_name]
    if isinstance(configured, str):
        return [configured]
    return list(configured)


def _field_value(record: dict, cfg: dict, logical_name: str,
                 default=None, required: bool = False):
    for candidate in _field_candidates(cfg, logical_name):
        value = _nested_get(record, candidate)
        if value is not None:
            return value
    if required:
        tried = ", ".join(_field_candidates(cfg, logical_name))
        raise KeyError(f"missing required field '{logical_name}' (tried: {tried})")
    return default


def _load_image_value(value, image_root: str = "") -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    path = str(value)
    if image_root and not os.path.isabs(path):
        path = os.path.join(image_root, path)
    return _open_rgb(path)


def load_probe_set(cfg_data: dict) -> list[ProbeSample]:
    src = cfg_data["source_type"]
    if src == "jsonl":
        samples = _load_jsonl(cfg_data)
    elif src == "hf":
        samples = _load_hf(cfg_data)
    else:
        raise ValueError(f"未知 source_type: {src}")

    # subsample
    seed = cfg_data.get("shuffle_seed", 0)
    max_n = cfg_data.get("max_samples")
    if max_n and len(samples) > max_n:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_n)
    logger.info("probe set ready: %d samples", len(samples))
    return samples


def _load_jsonl(cfg: dict) -> list[ProbeSample]:
    path = cfg["jsonl_path"]
    image_root = cfg.get("image_root", "")
    out: list[ProbeSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img_value = _field_value(r, cfg, "image", required=True)
            out.append(ProbeSample(
                id=str(_field_value(r, cfg, "id", default=i)),
                image=_load_image_value(img_value, image_root),
                question=str(_field_value(r, cfg, "question", default="")),
                answer=_field_value(r, cfg, "answer"),
            ))
    return out


def _load_hf(cfg: dict) -> list[ProbeSample]:
    try:
        from datasets import load_dataset
    except AttributeError as exc:
        if "PyExtensionType" in str(exc):
            raise RuntimeError(
                "datasets 与 pyarrow 版本不兼容。请运行: "
                "pip install -U 'datasets>=2.19' 'pyarrow<21'"
            ) from exc
        raise
    try:
        ds = load_dataset(cfg["hf_name"], split=cfg["hf_split"])
    except ValueError as exc:
        if "Unknown split" in str(exc):
            raise ValueError(
                f"HF 数据集 {cfg['hf_name']} 没有 split={cfg['hf_split']!r}。"
                f"原始错误: {exc}"
            ) from exc
        raise
    out: list[ProbeSample] = []
    for i, r in enumerate(ds):
        img = _field_value(r, cfg, "image", required=True)
        out.append(ProbeSample(
            id=str(_field_value(r, cfg, "id", default=i)),
            image=_load_image_value(img),
            question=str(_field_value(r, cfg, "question", default="")),
            answer=_field_value(r, cfg, "answer"),
        ))
    return out


def batched(seq: list, n: int):
    """简单分批迭代器。"""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
