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
import re
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
    raw: Optional[dict] = None
    image_path: Optional[str] = None
    bboxes: Optional[list] = None
    lvr_assistant: Optional[str] = None
    source_dataset: Optional[str] = None
    counterfactual_image: Optional[Image.Image] = None
    counterfactual_answer: Any = None
    rationale: Optional[str] = None
    region_mask: Any = None
    paired_id: Optional[str] = None
    task_metadata: Optional[dict] = None


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
    elif src == "lvr_json":
        samples = _load_lvr_json(cfg_data)
    elif src == "spd_faith":
        from .data_spd_faith import load_spd_faith

        samples = load_spd_faith(cfg_data)
    elif src == "maze":
        from .data_maze import load_maze

        samples = load_maze(cfg_data)
    elif src == "blink":
        from .data_blink import load_blink

        samples = load_blink(cfg_data)
    elif src == "vsi":
        from .data_vsi import load_vsi

        samples = load_vsi(cfg_data)
    elif src == "vstar":
        from .data_vstar import load_vstar

        samples = load_vstar(cfg_data)
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
                raw=r,
                image_path=str(img_value),
                bboxes=r.get("bboxes"),
                lvr_assistant=r.get("lvr_assistant"),
                source_dataset=r.get("dataset"),
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
            raw=dict(r),
            image_path=None,
            bboxes=r.get("bboxes") if isinstance(r, dict) else None,
            lvr_assistant=r.get("lvr_assistant") if isinstance(r, dict) else None,
            source_dataset=r.get("dataset") if isinstance(r, dict) else None,
        ))
    return out


def _strip_image_placeholder(text: str) -> str:
    return (
        text.replace("<image>", "")
        .replace("<|vision_start|>", "")
        .replace("<|vision_end|>", "")
        .replace("<|image_pad|>", "")
        .strip()
    )


def _extract_answer_from_lvr_assistant(text: str):
    if text is None:
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cleaned = text.replace("<lvr>", "").strip()
    return cleaned or None


def _conversation_by_role(conversations: list[dict], role: str) -> dict | None:
    aliases = {
        "human": {"human", "user"},
        "gpt": {"gpt", "assistant"},
        "user": {"human", "user"},
        "assistant": {"gpt", "assistant"},
    }[role]
    for msg in conversations:
        if str(msg.get("from", msg.get("role", ""))).lower() in aliases:
            return msg
    return None


def _first_image_path(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("empty image list")
        return value[0]
    return value


def _load_lvr_json(cfg: dict) -> list[ProbeSample]:
    path = cfg["json_path"]
    image_root = cfg.get("image_root", "")
    scan_limit = cfg.get("max_scan_records", cfg.get("max_records_before_sampling"))
    require_lvr_placeholder = bool(cfg.get("require_lvr_placeholder", True))

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"LVR JSON must be a list, got {type(records)}: {path}")

    out: list[ProbeSample] = []
    scanned = 0
    for i, r in enumerate(records):
        if scan_limit and i >= int(scan_limit):
            break
        scanned += 1
        if not isinstance(r, dict):
            logger.debug("skip LVR record %s: expected dict, got %s", i, type(r))
            continue

        conversations = r.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            logger.debug("skip LVR record %s: missing conversations", i)
            continue

        human = _conversation_by_role(conversations, "human")
        assistant = _conversation_by_role(conversations, "gpt")
        if human is None or assistant is None:
            logger.debug("skip LVR record %s: missing human/gpt message", i)
            continue

        human_value = str(human.get("value", human.get("content", "")))
        assistant_value = str(assistant.get("value", assistant.get("content", "")))
        if require_lvr_placeholder and "<lvr>" not in assistant_value:
            logger.debug("skip LVR record %s: assistant missing <lvr>", i)
            continue

        question = _strip_image_placeholder(human_value)
        answer = _extract_answer_from_lvr_assistant(assistant_value)

        img_value = r.get("image")
        if img_value is None:
            logger.debug("skip LVR record %s: missing image", i)
            continue
        img_path = str(_first_image_path(img_value))

        try:
            image = _load_image_value(img_path, image_root)
        except Exception as exc:  # noqa: BLE001
            if cfg.get("skip_missing_images", True):
                logger.debug("skip LVR record %s: image open failed: %s", i, exc)
                continue
            raise

        sample_id = (
            r.get("id")
            or r.get("question_id")
            or r.get("uid")
            or f"lvr_{i}"
        )

        out.append(ProbeSample(
            id=str(sample_id),
            image=image,
            question=question,
            answer=answer,
            raw=r,
            image_path=img_path,
            bboxes=r.get("bboxes"),
            lvr_assistant=assistant_value,
            source_dataset=r.get("dataset"),
        ))

    logger.info(
        "loaded LVR JSON: %s -> %d usable samples scanned=%d total_records=%d",
        path,
        len(out),
        scanned,
        len(records),
    )
    return out


def batched(seq: list, n: int):
    """简单分批迭代器。"""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
