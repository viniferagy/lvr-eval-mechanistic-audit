#!/usr/bin/env python
"""Prepare V*Bench into a source_type=vstar manifest."""
from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DATASET_ID = "craigwu/vstar_bench"


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, Image.Image):
        return "<PIL.Image>"
    return value


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "records", "samples", "train", "validation", "test"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        fail(f"input records must be a JSON list or JSONL: {path}")
    return [row for row in data if isinstance(row, dict)]


def records_from_hf(dataset: str, split: str, max_samples: int | None, config: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return [dict(row) for row in ds]


def records_from_local(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    records = load_json_records(path)
    return records[: int(max_samples)] if max_samples is not None else records


def first_value(record: dict[str, Any], keys: list[str]):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def normalize_bboxes(value: Any) -> list:
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


def save_image_value(value: Any, out_dir: Path, idx: int, image_root: Path | None) -> tuple[str | None, tuple[int, int] | None]:
    if isinstance(value, Image.Image):
        rel = f"vstar_{idx:06d}.png"
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        img = value.convert("RGB")
        img.save(path)
        return rel, img.size
    if isinstance(value, dict) and isinstance(value.get("image"), Image.Image):
        return save_image_value(value["image"], out_dir, idx, image_root)
    if isinstance(value, dict) and value.get("bytes") is not None:
        try:
            image = Image.open(io.BytesIO(value["bytes"]))
            return save_image_value(image, out_dir, idx, image_root)
        except Exception:  # noqa: BLE001
            return None, None
    if isinstance(value, dict) and value.get("path") is not None:
        return save_image_value(value["path"], out_dir, idx, image_root)
    if value is None:
        return None, None
    src = Path(str(value))
    if image_root is not None and not src.is_absolute():
        src = image_root / src
    if not src.is_file():
        return None, None
    rel = f"vstar_{idx:06d}{src.suffix or '.png'}"
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    try:
        size = Image.open(dest).size
    except Exception:  # noqa: BLE001
        size = None
    return rel, size


def normalize_answer(record: dict[str, Any]) -> Any:
    answer = first_value(record, ["answer", "label", "target", "correct_answer", "gt_answer"])
    choices = first_value(record, ["choices", "options", "answer_choices"])
    if isinstance(answer, int) and isinstance(choices, list) and 0 <= answer < len(choices):
        return choices[answer]
    if isinstance(answer, str) and isinstance(choices, dict) and answer in choices:
        return [answer, choices[answer]]
    return answer


def convert(records: list[dict[str, Any]], out_root: Path, *, image_root: Path | None, require_bbox: bool = True) -> dict:
    images_dir = out_root / "images"
    manifest_path = out_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "dataset": DATASET_ID,
        "schema": "lvr-eval source_type=vstar manifest v1",
        "n_source_records": len(records),
        "n_written": 0,
        "n_skipped_no_image": 0,
        "n_skipped_no_answer": 0,
        "n_missing_bbox": 0,
        "n_high_resolution": 0,
        "fields_seen": sorted({key for record in records for key in record}),
    }
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, record in enumerate(tqdm(records, desc="convert vstar")):
            answer = normalize_answer(record)
            if answer is None or str(answer).strip() == "":
                stats["n_skipped_no_answer"] += 1
                continue
            bbox = normalize_bboxes(first_value(record, ["bbox", "bboxes", "answer_bbox", "region_bboxes"]))
            if not bbox:
                stats["n_missing_bbox"] += 1
                if require_bbox:
                    fail(f"V*Bench sample {idx} missing bbox; V*Bench requires bbox for PF-A selectivity")
            image_value = first_value(record, ["image", "image_path", "img"])
            image_rel, image_size = save_image_value(image_value, images_dir, idx, image_root)
            if not image_rel:
                stats["n_skipped_no_image"] += 1
                continue
            high_resolution = bool(image_size and max(image_size) >= 1500)
            stats["n_high_resolution"] += int(high_resolution)
            out_record = {
                "id": str(first_value(record, ["id", "sample_id", "uid", "question_id"]) or f"vstar_{idx:06d}"),
                "paired_id": str(record.get("paired_id") or first_value(record, ["id", "sample_id", "uid", "question_id"]) or f"vstar_{idx:06d}"),
                "image": image_rel,
                "question": str(first_value(record, ["question", "prompt", "query", "instruction"]) or ""),
                "answer": jsonable(answer),
                "choices": jsonable(first_value(record, ["choices", "options", "answer_choices"])),
                "bboxes": jsonable(bbox),
                "category": first_value(record, ["category", "split", "subtask", "task"]),
                "dataset": "vstar",
                "high_resolution": high_resolution,
                "raw": jsonable(record),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            stats["n_written"] += 1
    (out_root / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats["n_written"] == 0:
        fail("no V*Bench records were written")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare V*Bench for source_type=vstar.")
    ap.add_argument("--out", default="data/vstar")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--config", default=None, help="Optional Hugging Face dataset config/subset name.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--input-json", default=None)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--allow-missing-bbox", action="store_true")
    args = ap.parse_args(argv)

    if args.input_json:
        records = records_from_local(Path(args.input_json), args.max_samples)
        image_root = Path(args.image_root) if args.image_root else Path(args.input_json).parent
    else:
        records = records_from_hf(args.dataset, args.split, args.max_samples, args.config)
        image_root = Path(args.image_root) if args.image_root else None
    stats = convert(records, Path(args.out), image_root=image_root, require_bbox=not args.allow_missing_bbox)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {Path(args.out) / 'manifest.jsonl'}")
    print(f"Wrote images:   {Path(args.out) / 'images'}")


if __name__ == "__main__":
    main()
