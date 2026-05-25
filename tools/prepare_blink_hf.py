#!/usr/bin/env python
"""Prepare BLINK into a source_type=blink manifest."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DATASET_ID = "BLINK-Benchmark/BLINK"


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


def records_from_hf(dataset: str, configs: list[str], split: str, max_samples: int | None) -> list[dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    if configs == ["all"]:
        configs = list(get_dataset_config_names(dataset))
    rows = []
    remaining = int(max_samples) if max_samples is not None else None
    for config in configs:
        ds = load_dataset(dataset, config, split=split)
        if remaining is not None:
            if remaining <= 0:
                break
            ds = ds.select(range(min(remaining, len(ds))))
        for row in ds:
            item = dict(row)
            item["_blink_config"] = config
            rows.append(item)
        if remaining is not None:
            remaining -= len(ds)
    return rows


def records_from_local(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    records = load_json_records(path)
    return records[: int(max_samples)] if max_samples is not None else records


def first_value(record: dict[str, Any], keys: list[str]):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def save_image_value(value: Any, out_dir: Path, idx: int, image_root: Path | None) -> str | None:
    if isinstance(value, Image.Image):
        rel = f"blink_{idx:06d}.png"
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        value.convert("RGB").save(path)
        return rel
    if isinstance(value, dict) and isinstance(value.get("image"), Image.Image):
        return save_image_value(value["image"], out_dir, idx, image_root)
    if isinstance(value, list):
        for item in value:
            rel = save_image_value(item, out_dir, idx, image_root)
            if rel:
                return rel
        return None
    if value is None:
        return None
    src = Path(str(value))
    if image_root is not None and not src.is_absolute():
        src = image_root / src
    if not src.is_file():
        return None
    rel = f"blink_{idx:06d}{src.suffix or '.png'}"
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return rel


def make_image_grid(images: list[Image.Image], *, thumb: int = 256) -> Image.Image | None:
    if not images:
        return None
    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for idx, image in enumerate(images):
        img = image.convert("RGB")
        img.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb + (thumb - img.width) // 2
        y = (idx // cols) * thumb + (thumb - img.height) // 2
        canvas.paste(img, (x, y))
    return canvas


def save_blink_images(record: dict[str, Any], out_dir: Path, idx: int, image_root: Path | None) -> str | None:
    image_values = [
        record.get(key)
        for key in ("image_1", "image_2", "image_3", "image_4")
        if record.get(key) is not None
    ]
    pil_images = []
    for value in image_values:
        if isinstance(value, Image.Image):
            pil_images.append(value.convert("RGB"))
        elif isinstance(value, dict) and isinstance(value.get("image"), Image.Image):
            pil_images.append(value["image"].convert("RGB"))
    if len(pil_images) > 1:
        grid = make_image_grid(pil_images)
        if grid is None:
            return None
        rel = f"blink_{idx:06d}.png"
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(path)
        return rel
    if pil_images:
        return save_image_value(pil_images[0], out_dir, idx, image_root)
    image_value = first_value(record, ["image", "image_path", "image_1", "img", "input_image"])
    return save_image_value(image_value, out_dir, idx, image_root)


def normalize_answer(record: dict[str, Any]) -> Any:
    answer = first_value(record, ["answer", "label", "target", "correct_answer", "gt_answer"])
    choices = first_value(record, ["choices", "options", "answer_choices"])
    if isinstance(answer, int) and isinstance(choices, list) and 0 <= answer < len(choices):
        return choices[answer]
    if isinstance(answer, str) and isinstance(choices, dict) and answer in choices:
        return [answer, choices[answer]]
    return answer


def convert(records: list[dict[str, Any]], out_root: Path, *, image_root: Path | None) -> dict:
    images_dir = out_root / "images"
    manifest_path = out_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "dataset": DATASET_ID,
        "schema": "lvr-eval source_type=blink manifest v1",
        "n_source_records": len(records),
        "n_written": 0,
        "n_skipped_no_image": 0,
        "n_skipped_no_answer": 0,
        "n_weak_oracle": 0,
        "fields_seen": sorted({key for record in records for key in record}),
    }
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, record in enumerate(tqdm(records, desc="convert blink")):
            answer = normalize_answer(record)
            if answer is None or str(answer).strip() == "":
                stats["n_skipped_no_answer"] += 1
                continue
            image_rel = save_blink_images(record, images_dir, idx, image_root)
            if not image_rel:
                stats["n_skipped_no_image"] += 1
                continue
            bboxes = first_value(record, ["bboxes", "bbox", "region_bboxes"]) or []
            if not bboxes:
                stats["n_weak_oracle"] += 1
            question = str(first_value(record, ["question", "prompt", "query", "instruction"]) or "")
            out_record = {
                "id": str(first_value(record, ["id", "sample_id", "uid", "question_id"]) or f"blink_{idx:06d}"),
                "paired_id": str(record.get("paired_id") or first_value(record, ["id", "sample_id", "uid", "question_id"]) or f"blink_{idx:06d}"),
                "image": image_rel,
                "question": question,
                "answer": jsonable(answer),
                "choices": jsonable(first_value(record, ["choices", "options", "answer_choices"])),
                "bboxes": jsonable(bboxes),
                "rationale": first_value(record, ["rationale", "explanation"]),
                "category": first_value(record, ["category", "task", "subtask", "sub_task"]) or record.get("_blink_config"),
                "dataset": "blink",
                "weak_oracle": not bool(bboxes),
                "raw": jsonable(record),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            stats["n_written"] += 1
    (out_root / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats["n_written"] == 0:
        fail("no BLINK records were written")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare BLINK for source_type=blink.")
    ap.add_argument("--out", default="data/blink")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--configs", nargs="+", default=["all"], help="BLINK config names, or all.")
    ap.add_argument("--split", default="val")
    ap.add_argument("--input-json", default=None)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args(argv)

    if args.input_json:
        records = records_from_local(Path(args.input_json), args.max_samples)
        image_root = Path(args.image_root) if args.image_root else Path(args.input_json).parent
    else:
        records = records_from_hf(args.dataset, args.configs, args.split, args.max_samples)
        image_root = Path(args.image_root) if args.image_root else None
    stats = convert(records, Path(args.out), image_root=image_root)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {Path(args.out) / 'manifest.jsonl'}")
    print(f"Wrote images:   {Path(args.out) / 'images'}")


if __name__ == "__main__":
    main()
