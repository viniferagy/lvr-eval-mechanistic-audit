#!/usr/bin/env python
"""Prepare VSI-Bench into a source_type=vsi manifest for accuracy sanity."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DATASET_ID = "nyu-visionx/VSI-Bench"


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


def records_from_hf(dataset: str, split: str, max_samples: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
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


def image_from_path(value: Any, image_root: Path | None) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and isinstance(value.get("image"), Image.Image):
        return value["image"].convert("RGB")
    if value is None:
        return None
    path = Path(str(value))
    if image_root is not None and not path.is_absolute():
        path = image_root / path
    if not path.is_file():
        return None
    return Image.open(path).convert("RGB")


def make_frame_grid(images: list[Image.Image], *, thumb: int = 256) -> Image.Image | None:
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


def save_image_or_grid(record: dict[str, Any], out_dir: Path, idx: int, image_root: Path | None) -> str | None:
    image_value = first_value(record, ["frame_grid", "video_frame_grid", "image", "image_path"])
    images_value = first_value(record, ["frames", "frame_paths", "images"])
    if image_value is None and images_value is None and image_root is not None:
        scene = str(record.get("scene_name") or record.get("id") or "").strip()
        candidates = []
        if scene:
            candidates.extend([
                image_root / f"{scene}.png",
                image_root / f"{scene}.jpg",
                image_root / scene / "frame_grid.png",
                image_root / scene / "grid.png",
            ])
            candidates.extend(sorted(image_root.glob(f"**/{scene}*.png"))[:4])
            candidates.extend(sorted(image_root.glob(f"**/{scene}*.jpg"))[:4])
        for candidate in candidates:
            if candidate.is_file():
                image_value = str(candidate)
                break
    image = image_from_path(image_value, image_root)
    if image is None and isinstance(images_value, list):
        frames = [img for item in images_value if (img := image_from_path(item, image_root)) is not None]
        image = make_frame_grid(frames)
    if image is None and image_value is not None:
        path = Path(str(image_value))
        if image_root is not None and not path.is_absolute():
            path = image_root / path
        if path.is_file():
            rel = f"vsi_{idx:06d}{path.suffix or '.png'}"
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            return rel
    if image is None:
        return None
    rel = f"vsi_{idx:06d}.png"
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return rel


def normalize_answer(record: dict[str, Any]) -> Any:
    answer = first_value(record, ["answer", "label", "target", "correct_answer", "gt_answer", "ground_truth"])
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
        "schema": "lvr-eval source_type=vsi manifest v1",
        "n_source_records": len(records),
        "n_written": 0,
        "n_skipped_no_image": 0,
        "n_skipped_no_answer": 0,
        "fields_seen": sorted({key for record in records for key in record}),
    }
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, record in enumerate(tqdm(records, desc="convert vsi")):
            answer = normalize_answer(record)
            if answer is None or str(answer).strip() == "":
                stats["n_skipped_no_answer"] += 1
                continue
            image_rel = save_image_or_grid(record, images_dir, idx, image_root)
            if not image_rel:
                stats["n_skipped_no_image"] += 1
                continue
            sample_id = str(first_value(record, ["id", "sample_id", "uid", "question_id"]) or f"vsi_{idx:06d}")
            out_record = {
                "id": sample_id,
                "paired_id": sample_id,
                "image": image_rel,
                "question": str(first_value(record, ["question", "prompt", "query", "instruction"]) or ""),
                "answer": jsonable(answer),
                "choices": jsonable(first_value(record, ["choices", "options", "answer_choices"])),
                "rationale": first_value(record, ["rationale", "explanation"]),
                "category": first_value(record, ["category", "task", "subtask"]),
                "dataset": "vsi_bench",
                "raw": jsonable(record),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            stats["n_written"] += 1
    (out_root / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats["n_written"] == 0:
        fail("no VSI records were written")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare VSI-Bench for source_type=vsi.")
    ap.add_argument("--out", default="data/vsi_bench")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--split", default="test")
    ap.add_argument("--input-json", default=None)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args(argv)

    if args.input_json:
        records = records_from_local(Path(args.input_json), args.max_samples)
        image_root = Path(args.image_root) if args.image_root else Path(args.input_json).parent
    else:
        records = records_from_hf(args.dataset, args.split, args.max_samples)
        image_root = Path(args.image_root) if args.image_root else None
    stats = convert(records, Path(args.out), image_root=image_root)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {Path(args.out) / 'manifest.jsonl'}")
    print(f"Wrote images:   {Path(args.out) / 'images'}")


if __name__ == "__main__":
    main()
