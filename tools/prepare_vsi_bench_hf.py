#!/usr/bin/env python
"""Prepare VSI-Bench into a source_type=vsi manifest for accuracy sanity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DATASET_ID = "nyu-visionx/VSI-Bench"
ANSWER_KEYS = [
    "ground_truth",
    "answer",
    "label",
    "target",
    "correct_answer",
    "gt_answer",
]


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
    if isinstance(value, dict):
        for key in ("image", "pil", "frame_grid"):
            if isinstance(value.get(key), Image.Image):
                return value[key].convert("RGB")
        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            return Image.open(BytesIO(raw_bytes)).convert("RGB")
        for key in ("path", "filename", "file", "image_path"):
            if value.get(key):
                return image_from_path(value[key], image_root)
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_file() and image_root is not None and not path.is_absolute():
        path = image_root / path
    if not path.is_file():
        return None
    return Image.open(path).convert("RGB")


def path_from_value(value: Any, image_root: Path | None) -> Path | None:
    if isinstance(value, dict):
        for key in ("path", "filename", "file", "image_path"):
            if value.get(key):
                return path_from_value(value[key], image_root)
        return None
    if not isinstance(value, (str, Path)):
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    if image_root is not None and not path.is_absolute():
        rooted = image_root / path
        if rooted.is_file():
            return rooted
    return None


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


def _identity_values(record: dict[str, Any], idx: int) -> list[str]:
    values: list[str] = []
    for key in ("id", "idx", "sample_id", "uid", "question_id", "scene_name"):
        value = record.get(key)
        if value is not None:
            values.append(str(value).strip())
    if not values:
        values.append(f"vsi_{idx:06d}")
    return [value for value in values if value]


def _image_root_candidates(record: dict[str, Any], idx: int, image_root: Path | None) -> list[Path]:
    if image_root is None:
        return []
    names = _identity_values(record, idx)
    datasets = [
        str(record.get(key)).strip()
        for key in ("dataset", "source_dataset", "data_source")
        if record.get(key) is not None and str(record.get(key)).strip()
    ]
    candidates: list[Path] = []
    suffixes = [".png", ".jpg", ".jpeg", ".webp"]
    if datasets:
        for name in names:
            for dataset in datasets:
                for suffix in suffixes:
                    candidates.append(image_root / dataset / f"{name}{suffix}")
                for grid_name in ("frame_grid.png", "frame_grid.jpg", "grid.png", "grid.jpg"):
                    candidates.append(image_root / dataset / name / grid_name)
        seen_dataset: set[Path] = set()
        out_dataset: list[Path] = []
        for path in candidates:
            if path not in seen_dataset:
                seen_dataset.add(path)
                out_dataset.append(path)
        return out_dataset
    for name in names:
        for suffix in suffixes:
            candidates.append(image_root / f"{name}{suffix}")
        for grid_name in ("frame_grid.png", "frame_grid.jpg", "grid.png", "grid.jpg"):
            candidates.append(image_root / name / grid_name)
        candidates.extend(sorted(image_root.glob(f"**/{name}*.png"))[:4])
        candidates.extend(sorted(image_root.glob(f"**/{name}*.jpg"))[:4])
        candidates.extend(sorted(image_root.glob(f"**/{name}*.jpeg"))[:4])
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _safe_path_part(value: Any, fallback: str = "unknown") -> str:
    text = str(value).strip() if value is not None else fallback
    if not text:
        text = fallback
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def save_image_or_grid(record: dict[str, Any], out_dir: Path, idx: int, image_root: Path | None) -> str | None:
    return _save_image_or_grid(record, out_dir, idx, image_root, image_cache=None)


def _save_image_or_grid(
    record: dict[str, Any],
    out_dir: Path,
    idx: int,
    image_root: Path | None,
    *,
    image_cache: dict[str, str] | None,
) -> str | None:
    image_value = first_value(record, ["frame_grid", "video_frame_grid", "image", "image_path"])
    images_value = first_value(record, ["frames", "frame_paths", "images"])
    if image_value is None and images_value is None and image_root is not None:
        for candidate in _image_root_candidates(record, idx, image_root):
            if candidate.is_file():
                image_value = str(candidate)
                break
    source_path = path_from_value(image_value, image_root)
    if source_path is not None:
        cache_key = str(source_path.resolve())
        if image_cache is not None and cache_key in image_cache:
            return image_cache[cache_key]
        dataset = _safe_path_part(record.get("dataset") or record.get("source_dataset"), "external")
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:10]
        suffix = source_path.suffix if source_path.suffix else ".jpg"
        rel = f"external/{dataset}/{source_path.stem}_{digest}{suffix}"
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            shutil.copy2(source_path, dest)
        if image_cache is not None:
            image_cache[cache_key] = rel
        return rel
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
    answer = first_value(record, ANSWER_KEYS)
    choices = first_value(record, ["choices", "options", "answer_choices"])
    if isinstance(answer, int) and isinstance(choices, list) and 0 <= answer < len(choices):
        return choices[answer]
    if isinstance(answer, str) and isinstance(choices, dict) and answer in choices:
        return [answer, choices[answer]]
    if isinstance(answer, (int, float)) and math.isfinite(float(answer)):
        return {
            "answer": str(answer),
            "numeric_value": float(answer),
            "abs_tolerance": record.get("abs_tolerance", record.get("tolerance")),
            "rel_tolerance": record.get("rel_tolerance"),
            "unit": record.get("unit") or record.get("answer_unit"),
        }
    if isinstance(answer, str):
        numeric_type = str(record.get("answer_type") or record.get("question_type") or "").lower()
        if any(key in numeric_type for key in ("distance", "count", "number", "numeric", "size")):
            numbers = []
            for match in __import__("re").findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", answer):
                try:
                    numbers.append(float(match))
                except ValueError:
                    continue
            if numbers:
                return {
                    "answer": answer,
                    "numeric_value": numbers[0],
                    "abs_tolerance": record.get("abs_tolerance", record.get("tolerance")),
                    "rel_tolerance": record.get("rel_tolerance"),
                    "unit": record.get("unit") or record.get("answer_unit"),
                }
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
    per_dataset: Counter[str] = Counter()
    per_question_type: Counter[str] = Counter()
    image_cache: dict[str, str] = {}
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, record in enumerate(tqdm(records, desc="convert vsi")):
            answer = normalize_answer(record)
            if answer is None or str(answer).strip() == "":
                stats["n_skipped_no_answer"] += 1
                continue
            image_rel = _save_image_or_grid(record, images_dir, idx, image_root, image_cache=image_cache)
            if not image_rel:
                stats["n_skipped_no_image"] += 1
                continue
            sample_id = str(first_value(record, ["id", "sample_id", "uid", "question_id"]) or f"vsi_{idx:06d}")
            category = first_value(record, ["category", "task", "subtask", "question_type"])
            out_record = {
                "id": sample_id,
                "paired_id": sample_id,
                "image": image_rel,
                "question": str(first_value(record, ["question", "prompt", "query", "instruction"]) or ""),
                "answer": jsonable(answer),
                "choices": jsonable(first_value(record, ["choices", "options", "answer_choices"])),
                "rationale": first_value(record, ["rationale", "explanation"]),
                "category": category,
                "dataset": "vsi_bench",
                "raw": jsonable(record),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            stats["n_written"] += 1
            per_dataset[str(record.get("dataset") or "unknown")] += 1
            per_question_type[str(record.get("question_type") or category or "unknown")] += 1
    stats["n_unique_external_images"] = len(image_cache)
    stats["n_per_dataset"] = dict(sorted(per_dataset.items()))
    stats["n_per_question_type"] = dict(sorted(per_question_type.items()))
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
