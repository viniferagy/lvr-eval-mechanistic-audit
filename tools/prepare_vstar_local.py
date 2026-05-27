#!/usr/bin/env python
"""Prepare V*Bench from the full Hugging Face repo snapshot.

Do not use ``datasets.load_dataset("craigwu/vstar_bench")`` for the main
V*Bench preparation path: the parquet/Data Studio projection drops the
per-image JSON files that contain bbox annotations. This tool downloads the
dataset repository files directly and reads ``sa_XXXXX.json`` next to each
image.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DATASET_ID = "craigwu/vstar_bench"
SUBDIRS = {
    "direct_attributes": "attribute_recognition",
    "relative_position": "spatial_relationship_reasoning",
}

logger = logging.getLogger("prepare_vstar")


def fail(msg: str) -> None:
    raise SystemExit(f"[prepare_vstar] FAIL: {msg}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare V*Bench full-repo snapshot into a vstar manifest.")
    ap.add_argument("--out", default="data/vstar", help="Output directory for manifest.jsonl and images/")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--cache-dir", default="./hf_cache_vstar", help="Hugging Face snapshot cache directory")
    ap.add_argument("--input-dir", default=None, help="Use an already-downloaded full V*Bench repo directory")
    ap.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them")
    ap.add_argument("--min-image-side", type=int, default=1024)
    ap.add_argument("--expected-min-samples", type=int, default=180)
    ap.add_argument("--allow-empty-bbox", action="store_true")
    return ap.parse_args(argv)


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
    return value


def normalize_bbox_xywh_to_xyxy(bbox_raw: Any, img_w: int, img_h: int) -> list[list[float]]:
    """Convert V*Bench ``[x, y, w, h]`` boxes to absolute ``[x1, y1, x2, y2]``."""
    if not isinstance(bbox_raw, list) or not bbox_raw:
        return []

    boxes = bbox_raw
    if len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
        boxes = [boxes]

    out: list[list[float]] = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x, y, w, h = [float(v) for v in box]
        x1 = max(0.0, min(x, float(img_w)))
        y1 = max(0.0, min(y, float(img_h)))
        x2 = max(0.0, min(x + w, float(img_w)))
        y2 = max(0.0, min(y + h, float(img_h)))
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


def _image_for_json(jpath: Path) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = jpath.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _write_image(src: Path, out_img_path: Path, *, copy_images: bool) -> None:
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    if out_img_path.exists() or out_img_path.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, out_img_path)
    else:
        os.symlink(src.resolve(), out_img_path)


def convert_repo(
    repo_root: Path,
    out_root: Path,
    *,
    copy_images: bool = False,
    min_image_side: int = 1024,
    allow_empty_bbox: bool = False,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    img_out = out_root / "images"
    img_out.mkdir(exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"

    stats = {
        "dataset": DATASET_ID,
        "schema": "lvr-eval source_type=vstar manifest v2 snapshot_json_bbox",
        "source_repo_root": str(repo_root),
        "public_data_loader": "huggingface_hub.snapshot_download",
        "note": "datasets.load_dataset/parquet projection omits per-image bbox JSON; this manifest reads sa_XXXXX.json directly.",
        "n_seen": 0,
        "n_written": 0,
        "n_skipped_no_json": 0,
        "n_skipped_no_image": 0,
        "n_skipped_no_answer": 0,
        "n_missing_bbox": 0,
        "n_empty_bbox_after_normalize": 0,
        "n_below_min_side": 0,
        "n_high_resolution": 0,
        "per_category": {cat: 0 for cat in SUBDIRS.values()},
        "bbox_area_ratio": {"min": None, "median": None, "max": None},
    }
    bbox_ratios: list[float] = []

    with manifest_path.open("w", encoding="utf-8") as fout:
        for subdir, category in SUBDIRS.items():
            sub_path = repo_root / subdir
            if not sub_path.is_dir():
                fail(f"missing subdir {sub_path}; expected full HF repo snapshot")
            for jpath in tqdm(sorted(sub_path.glob("*.json")), desc=f"convert {subdir}"):
                stats["n_seen"] += 1
                try:
                    record = json.loads(jpath.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("bad json %s: %s", jpath, exc)
                    stats["n_skipped_no_json"] += 1
                    continue

                img_path = _image_for_json(jpath)
                if img_path is None:
                    stats["n_skipped_no_image"] += 1
                    continue
                try:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                except Exception as exc:  # noqa: BLE001
                    logger.warning("bad image %s: %s", img_path, exc)
                    stats["n_skipped_no_image"] += 1
                    continue

                bbox_raw = record.get("bbox") or record.get("bboxes") or record.get("answer_bbox") or []
                if not bbox_raw:
                    stats["n_missing_bbox"] += 1
                    if not allow_empty_bbox:
                        fail(f"sample {jpath.stem} has no bbox; V*Bench requires bbox for PF-A")
                bbox_xyxy = normalize_bbox_xywh_to_xyxy(bbox_raw, img_w, img_h)
                if not bbox_xyxy:
                    stats["n_empty_bbox_after_normalize"] += 1
                    if not allow_empty_bbox:
                        continue

                options = record.get("options") or record.get("answer_choices") or []
                if not options:
                    stats["n_skipped_no_answer"] += 1
                    continue
                answer_index = 0
                answer_letter = "A"
                answer_text = options[0]

                below_min = min(img_w, img_h) < int(min_image_side)
                stats["n_below_min_side"] += int(below_min)
                stats["n_high_resolution"] += int(not below_min)

                out_img_subdir = img_out / subdir
                out_img_path = out_img_subdir / img_path.name
                _write_image(img_path, out_img_path, copy_images=copy_images)

                area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in bbox_xyxy)
                ratio = float(area / max(float(img_w * img_h), 1.0))
                bbox_ratios.append(ratio)

                sample_id = f"vstar_{category}_{jpath.stem}"
                out_record = {
                    "id": sample_id,
                    "paired_id": sample_id,
                    "image": str(Path(subdir) / img_path.name),
                    "question": str(record.get("question", "")).strip(),
                    "options": jsonable(options),
                    "answer": answer_letter,
                    "answer_text": jsonable(answer_text),
                    "answer_index": answer_index,
                    "bboxes": bbox_xyxy,
                    "bbox_format": "xyxy_absolute",
                    "target_object": jsonable(record.get("target_object", [])),
                    "category": category,
                    "subdir": subdir,
                    "dataset": "vstar",
                    "image_width": img_w,
                    "image_height": img_h,
                    "high_resolution": not below_min,
                    "task_metadata": {
                        "source_type": "vstar",
                        "category": category,
                        "high_resolution": not below_min,
                        "bbox_count": len(bbox_xyxy),
                        "bbox_area_total": area,
                        "bbox_area_ratio": ratio,
                    },
                    "raw": jsonable(record),
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                stats["n_written"] += 1
                stats["per_category"][category] += 1

    if bbox_ratios:
        ordered = sorted(bbox_ratios)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        stats["bbox_area_ratio"] = {
            "min": ordered[0],
            "median": median,
            "max": ordered[-1],
        }
    (out_root / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats["n_written"] == 0:
        fail("no V*Bench records were written")
    return stats


def snapshot_download_repo(dataset: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        allow_patterns=[
            "direct_attributes/*",
            "relative_position/*",
            "test_questions.jsonl",
            "README*",
        ],
    ))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.input_dir:
        repo_root = Path(args.input_dir)
    else:
        logger.info("snapshot_download(%s) -> %s", args.dataset, args.cache_dir)
        repo_root = snapshot_download_repo(args.dataset, Path(args.cache_dir))
    logger.info("V*Bench repo root: %s", repo_root)
    stats = convert_repo(
        repo_root,
        Path(args.out),
        copy_images=bool(args.copy_images),
        min_image_side=int(args.min_image_side),
        allow_empty_bbox=bool(args.allow_empty_bbox),
    )
    expected = int(args.expected_min_samples)
    if expected > 0 and int(stats["n_written"]) < expected:
        fail(f"only {stats['n_written']} samples written; expected at least {expected}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {Path(args.out) / 'manifest.jsonl'}")
    print(f"Wrote images:   {Path(args.out) / 'images'}")


if __name__ == "__main__":
    main()
