#!/usr/bin/env python
"""Convert Jackson-Lv/SPD-Faith-Bench into an spd_faith JSONL manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from collections import Counter

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


DATASET_ID = "Jackson-Lv/SPD-Faith-Bench"
SPLITS = ["easy", "medium", "hard", "multi_diff"]


def ensure_pil_image(img: Any) -> Image.Image:
    if isinstance(img, dict) and "image" in img:
        img = img["image"]
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(img)}")
    return img


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=95)


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


def diff_text(differences: Any) -> str:
    differences = jsonable(differences)
    if not differences:
        return "A visual difference exists between the original and modified image."

    parts = []
    for d in differences if isinstance(differences, list) else [differences]:
        if not isinstance(d, dict):
            continue
        desc = d.get("description")
        typ = d.get("type")
        cat = d.get("category")
        if desc:
            parts.append(str(desc))
        elif typ or cat:
            parts.append(f"type={typ}, category={cat}")
    if parts:
        return " ".join(parts)
    return json.dumps(differences, ensure_ascii=False)


def make_question(differences: Any) -> str:
    desc = diff_text(differences)
    return (
        "Given the edit description below, decide whether the shown image is the "
        "original image or the modified image.\n"
        f"Edit description: {desc}\n"
        "Answer exactly one word: original or modified."
    )


def bbox_list_from_differences(differences: Any, width: int, height: int) -> list[list[float]]:
    boxes = []
    for d in jsonable(differences) or []:
        if not isinstance(d, dict):
            continue
        x = d.get("bbox_x")
        y = d.get("bbox_y")
        w = d.get("bbox_w")
        h = d.get("bbox_h")
        try:
            x, y, w, h = float(x), float(y), float(w), float(h)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        x0 = max(0.0, min(1.0, x / width))
        y0 = max(0.0, min(1.0, y / height))
        x1 = max(0.0, min(1.0, (x + w) / width))
        y1 = max(0.0, min(1.0, (y + h) / height))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/spd_faith_hf")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--max-per-split", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out)
    image_root = out_root / "images"
    manifest_path = out_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset": DATASET_ID,
        "splits": {},
        "n_total": 0,
        "n_written": 0,
        "schema": "lvr-eval source_type=spd_faith paired manifest v1",
        "answer_policy": "clean=original, counterfactual=modified",
        "n_counterfactual_resized": 0,
    }

    with manifest_path.open("w", encoding="utf-8") as fout:
        for split in args.splits:
            ds = load_dataset(DATASET_ID, split=split)
            if args.max_per_split is not None:
                ds = ds.select(range(min(args.max_per_split, len(ds))))

            n_written = 0
            source_ids = Counter(str(row.get("image_id")) for row in ds)
            n_duplicate_source_ids = sum(1 for count in source_ids.values() if count > 1)
            for row_idx, row in enumerate(tqdm(ds, desc=f"convert {split}")):
                image_id = str(row.get("image_id"))
                image1 = row.get("image1")
                image2 = row.get("image2")
                if image1 is None or image2 is None:
                    continue
                image1 = ensure_pil_image(image1)
                image2 = ensure_pil_image(image2)

                if image2.size != image1.size:
                    image2 = image2.resize(image1.size, Image.Resampling.BICUBIC)
                    stats["n_counterfactual_resized"] += 1

                sample_key = f"{row_idx:04d}_{image_id}"
                manifest_id = f"{split}_{sample_key}"
                clean_rel = f"{split}/{sample_key}_clean.jpg"
                cf_rel = f"{split}/{sample_key}_cf.jpg"
                save_image(image1, image_root / clean_rel)
                save_image(image2, image_root / cf_rel)

                differences = jsonable(row.get("differences") or [])
                width, height = image1.size
                record = {
                    "id": manifest_id,
                    "paired_id": manifest_id,
                    "split": split,
                    "source_image_id": image_id,
                    "image_clean": clean_rel,
                    "image_counterfactual": cf_rel,
                    "question": make_question(differences),
                    "gold_answer_clean": "original",
                    "gold_answer_counterfactual": "modified",
                    "answer": "original",
                    "counterfactual_answer": "modified",
                    "rationale": diff_text(differences),
                    "num_differences": jsonable(row.get("num_differences")),
                    "differences": differences,
                    "bboxes": bbox_list_from_differences(differences, width, height),
                    "dataset": "spd_faith_hf",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            stats["splits"][split] = {
                "written": n_written,
                "source_image_ids": len(source_ids),
                "duplicate_source_image_ids": n_duplicate_source_ids,
            }
            stats["n_total"] += len(ds)
            stats["n_written"] += n_written

    stats_path = out_root / "prepare_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote images:   {image_root}")


if __name__ == "__main__":
    main()
