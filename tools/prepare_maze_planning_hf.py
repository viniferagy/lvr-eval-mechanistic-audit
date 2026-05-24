#!/usr/bin/env python
"""Convert huanyu112/MazePlanning-Test into a source_type=maze manifest."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DATASET_ID = "huanyu112/MazePlanning-Test"
LABEL_FILE = "updated_data.json"
IMAGE_PREFIX = "imgs/"
ACTION_VOCAB = {"go forward", "turn left", "turn right"}


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
    return value


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)
    if isinstance(records, dict):
        for key in ("data", "records", "samples", "train", "test"):
            if isinstance(records.get(key), list):
                records = records[key]
                break
    if not isinstance(records, list):
        fail(f"input records must be a JSON list or JSONL: {path}")
    out = [record for record in records if isinstance(record, dict)]
    if len(out) != len(records):
        fail(f"input records contain non-object rows: {path}")
    return out


def first_image_path(record: dict[str, Any]) -> str | None:
    value = record.get("input_img") or record.get("image") or record.get("image_path")
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value is not None else None


def action_answer(actions: Any) -> str | None:
    if isinstance(actions, str):
        text = actions.strip()
        if text:
            return text
        return None
    if isinstance(actions, (list, tuple)):
        cleaned = [str(item).strip().lower() for item in actions if str(item).strip()]
        return ", ".join(cleaned) if cleaned else None
    return None


def clean_prompt(text: Any) -> str:
    prompt = str(text or "").strip()
    prompt = re.sub(r"^\s*USER:\s*", "", prompt, flags=re.IGNORECASE)
    prompt = prompt.replace("<image>", "").strip()
    if not prompt:
        prompt = (
            "Given the maze in the image, determine a valid action sequence from the "
            "green arrow to the red endpoint. Use only go forward, turn left, and turn right."
        )
    if "Answer with a comma-separated action sequence" not in prompt:
        prompt = (
            f"{prompt}\n\n"
            "Answer with a comma-separated action sequence using only: "
            "go forward, turn left, turn right."
        )
    return prompt


def infer_grid_size(path: str | None) -> str | None:
    if not path:
        return None
    match = re.search(r"size[_-](\d+)[_-](\d+)", path)
    if match:
        return f"{match.group(1)}x{match.group(2)}"
    return None


def copy_image(src_root: Path, rel_path: str, dst_root: Path, idx: int) -> str:
    rel = rel_path.replace("\\", "/")
    source = src_root / rel
    if not source.is_file() and rel.startswith(IMAGE_PREFIX):
        source = src_root / rel[len(IMAGE_PREFIX):]
    if not source.is_file():
        fail(f"missing Maze image: {src_root / rel_path}")
    suffix = source.suffix or ".png"
    out_rel = f"maze_{idx:03d}{suffix.lower()}"
    dest = dst_root / out_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return out_rel


def save_hf_image(row: dict[str, Any], dst_root: Path, idx: int) -> str:
    image = row.get("image")
    if not isinstance(image, Image.Image):
        fail(f"HF image row {idx} is missing a PIL image")
    out_rel = f"maze_{idx:03d}.png"
    path = dst_root / out_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return out_rel


def maze_bbox_from_image(path: Path, *, white_threshold: int = 250) -> list[float] | None:
    image = Image.open(path).convert("RGB")
    arr = image.load()
    width, height = image.size
    xs = []
    ys = []
    for y in range(height):
        for x in range(width):
            r, g, b = arr[x, y]
            if r < white_threshold or g < white_threshold or b < white_threshold:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    pad = max(2, int(round(0.02 * max(width, height))))
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(width, max(xs) + pad + 1)
    y1 = min(height, max(ys) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def records_from_hf(max_samples: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    label_path = Path(hf_hub_download(DATASET_ID, LABEL_FILE, repo_type="dataset"))
    labels = load_json_records(label_path)
    ds = load_dataset(DATASET_ID, split="train")
    if max_samples is not None:
        n = min(int(max_samples), len(labels), len(ds))
        labels = labels[:n]
        ds = ds.select(range(n))
    if len(ds) < len(labels):
        fail(f"HF image table has fewer rows than labels: images={len(ds)} labels={len(labels)}")
    rows = []
    for idx, label in enumerate(labels):
        row = dict(label)
        row["_hf_image"] = ds[int(idx)].get("image")
        rows.append(row)
    return rows


def records_from_local(input_path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    records = load_json_records(input_path)
    return records[: int(max_samples)] if max_samples is not None else records


def validate_actions(actions: list[str], *, strict: bool) -> None:
    bad = [action for action in actions if action not in ACTION_VOCAB]
    if bad and strict:
        fail(f"unexpected Maze actions: {bad[:5]}")


def convert_records(records: list[dict[str, Any]], out_root: Path, *, image_root: Path | None, strict_actions: bool) -> dict:
    images_dir = out_root / "images"
    manifest_path = out_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset": DATASET_ID,
        "schema": "lvr-eval source_type=maze manifest v1",
        "n_source_records": len(records),
        "n_written": 0,
        "n_skipped_no_answer": 0,
        "n_skipped_no_image": 0,
        "action_vocab": sorted(ACTION_VOCAB),
        "fields_seen": sorted({key for record in records for key in record}),
    }
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, record in enumerate(tqdm(records, desc="convert maze")):
            actions = record.get("label_actions") or record.get("answer") or record.get("path") or record.get("solution")
            answer = action_answer(actions)
            if not answer:
                stats["n_skipped_no_answer"] += 1
                continue
            action_list = [part.strip().lower() for part in answer.split(",") if part.strip()]
            validate_actions(action_list, strict=strict_actions)

            rel_image = first_image_path(record)
            if record.get("_hf_image") is not None:
                image_rel = save_hf_image({"image": record["_hf_image"]}, images_dir, idx)
            elif rel_image and image_root is not None:
                image_rel = copy_image(image_root, rel_image, images_dir, idx)
            else:
                stats["n_skipped_no_image"] += 1
                continue
            bbox = maze_bbox_from_image(images_dir / image_rel)

            source_id = Path(rel_image or image_rel).stem
            sample_id = str(record.get("id") or record.get("sample_id") or record.get("maze_id") or source_id or f"maze_{idx:03d}")
            out_record = {
                "id": sample_id,
                "paired_id": sample_id,
                "image": image_rel,
                "question": clean_prompt(record.get("input_text") or record.get("question") or record.get("prompt")),
                "answer": answer,
                "rationale": answer,
                "bboxes": [bbox] if bbox is not None else [],
                "steps": action_list,
                "grid_size": record.get("grid_size") or record.get("size") or infer_grid_size(rel_image),
                "source_image_path": rel_image,
                "dataset": "maze_planning_hf",
                "raw": jsonable({k: v for k, v in record.items() if k != "_hf_image"}),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            stats["n_written"] += 1

    (out_root / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats["n_written"] == 0:
        fail("no Maze records were written")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare huanyu112/MazePlanning-Test for source_type=maze.")
    ap.add_argument("--out", default="data/maze_planning")
    ap.add_argument("--input-json", default=None, help="Optional local updated_data.json/jsonl instead of HF labels.")
    ap.add_argument("--image-root", default=None, help="Required with --input-json unless image paths are absolute.")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--strict-actions", action="store_true")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    if args.input_json:
        records = records_from_local(Path(args.input_json), max_samples=args.max_samples)
        root = Path(args.image_root) if args.image_root else Path(args.input_json).parent
        stats = convert_records(records, out_root, image_root=root, strict_actions=args.strict_actions)
    else:
        records = records_from_hf(max_samples=args.max_samples)
        stats = convert_records(records, out_root, image_root=None, strict_actions=args.strict_actions)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {out_root / 'manifest.jsonl'}")
    print(f"Wrote images:   {out_root / 'images'}")


if __name__ == "__main__":
    main()
