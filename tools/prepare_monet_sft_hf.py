#!/usr/bin/env python
"""Convert NOVAglow646/Monet-SFT-125K into a local JSONL manifest.

This script is intentionally schema-tolerant: the public dataset contains SFT
conversation/image records, and only a small audit subset is needed for adapter
preflight. Rows without an image or a usable prompt are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


DATASET_ID = "NOVAglow646/Monet-SFT-125K"


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Image.Image):
        return f"<PIL.Image size={value.size}>"
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def first_present(record: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def iter_content_items(record: dict[str, Any]):
    data = record.get("data")
    if isinstance(data, list):
        for message in data:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", message.get("from", ""))).lower()
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        yield role, item
            elif isinstance(content, str):
                yield role, {"type": "text", "text": content}


def text_from_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        value = message.get("value", message.get("content", message.get("text", "")))
        if isinstance(value, list):
            return " ".join(text_from_message(item) for item in value)
        return str(value or "")
    return str(message or "")


def text_from_content_item(item: dict[str, Any]) -> str:
    value = item.get("text", item.get("value", item.get("content", "")))
    return str(value or "")


def extract_prompt_answer(record: dict[str, Any]) -> tuple[str | None, str | None]:
    if isinstance(record.get("data"), list):
        user_parts = []
        assistant_parts = []
        for role, item in iter_content_items(record):
            if str(item.get("type", "")).lower() == "image":
                continue
            text = text_from_content_item(item).replace("<image>", "").strip()
            if not text:
                continue
            if role in {"human", "user"}:
                user_parts.append(text)
            elif role in {"gpt", "assistant"}:
                assistant_parts.append(text)
        prompt = "\n".join(user_parts).strip()
        answer = "\n".join(assistant_parts).strip() if assistant_parts else None
        if prompt:
            return prompt, answer

    conv = first_present(record, ["conversations", "messages", "conversation"])
    if isinstance(conv, list):
        user_parts = []
        assistant_parts = []
        for msg in conv:
            role = ""
            if isinstance(msg, dict):
                role = str(msg.get("from", msg.get("role", ""))).lower()
            text = text_from_message(msg).replace("<image>", "").strip()
            if not text:
                continue
            if role in {"human", "user"}:
                user_parts.append(text)
            elif role in {"gpt", "assistant"}:
                assistant_parts.append(text)
        prompt = "\n".join(user_parts).strip()
        answer = assistant_parts[-1].strip() if assistant_parts else None
        if prompt:
            return prompt, answer

    prompt = first_present(record, ["question", "query", "prompt", "problem", "input", "instruction"])
    answer = first_present(record, ["answer", "label", "target", "output", "response"])
    if prompt is not None:
        return str(prompt).replace("<image>", "").strip(), str(answer).strip() if answer is not None else None
    return None, None


def strip_latent_tokens(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = re.sub(r"<abs_vis_token>.*?</abs_vis_token>", "<latent>", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<abs_vis_token>", "<latent>")
    cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
    return cleaned.strip() or None


def image_ref_from_record(record: dict[str, Any]) -> Any:
    for _role, item in iter_content_items(record):
        image = item.get("image")
        if image:
            return image
    value = first_present(record, ["image", "images", "input_img", "img", "picture"])
    return value


def image_from_record(record: dict[str, Any]) -> Image.Image | None:
    value = image_ref_from_record(record)
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Image.Image):
                return item.convert("RGB")
            if isinstance(item, dict) and isinstance(item.get("image"), Image.Image):
                return item["image"].convert("RGB")
    if isinstance(value, dict) and isinstance(value.get("image"), Image.Image):
        return value["image"].convert("RGB")
    return None


def image_from_ref(ref: Any) -> Image.Image | None:
    if isinstance(ref, Image.Image):
        return ref.convert("RGB")
    return None


def image_from_archive_root(ref: Any, archive_root: Path | None) -> Image.Image | None:
    if archive_root is None or not isinstance(ref, str) or not ref:
        return None
    direct = archive_root / ref
    if direct.is_file():
        return Image.open(direct).convert("RGB")
    parts = ref.split("/", 1)
    if len(parts) != 2:
        return None
    dataset_name, inner = parts
    zip_path = archive_root / dataset_name / "images.zip"
    if not zip_path.is_file():
        zip_path = archive_root / f"{dataset_name}_images.zip"
    if not zip_path.is_file():
        return None
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [inner, f"images/{inner}", ref, Path(inner).name]
        for name in candidates:
            try:
                with zf.open(name) as f:
                    return Image.open(f).convert("RGB")
            except KeyError:
                continue
    return None


def download_image_ref(ref: Any) -> Image.Image | None:
    image = image_from_ref(ref)
    if image is not None:
        return image
    if not isinstance(ref, str) or not ref:
        return None
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(DATASET_ID, ref, repo_type="dataset"))
    return Image.open(path).convert("RGB")


def load_hf_records(max_samples: int | None):
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return ds


def convert(out_root: Path, *, max_samples: int | None, image_archive_root: Path | None) -> dict:
    records = load_hf_records(max_samples)
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    stats = {
        "dataset": DATASET_ID,
        "schema": "lvr-eval source_type=jsonl Monet-SFT preflight manifest v1",
        "n_source_records": len(records),
        "n_written": 0,
        "n_skipped_no_image": 0,
        "n_skipped_no_prompt": 0,
        "fields_seen": [],
    }
    fields_seen: set[str] = set()
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(tqdm(records, desc="convert Monet-SFT")):
            record = dict(row)
            fields_seen.update(record)
            ref = image_ref_from_record(record)
            image = image_from_record(record)
            if image is None:
                image = image_from_archive_root(ref, image_archive_root)
            if image is None:
                try:
                    image = download_image_ref(ref)
                except Exception:  # noqa: BLE001
                    image = None
            if image is None:
                stats["n_skipped_no_image"] += 1
                continue
            prompt, answer = extract_prompt_answer(record)
            prompt = strip_latent_tokens(prompt)
            answer = strip_latent_tokens(answer)
            if not prompt:
                stats["n_skipped_no_prompt"] += 1
                continue
            rel = f"monet_sft_{idx:06d}.png"
            image.save(images_dir / rel)
            sample_id = str(first_present(record, ["id", "sample_id", "uid"]) or f"monet_sft_{idx:06d}")
            out_record = {
                "id": sample_id,
                "image": rel,
                "question": prompt,
                "answer": answer,
                "dataset": "monet_sft_125k",
                "source_dataset": DATASET_ID,
                "raw": jsonable({k: v for k, v in record.items() if k not in {"image", "images", "input_img", "img", "picture"}}),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            stats["n_written"] += 1
    stats["fields_seen"] = sorted(fields_seen)
    (out_root / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats["n_written"] == 0:
        fail("no Monet-SFT records were written")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare NOVAglow646/Monet-SFT-125K for adapter preflight.")
    ap.add_argument("--out", default="data/monet_sft")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--image-archive-root",
        default=None,
        help=(
            "Optional root containing extracted Monet-SFT images or per-dataset images.zip files. "
            "The HF dataset stores images in archives such as CogCoM/images.zip."
        ),
    )
    args = ap.parse_args(argv)
    archive_root = Path(args.image_archive_root) if args.image_archive_root else None
    stats = convert(Path(args.out), max_samples=args.max_samples, image_archive_root=archive_root)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
