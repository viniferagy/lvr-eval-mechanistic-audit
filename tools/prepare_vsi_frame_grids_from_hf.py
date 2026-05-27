#!/usr/bin/env python
"""Build VSI-Bench frame-grid JPEGs from official Hugging Face video zips.

The HF table for `nyu-visionx/VSI-Bench` contains annotations only. Visual
payloads live in three zip files in the same dataset repository:
`arkitscenes.zip`, `scannet.zip`, and `scannetpp.zip`.

This script downloads one or more of those zips, extracts them, and writes
`<out>/<dataset>/<scene_name>.jpg` grids. That naming is intentionally matched
by `tools/prepare_vsi_bench_hf.py --image-root <out>`.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


DATASET_ID = "nyu-visionx/VSI-Bench"
ZIP_BY_SOURCE = {
    "arkitscenes": "arkitscenes.zip",
    "scannet": "scannet.zip",
    "scannetpp": "scannetpp.zip",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value).strip())


def load_scene_counts(split: str, sources: set[str]) -> dict[str, Counter]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    counts: dict[str, Counter] = {source: Counter() for source in sources}
    for row in ds:
        source = str(row.get("dataset") or "")
        if source not in sources:
            continue
        scene = str(row.get("scene_name") or "").strip()
        if scene:
            counts[source][scene] += 1
    return counts


def download_zip(source: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    if source not in ZIP_BY_SOURCE:
        fail(f"unknown VSI source {source!r}; expected one of {sorted(ZIP_BY_SOURCE)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=ZIP_BY_SOURCE[source],
        cache_dir=str(cache_dir),
    ))


def extract_zip(zip_path: Path, extract_root: Path, source: str) -> Path:
    target = extract_root / source
    stamp = target / ".extract_complete"
    if stamp.is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc=f"extract {zip_path.name}"):
            zf.extract(member, target)
    stamp.write_text(str(zip_path), encoding="utf-8")
    return target


def index_visual_files(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    videos: dict[str, list[Path]] = defaultdict(list)
    images: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        stem_values = {path.stem, path.parent.name}
        if suffix in VIDEO_SUFFIXES:
            for stem in stem_values:
                videos[stem].append(path)
        elif suffix in IMAGE_SUFFIXES:
            for stem in stem_values:
                images[stem].append(path)
    return videos, images


def match_paths(index: dict[str, list[Path]], scene: str) -> list[Path]:
    if scene in index:
        return sorted(index[scene])
    safe = safe_name(scene)
    if safe in index:
        return sorted(index[safe])
    matches: list[Path] = []
    for key, paths in index.items():
        if scene in key or key in scene:
            matches.extend(paths)
    return sorted(set(matches))


def make_grid(images: list[Image.Image], *, thumb: int, grid_cols: int | None = None) -> Image.Image | None:
    if not images:
        return None
    cols = int(grid_cols) if grid_cols and grid_cols > 0 else int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for idx, image in enumerate(images):
        frame = image.convert("RGB")
        frame.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb + (thumb - frame.width) // 2
        y = (idx // cols) * thumb + (thumb - frame.height) // 2
        canvas.paste(frame, (x, y))
    return canvas


def grid_from_images(paths: list[Path], *, n_frames: int, thumb: int, grid_cols: int | None = None) -> Image.Image | None:
    if not paths:
        return None
    if len(paths) > n_frames:
        indices = [int(i * (len(paths) - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
        paths = [paths[i] for i in indices]
    frames: list[Image.Image] = []
    for path in paths:
        try:
            frames.append(Image.open(path).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
    return make_grid(frames, thumb=thumb, grid_cols=grid_cols)


def _cv2_grid(video_path: Path, *, n_frames: int, thumb: int, grid_cols: int | None = None) -> Image.Image | None:
    try:
        import cv2  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None
        indices = [int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
        frames: list[Image.Image] = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        return make_grid(frames, thumb=thumb, grid_cols=grid_cols)
    finally:
        cap.release()


def cv2_decode_sanity(video_path: Path) -> dict:
    """Read one frame so H.264/MP4 codec failures do not pass silently."""
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "backend": "cv2",
            "path": str(video_path),
            "available": False,
            "ok": False,
            "error": repr(exc),
        }
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return {
            "backend": "cv2",
            "path": str(video_path),
            "available": True,
            "ok": bool(ok),
            "frame_shape": list(frame.shape) if ok and frame is not None else None,
            "frame_count": total,
            "fps": fps,
        }
    finally:
        cap.release()


def _imageio_grid(video_path: Path, *, n_frames: int, thumb: int, grid_cols: int | None = None) -> Image.Image | None:
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        meta = iio.immeta(video_path)
        total = int(meta.get("nframes") or 0)
    except Exception:  # noqa: BLE001
        total = 0
    frames: list[Image.Image] = []
    try:
        if total > 0:
            indices = {int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)}
            for idx, frame in enumerate(iio.imiter(video_path)):
                if idx in indices:
                    frames.append(Image.fromarray(frame).convert("RGB"))
                if len(frames) >= n_frames:
                    break
        else:
            stride = 30
            for idx, frame in enumerate(iio.imiter(video_path)):
                if idx % stride == 0:
                    frames.append(Image.fromarray(frame).convert("RGB"))
                if len(frames) >= n_frames:
                    break
    except Exception:  # noqa: BLE001
        return None
    return make_grid(frames, thumb=thumb, grid_cols=grid_cols)


def grid_from_video(path: Path, *, n_frames: int, thumb: int, grid_cols: int | None = None) -> Image.Image | None:
    return (
        _cv2_grid(path, n_frames=n_frames, thumb=thumb, grid_cols=grid_cols)
        or _imageio_grid(path, n_frames=n_frames, thumb=thumb, grid_cols=grid_cols)
    )


def write_scene_grid(
    source: str,
    scene: str,
    *,
    video_paths: list[Path],
    image_paths: list[Path],
    out_root: Path,
    n_frames: int,
    thumb: int,
    grid_cols: int | None,
    quality: int,
) -> bool:
    grid = None
    if image_paths:
        grid = grid_from_images(image_paths, n_frames=n_frames, thumb=thumb, grid_cols=grid_cols)
    if grid is None:
        for video_path in video_paths:
            grid = grid_from_video(video_path, n_frames=n_frames, thumb=thumb, grid_cols=grid_cols)
            if grid is not None:
                break
    if grid is None:
        return False
    out_path = out_root / source / f"{safe_name(scene)}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=quality)
    return True


def has_video_decoder() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import imageio.v3  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def write_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_sources(value: str) -> set[str]:
    if value == "all":
        return set(ZIP_BY_SOURCE)
    sources = {item.strip() for item in value.split(",") if item.strip()}
    unknown = sources - set(ZIP_BY_SOURCE)
    if unknown:
        fail(f"unknown source(s): {sorted(unknown)}")
    return sources


def limit_scenes(counter: Counter, max_scenes: int | None) -> list[str]:
    scenes = [scene for scene, _ in counter.most_common()]
    return scenes[:max_scenes] if max_scenes is not None else scenes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare VSI frame-grid images from official HF video zips.")
    ap.add_argument("--sources", default="scannetpp", help="Comma list or 'all'. Defaults to smallest official zip source.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/vsi_raw/frame_grids")
    ap.add_argument("--cache-dir", default="data/vsi_raw/hf_cache")
    ap.add_argument("--extract-root", default="data/vsi_raw/videos")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--grid-cols", type=int, default=8)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--max-scenes-per-source", type=int, default=None)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args(argv)

    sources = parse_sources(args.sources)
    scene_counts = load_scene_counts(args.split, sources)
    stats = {
        "schema": "lvr-eval vsi_frame_grid_from_hf_video_zips v1",
        "dataset": DATASET_ID,
        "split": args.split,
        "sources": sorted(sources),
        "out": args.out,
        "n_frames": int(args.n_frames),
        "thumb": int(args.thumb),
        "grid_cols": int(args.grid_cols) if args.grid_cols and args.grid_cols > 0 else None,
        "grid_rows": int(math.ceil(args.n_frames / args.grid_cols)) if args.grid_cols and args.grid_cols > 0 else None,
        "grid_protocol": (
            f"{args.n_frames}-frame static image grid; not the native VSI video "
            "protocol and not directly comparable to published video-LLM numbers"
        ),
        "scene_counts": {source: len(counter) for source, counter in scene_counts.items()},
        "qa_counts": {source: int(sum(counter.values())) for source, counter in scene_counts.items()},
        "n_written_scenes": 0,
        "n_missing_visual": 0,
        "n_decode_failed": 0,
        "per_source": {},
    }
    print(json.dumps({k: stats[k] for k in ("dataset", "split", "sources", "scene_counts", "qa_counts")}, indent=2))
    if args.list_only:
        return

    if not has_video_decoder():
        print(
            "[vsi] No Python video decoder found (cv2 or imageio). "
            "Image-sequence zips can still be processed; video files will fail until "
            "`opencv-python-headless` or `imageio[ffmpeg]` is installed.",
            file=sys.stderr,
        )

    out_root = Path(args.out)
    cache_dir = Path(args.cache_dir)
    extract_root = Path(args.extract_root)
    for source in sorted(sources):
        zip_path = download_zip(source, cache_dir)
        if args.download_only:
            stats["per_source"][source] = {"zip_path": str(zip_path), "status": "downloaded"}
            continue
        source_root = extract_zip(zip_path, extract_root, source)
        video_index, image_index = index_visual_files(source_root)
        scenes = limit_scenes(scene_counts[source], args.max_scenes_per_source)
        source_stats = {
            "zip_path": str(zip_path),
            "extract_root": str(source_root),
            "n_scene_targets": len(scenes),
            "n_video_index_keys": len(video_index),
            "n_image_index_keys": len(image_index),
            "decode_sanity": None,
            "n_written": 0,
            "n_missing_visual": 0,
            "n_decode_failed": 0,
            "examples": [],
        }
        first_video = next((paths[0] for _key, paths in sorted(video_index.items()) if paths), None)
        if first_video is not None:
            source_stats["decode_sanity"] = cv2_decode_sanity(first_video)
            if not source_stats["decode_sanity"].get("ok"):
                print(
                    "[vsi] cv2 decode sanity failed for "
                    f"{first_video}; falling back to imageio when available.",
                    file=sys.stderr,
                )
        for scene in tqdm(scenes, desc=f"grid {source}"):
            videos = match_paths(video_index, scene)
            images = match_paths(image_index, scene)
            if not videos and not images:
                source_stats["n_missing_visual"] += 1
                stats["n_missing_visual"] += 1
                continue
            ok = write_scene_grid(
                source,
                scene,
                video_paths=videos,
                image_paths=images,
                out_root=out_root,
                n_frames=args.n_frames,
                thumb=args.thumb,
                grid_cols=args.grid_cols,
                quality=args.quality,
            )
            if ok:
                source_stats["n_written"] += 1
                stats["n_written_scenes"] += 1
                if len(source_stats["examples"]) < 5:
                    source_stats["examples"].append({
                        "scene_name": scene,
                        "qa_count": int(scene_counts[source][scene]),
                        "out": str(out_root / source / f"{safe_name(scene)}.jpg"),
                        "n_video_candidates": len(videos),
                        "n_image_candidates": len(images),
                    })
            else:
                source_stats["n_decode_failed"] += 1
                stats["n_decode_failed"] += 1
        stats["per_source"][source] = source_stats
        write_stats(out_root / "frame_grid_prepare_stats.json", stats)

    write_stats(out_root / "frame_grid_prepare_stats.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote VSI frame-grid stats: {out_root / 'frame_grid_prepare_stats.json'}")
    if not args.download_only and stats["n_written_scenes"] == 0:
        fail("no frame-grid images were written")


if __name__ == "__main__":
    main()
