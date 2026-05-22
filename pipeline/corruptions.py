"""Region corruption helpers for v2 metric fixtures."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BinaryMask:
    data: np.ndarray
    kind: str

    @property
    def coverage(self) -> float:
        return float(np.asarray(self.data, dtype=bool).mean())


def _empty_mask(image: Image.Image) -> np.ndarray:
    w, h = image.size
    return np.zeros((h, w), dtype=bool)


def _bbox_to_pixels(bbox: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    left = max(0, min(width, int(round(min(x0, x1)))))
    right = max(left + 1, min(width, int(round(max(x0, x1)))))
    top = max(0, min(height, int(round(min(y0, y1)))))
    bottom = max(top + 1, min(height, int(round(max(y0, y1)))))
    return left, top, right, bottom


def relevant_mask(image: Image.Image, bboxes=None, region_mask=None) -> BinaryMask:
    if region_mask is not None:
        arr = np.asarray(region_mask, dtype=bool)
        return BinaryMask(arr, "relevant")
    mask = _empty_mask(image)
    if bboxes:
        width, height = image.size
        for bbox in bboxes:
            left, top, right, bottom = _bbox_to_pixels(bbox, width, height)
            mask[top:bottom, left:right] = True
    return BinaryMask(mask, "relevant")


def random_mask(image: Image.Image, *, coverage: float = 0.25, seed: int = 0) -> BinaryMask:
    rng = np.random.default_rng(seed)
    mask = rng.random(_empty_mask(image).shape) < float(coverage)
    return BinaryMask(mask, "random")


def irrelevant_mask(
    image: Image.Image,
    relevant: BinaryMask,
    *,
    coverage: float | None = None,
    seed: int = 0,
) -> BinaryMask:
    rng = random.Random(seed)
    base = np.asarray(relevant.data, dtype=bool)
    mask = np.zeros_like(base)
    target = int(round((coverage if coverage is not None else relevant.coverage) * mask.size))
    candidates = [(r, c) for r, c in zip(*np.where(~base))]
    rng.shuffle(candidates)
    for r, c in candidates[:target]:
        mask[r, c] = True
    return BinaryMask(mask, "irrelevant")


def apply_mask(image: Image.Image, mask: BinaryMask, *, fill=(0, 0, 0), severity: float = 1.0) -> Image.Image:
    img = image.convert("RGB")
    if severity <= 0:
        return img
    arr = np.asarray(img).copy()
    m = np.asarray(mask.data, dtype=bool)
    arr[m] = np.asarray(fill, dtype=arr.dtype)
    return Image.fromarray(arr)
