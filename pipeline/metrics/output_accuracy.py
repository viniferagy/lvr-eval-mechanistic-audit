"""Optional output-level accuracy scorer."""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", s.strip().lower())
    return re.sub(r"\s+", " ", s)


def exact_match(preds: list[str], samples: list) -> float:
    if not samples:
        return 0.0
    hit = 0
    for pred, sample in zip(preds, samples):
        gt = getattr(sample, "answer", None)
        if gt is None:
            continue
        if isinstance(gt, (list, tuple)):
            hit += int(any(_norm(str(g)) in _norm(pred) for g in gt))
        else:
            hit += int(_norm(str(gt)) in _norm(pred))
    return hit / len(samples)

