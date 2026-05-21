"""Sanity checks for PF-3 attention-distance curves."""
from __future__ import annotations

import numpy as np

from .common import FAIL, PASS, WARN, as_float_array, make_check, make_report


DEFAULTS = {
    "min_layers": 20,
    "near_zero_kl": 1e-6,
    "max_sample_cv": 0.5,
}


def check_pf3_curve(curve, model: str, cfg: dict | None = None,
                    source: str = "baseline",
                    mode_results: dict[str, np.ndarray] | None = None,
                    skip_count: int | None = None,
                    total_count: int | None = None) -> dict:
    """Validate a PF-3 aggregate KL curve."""
    thresholds = dict(DEFAULTS)
    thresholds.update(((cfg or {}).get("validation", {}).get("pf3", {})))

    arr = as_float_array(curve)
    checks = []
    details = {"thresholds": thresholds}

    if arr.size == 0:
        checks.append(make_check("curve_present", FAIL, "missing PF-3 curve"))
        return make_report("pf3_attention_distance", model, checks, source, details)

    checks.append(make_check("curve_present", PASS, length=int(arr.size)))

    min_layers = int(thresholds["min_layers"])
    checks.append(make_check(
        "curve_length",
        PASS if arr.size >= min_layers else FAIL,
        length=int(arr.size),
        min_layers=min_layers,
    ))

    finite = bool(np.all(np.isfinite(arr)))
    checks.append(make_check("finite", PASS if finite else FAIL))
    if not finite:
        return make_report("pf3_attention_distance", model, checks, source, details)

    min_kl = float(arr.min())
    mean_kl = float(arr.mean())
    details.update({"min_kl": min_kl, "mean_kl": mean_kl, "peak_kl": float(arr.max())})

    checks.append(make_check(
        "non_negative_kl",
        PASS if min_kl >= -1e-8 else FAIL,
        "KL should be non-negative",
        min_kl=min_kl,
    ))

    checks.append(make_check(
        "nonzero_signal",
        PASS if mean_kl > float(thresholds["near_zero_kl"]) else WARN,
        "PF-3 mean KL is near zero; possible modality bypass or weak corruption",
        mean_kl=mean_kl,
        near_zero=float(thresholds["near_zero_kl"]),
    ))

    L = arr.size
    early = float(arr[:max(1, L // 4)].mean())
    mid = float(arr[L // 4: 3 * L // 4].mean())
    late = float(arr[3 * L // 4:].mean())
    details.update({"early_kl": early, "mid_kl": mid, "late_kl": late})
    checks.append(make_check(
        "mid_layer_signal",
        PASS if mid >= max(early, late) * 0.5 else WARN,
        "mid-layer PF-3 signal is unexpectedly weak",
        early=early,
        mid=mid,
        late=late,
    ))

    if skip_count is not None and total_count:
        skip_ratio = skip_count / max(total_count, 1)
        checks.append(make_check(
            "skip_ratio",
            PASS if skip_ratio < 0.5 else WARN,
            "many samples/seeds were skipped, often due to dynamic token-length mismatch",
            skip_count=skip_count,
            total_count=total_count,
            skip_ratio=skip_ratio,
        ))

    if mode_results and "mask_50pct" in mode_results and "mask_80pct" in mode_results:
        m50 = float(np.asarray(mode_results["mask_50pct"], dtype=float).mean())
        m80 = float(np.asarray(mode_results["mask_80pct"], dtype=float).mean())
        checks.append(make_check(
            "corruption_ordering",
            PASS if m80 > m50 else WARN,
            "mask_80pct should usually exceed mask_50pct",
            mask_50pct=m50,
            mask_80pct=m80,
        ))

    return make_report("pf3_attention_distance", model, checks, source, details)
