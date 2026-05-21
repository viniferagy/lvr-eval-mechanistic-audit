"""Sanity checks for BF-3 confidence progression curves."""
from __future__ import annotations

import numpy as np

from .common import FAIL, PASS, WARN, as_float_array, make_check, make_report, spearman_no_scipy


DEFAULTS = {
    "min_layers": 20,
    "min_drop": 0.5,
    "max_drop": 10.0,
    "warn_monotonicity": 0.3,
    "pass_monotonicity": 0.5,
}


def check_bf3_curve(curve, model: str, cfg: dict | None = None,
                    source: str = "baseline") -> dict:
    """Validate a BF-3 aggregate entropy curve."""
    thresholds = dict(DEFAULTS)
    thresholds.update(((cfg or {}).get("validation", {}).get("bf3", {})))

    arr = as_float_array(curve)
    checks = []
    details = {"thresholds": thresholds}

    if arr.size == 0:
        checks.append(make_check("curve_present", FAIL, "missing BF-3 curve"))
        return make_report("bf3_confidence_progression", model, checks, source, details)

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
        return make_report("bf3_confidence_progression", model, checks, source, details)

    early = float(arr[:4].mean())
    late = float(arr[-4:].mean())
    drop = early - late
    details.update({"early_entropy": early, "late_entropy": late,
                    "early_to_late_drop": float(drop)})

    checks.append(make_check(
        "early_gt_late",
        PASS if early > late else FAIL,
        "early entropy should exceed late entropy",
        early=early,
        late=late,
    ))

    min_drop = float(thresholds["min_drop"])
    max_drop = float(thresholds["max_drop"])
    in_range = min_drop <= drop <= max_drop
    checks.append(make_check(
        "drop_range",
        PASS if in_range else WARN,
        "drop outside expected BF-3 range",
        drop=float(drop),
        min_drop=min_drop,
        max_drop=max_drop,
    ))

    layers = np.arange(arr.size)
    rho = spearman_no_scipy(layers, -arr)
    details["monotonicity_spearman"] = rho
    if not np.isfinite(rho):
        mono_status = WARN
    elif rho >= float(thresholds["pass_monotonicity"]):
        mono_status = PASS
    elif rho >= float(thresholds["warn_monotonicity"]):
        mono_status = WARN
    else:
        mono_status = FAIL
    checks.append(make_check(
        "monotonicity",
        mono_status,
        "entropy should usually decrease across layers",
        spearman=float(rho),
    ))

    return make_report("bf3_confidence_progression", model, checks, source, details)
