"""Sanity checks for BF-1 latent ablation sweeps."""
from __future__ import annotations

import numpy as np

from .common import FAIL, PASS, WARN, as_float_array, make_check, make_report


DEFAULTS = {
    "min_abs_delta": 1e-8,
}


def _collect_deltas(result: dict, key: str) -> np.ndarray:
    vals = []
    for layer in result.get("layers", []):
        value = layer.get("delta", {}).get(key)
        if value is not None:
            vals.append(value)
    return as_float_array(vals)


def check_bf1_result(result: dict, cfg: dict | None = None) -> dict:
    """Validate a BF-1 result object."""
    thresholds = dict(DEFAULTS)
    thresholds.update(((cfg or {}).get("validation", {}).get("bf1", {})))

    model = str(result.get("model", "unknown"))
    checks = []
    details = {"thresholds": thresholds}

    baseline = result.get("baseline")
    checks.append(make_check(
        "baseline_present",
        PASS if isinstance(baseline, dict) and bool(baseline) else FAIL,
    ))

    layers = result.get("layers", [])
    n_layers = result.get("n_layers")
    expected_layers = (cfg or {}).get("bf1", {}).get("layers") if cfg else None
    expected_count = len(expected_layers) if expected_layers else n_layers
    checks.append(make_check(
        "layer_results_present",
        PASS if layers else FAIL,
        observed=len(layers),
        expected=expected_count,
    ))

    if expected_count is not None:
        checks.append(make_check(
            "layer_count",
            PASS if len(layers) == expected_count else WARN,
            observed=len(layers),
            expected=expected_count,
        ))

    for key in ("bf3", "pf3"):
        vals = _collect_deltas(result, key)
        if vals.size == 0:
            checks.append(make_check(f"delta_{key}_present", WARN, f"no {key} deltas"))
            continue
        finite = bool(np.all(np.isfinite(vals)))
        checks.append(make_check(f"delta_{key}_finite", PASS if finite else FAIL))
        if finite:
            max_abs = float(np.max(np.abs(vals)))
            details[f"max_abs_delta_{key}"] = max_abs
            checks.append(make_check(
                f"delta_{key}_nonzero",
                PASS if max_abs > float(thresholds["min_abs_delta"]) else WARN,
                "all deltas are near zero; hook/readout may not be active",
                max_abs_delta=max_abs,
                min_abs_delta=float(thresholds["min_abs_delta"]),
            ))

    if baseline:
        for key in ("bf3_curve", "pf3_curve"):
            base_curve = baseline.get(key)
            if base_curve is None:
                continue
            identical_count = 0
            comparable_count = 0
            for layer in layers:
                curve = layer.get(key)
                if curve is None:
                    continue
                a = as_float_array(base_curve)
                b = as_float_array(curve)
                if a.size == b.size and a.size > 0:
                    comparable_count += 1
                    identical_count += int(bool(np.allclose(a, b, rtol=0.0, atol=0.0)))
            if comparable_count:
                checks.append(make_check(
                    f"{key}_changes_under_ablation",
                    PASS if identical_count < comparable_count else WARN,
                    "ablated curves exactly match baseline for all comparable layers",
                    identical_count=identical_count,
                    comparable_count=comparable_count,
                ))

    return make_report("bf1_latent_ablation", model, checks, "full_sweep", details)
