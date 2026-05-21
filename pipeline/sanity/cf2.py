"""Sanity checks for CF-2 decay sweeps."""
from __future__ import annotations

import numpy as np

from .common import FAIL, PASS, WARN, as_float_array, is_finite_scalar, make_check, make_report, spearman_no_scipy


DEFAULTS = {
    "warn_trend_spearman": 0.2,
}


def check_cf2_result(result: dict, cfg: dict | None = None) -> dict:
    """Validate a CF-2 result object."""
    thresholds = dict(DEFAULTS)
    thresholds.update(((cfg or {}).get("validation", {}).get("cf2", {})))

    model = str(result.get("model", "unknown"))
    readout = result.get("readout", "unknown")
    families = result.get("families", {})
    checks = []
    details = {"thresholds": thresholds, "readout": readout}

    checks.append(make_check(
        "families_present",
        PASS if isinstance(families, dict) and bool(families) else FAIL,
        observed=list(families) if isinstance(families, dict) else None,
    ))

    for fam, rec in families.items():
        sev = as_float_array(rec.get("severities"))
        curve = as_float_array(rec.get("curve"))
        prefix = f"{fam}."

        checks.append(make_check(
            prefix + "length_match",
            PASS if sev.size > 0 and sev.size == curve.size else FAIL,
            severity_count=int(sev.size),
            curve_count=int(curve.size),
        ))
        if sev.size == 0 or curve.size == 0:
            continue

        checks.append(make_check(
            prefix + "clean_baseline",
            PASS if np.isclose(sev[0], 0.0) else FAIL,
            "first severity should be 0.0",
            first_severity=float(sev[0]),
        ))
        strictly_increasing = bool(np.all(np.diff(sev) > 0))
        checks.append(make_check(
            prefix + "severity_increasing",
            PASS if strictly_increasing else FAIL,
            severities=sev.tolist(),
        ))

        finite_curve = bool(np.all(np.isfinite(curve)))
        checks.append(make_check(prefix + "finite_curve", PASS if finite_curve else FAIL))

        feats = rec.get("features", {})
        auc = feats.get("auc")
        checks.append(make_check(prefix + "auc_finite", PASS if is_finite_scalar(auc) else FAIL, auc=auc))

        char = feats.get("char_severity")
        if char is not None:
            in_range = float(sev.min()) <= float(char) <= float(sev.max())
            checks.append(make_check(
                prefix + "char_severity_range",
                PASS if in_range else WARN,
                char_severity=char,
                min_severity=float(sev.min()),
                max_severity=float(sev.max()),
            ))

        if finite_curve and sev.size >= 3:
            rho = spearman_no_scipy(sev, curve)
            details[f"{fam}_trend_spearman"] = rho
            checks.append(make_check(
                prefix + "trend",
                PASS if np.isfinite(rho) and rho >= float(thresholds["warn_trend_spearman"]) else WARN,
                "metric should usually rise as image degradation increases",
                spearman=float(rho),
            ))

    return make_report("cf2_pf_decay_curve", model, checks, "full_sweep", details)
