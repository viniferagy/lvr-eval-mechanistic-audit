"""Shared helpers for metric sanity reports."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"


def as_float_array(values: Any) -> np.ndarray:
    """Convert nested numeric values to a flat float array."""
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1)


def is_finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_finite_sequence(values: Any) -> bool:
    arr = as_float_array(values)
    return arr.size > 0 and bool(np.all(np.isfinite(arr)))


def jsonable(value: Any) -> Any:
    """Make numpy/scalar values JSON-serializable."""
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def make_check(name: str, status: str, message: str = "", **values: Any) -> dict:
    rec = {"name": name, "status": status}
    if message:
        rec["message"] = message
    if values:
        rec["values"] = jsonable(values)
    return rec


def make_report(metric_id: str, model: str, checks: Iterable[dict],
                source: str | None = None, details: dict | None = None) -> dict:
    check_list = list(checks)
    failed = any(c["status"] == FAIL for c in check_list)
    warned = any(c["status"] == WARN for c in check_list)
    report = {
        "metric_id": metric_id,
        "model": model,
        "source": source,
        "passed": not failed,
        "status": FAIL if failed else (WARN if warned else PASS),
        "checks": check_list,
    }
    if details:
        report["details"] = jsonable(details)
    return report


def spearman_no_scipy(x: Any, y: Any) -> float:
    """Small Spearman fallback for smoke/sanity paths."""
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    if x_arr.size != y_arr.size or x_arr.size < 2:
        return float("nan")

    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks

    rx = rank(x_arr)
    ry = rank(y_arr)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
