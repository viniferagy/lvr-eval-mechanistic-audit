"""Bootstrap confidence intervals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    n: int
    mean: float
    ci_low: float
    ci_high: float
    seed: int
    n_resamples: int

    def as_dict(self) -> dict:
        return {
            "n": int(self.n),
            "mean": float(self.mean),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "seed": int(self.seed),
            "n_resamples": int(self.n_resamples),
        }


def _clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr[np.isfinite(arr)]


def paired_bootstrap(
    values_a,
    values_b=None,
    *,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult | None:
    a = _clean(values_a)
    if values_b is not None:
        b = _clean(values_b)
        n = min(len(a), len(b))
        if n == 0:
            return None
        values = a[:n] - b[:n]
    else:
        values = a
        n = len(values)
    if n == 0:
        return None

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        stats.append(float(stat_fn(values[idx])))
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return BootstrapResult(
        n=n,
        mean=float(stat_fn(values)),
        ci_low=float(low),
        ci_high=float(high),
        seed=seed,
        n_resamples=int(n_resamples),
    )
