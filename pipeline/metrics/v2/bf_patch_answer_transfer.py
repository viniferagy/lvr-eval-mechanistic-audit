"""BF-Patch answer-transfer skeleton."""
from __future__ import annotations

from ..base import MetricSpec


METRIC_ID = "bf_patch_answer_transfer"
LEGACY_NAME = "bf_patch"
DEFAULT_LAYERS = [0, 7, 14, 21, 27]
DEFAULT_POSITION_BUCKETS = ["image", "query", "latent"]


def patch_grid(layers=None, position_buckets=None) -> list[dict]:
    layers = list(DEFAULT_LAYERS if layers is None else layers)
    buckets = list(DEFAULT_POSITION_BUCKETS if position_buckets is None else position_buckets)
    return [{"layer": int(layer), "position_bucket": bucket} for layer in layers for bucket in buckets]


def answer_transfer_rate(records: list[dict]) -> float | None:
    valid = [r for r in records if r.get("patched_answer") is not None and r.get("source_answer") is not None]
    if not valid:
        return None
    return sum(1 for r in valid if r["patched_answer"] == r["source_answer"]) / len(valid)


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "grid": patch_grid(),
        "scalars": ["logit_margin_shift", "answer_transfer_rate", "n_paired"],
        "status": "skeleton_fixture_only",
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    paired = [
        sample for sample in samples
        if sample.counterfactual_image is not None and sample.counterfactual_answer is not None
    ]
    cells = []
    for cell in patch_grid():
        cells.append({**cell, "logit_margin_shift": None, "answer_transfer_rate": None, "n_paired": len(paired)})
    return {"model": model_tag, "schema": build_schema(), "cells": cells, "n_paired": len(paired)}


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="BF-Patch Answer Transfer",
    kind="v2_skeleton",
    run_fn=run,
)
