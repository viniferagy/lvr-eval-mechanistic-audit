"""BF-Swap controlled paired latent replacement."""
from __future__ import annotations

import numpy as np

from .bf_patch_answer_transfer import patch_grid, patch_one_pair
from ..base import MetricSpec


METRIC_ID = "bf_swap_latent_replacement"
LEGACY_NAME = "bf_swap"
DEFAULT_LAYERS = [7, 14, 21, 27]
DEFAULT_POSITION_BUCKETS = ["query", "latent"]


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "grid": patch_grid(DEFAULT_LAYERS, DEFAULT_POSITION_BUCKETS),
        "scalars": ["swap_margin_shift", "swap_answer_transfer_rate", "n_paired"],
        "status": "runnable_v0",
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("bf_swap", {}) or {}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _cell_summary(cell: dict, records: list[dict]) -> dict:
    valid = [r for r in records if r.get("error") is None]
    shifts = [
        float(r["logit_margin_shift"])
        for r in valid
        if r.get("logit_margin_shift") is not None
    ]
    transfers = [
        float(bool(r.get("answer_transferred")))
        for r in valid
        if r.get("answer_transferred") is not None
    ]
    return {
        **cell,
        "swap_margin_shift": _mean(shifts),
        "swap_answer_transfer_rate": _mean(transfers),
        "n_paired": len(records),
        "n_success": len(valid),
        "n_error": len(records) - len(valid),
        "records": records,
    }


def _flatten_sample_records(cells: list[dict]) -> list[dict]:
    out = []
    for cell in cells:
        for record in cell.get("records", []):
            if record.get("error") is not None:
                continue
            out.append({
                "id": record.get("id"),
                "paired_id": record.get("paired_id"),
                "layer": cell.get("layer"),
                "position_bucket": cell.get("position_bucket"),
                "reduction": {
                    "swap_margin_shift": record.get("logit_margin_shift"),
                    "swap_answer_transfer": float(bool(record.get("answer_transferred"))),
                },
            })
    return out


def reduce_cells(cells: list[dict]) -> dict | None:
    shifts = [float(c["swap_margin_shift"]) for c in cells if c.get("swap_margin_shift") is not None]
    transfers = [
        float(c["swap_answer_transfer_rate"])
        for c in cells
        if c.get("swap_answer_transfer_rate") is not None
    ]
    if not shifts and not transfers:
        return None
    return {
        "swap_margin_shift": _mean(shifts),
        "swap_answer_transfer_rate": _mean(transfers),
        "n_paired": int(max((c.get("n_paired", 0) for c in cells), default=0)),
        "n_cells": len(cells),
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    local = _cfg(cfg)
    layers = [int(v) for v in local.get("layers", DEFAULT_LAYERS)]
    buckets = [str(v) for v in local.get("position_buckets", DEFAULT_POSITION_BUCKETS)]
    paired = [
        sample for sample in samples
        if sample.counterfactual_image is not None and sample.counterfactual_answer is not None
    ]
    max_pairs = local.get("max_pairs")
    if max_pairs is not None:
        paired = paired[:int(max_pairs)]

    cells = []
    for cell in patch_grid(layers, buckets):
        records = []
        for sample in paired:
            try:
                records.append(patch_one_pair(
                    wrapper,
                    sample,
                    layer=cell["layer"],
                    position_bucket=cell["position_bucket"],
                ))
            except Exception as exc:  # noqa: BLE001
                records.append({
                    "id": sample.id,
                    "paired_id": sample.paired_id,
                    "layer": cell["layer"],
                    "position_bucket": cell["position_bucket"],
                    "source_answer": str(sample.counterfactual_answer),
                    "target_answer": str(sample.answer),
                    "error": repr(exc),
                })
        cells.append(_cell_summary(cell, records))

    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "layers": layers,
            "position_buckets": buckets,
            "max_pairs": max_pairs,
            "source": "counterfactual",
            "target": "clean",
            "primary": "paired_hidden_state_swap",
        },
        "cells": cells,
        "samples": _flatten_sample_records(cells),
        "reduction": reduce_cells(cells),
        "n_paired": len(paired),
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="BF-Swap Latent Replacement",
    kind="internal_curve",
    run_fn=run,
)
