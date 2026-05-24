"""BF-Swap controlled paired latent replacement."""
from __future__ import annotations

from dataclasses import replace
import random

import numpy as np

from . import trace_latent as TL
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
        "controls": ["self_swap", "reverse_swap", "random_pair_swap"],
        "status": "runnable_v0",
        "trace_latent_optional": True,
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("bf_swap", {}) or {}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _cell_summary(cell: dict, records: list[dict]) -> dict:
    valid = [r for r in records if r.get("error") is None]
    shifts = [
        float(r["logprob_margin_shift"])
        for r in valid
        if r.get("logprob_margin_shift") is not None
        and np.isfinite(float(r["logprob_margin_shift"]))
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


def _flatten_sample_records(cells: list[dict], *, include_controls: bool = False) -> list[dict]:
    out = []
    for cell in cells:
        if cell.get("control") and not include_controls:
            continue
        for record in cell.get("records", []):
            if record.get("error") is not None:
                continue
            out.append({
                "id": record.get("id"),
                "paired_id": record.get("paired_id"),
                "layer": cell.get("layer"),
                "position_bucket": cell.get("position_bucket"),
                "reduction": {
                    "swap_margin_shift": record.get("logprob_margin_shift"),
                    "swap_answer_transfer": float(bool(record.get("answer_transferred"))),
                },
            })
    return out


def reduce_cells(cells: list[dict]) -> dict | None:
    primary_cells = [cell for cell in cells if not cell.get("control")]
    shifts = [
        float(c["swap_margin_shift"])
        for c in primary_cells
        if c.get("swap_margin_shift") is not None
        and np.isfinite(float(c["swap_margin_shift"]))
    ]
    transfers = [
        float(c["swap_answer_transfer_rate"])
        for c in primary_cells
        if c.get("swap_answer_transfer_rate") is not None
    ]
    if not shifts and not transfers:
        return None
    return {
        "swap_margin_shift": _mean(shifts),
        "swap_answer_transfer_rate": _mean(transfers),
        "n_paired": int(max((c.get("n_paired", 0) for c in primary_cells), default=0)),
        "n_cells": len(primary_cells),
    }


def _self_swap_sample(sample):
    return replace(sample, counterfactual_image=sample.image, counterfactual_answer=sample.answer)


def _reverse_swap_sample(sample):
    return replace(
        sample,
        image=sample.counterfactual_image,
        answer=sample.counterfactual_answer,
        counterfactual_image=sample.image,
        counterfactual_answer=sample.answer,
    )


def _random_pair_sample(sample, source_sample):
    return replace(
        sample,
        counterfactual_image=source_sample.counterfactual_image,
        counterfactual_answer=source_sample.counterfactual_answer,
        task_metadata={
            **(sample.task_metadata or {}),
            "random_pair_source_id": source_sample.id,
            "random_pair_source_paired_id": source_sample.paired_id,
        },
    )


def _control_samples(control: str, paired: list, seed: int) -> list:
    if control == "self_swap":
        return [_self_swap_sample(sample) for sample in paired]
    if control == "reverse_swap":
        return [_reverse_swap_sample(sample) for sample in paired]
    if control == "random_pair_swap":
        if len(paired) < 2:
            return []
        rng = random.Random(seed)
        shuffled = None
        for _ in range(100):
            candidate = list(paired)
            rng.shuffle(candidate)
            if all(a.id != b.id for a, b in zip(paired, candidate)):
                shuffled = candidate
                break
        if shuffled is None:
            candidate = list(paired[1:]) + list(paired[:1])
            if all(a.id != b.id for a, b in zip(paired, candidate)):
                shuffled = candidate
            else:
                raise RuntimeError("failed to construct random_pair_swap derangement")
        return [_random_pair_sample(sample, source) for sample, source in zip(paired, shuffled)]
    raise ValueError(f"unknown BF-Swap control: {control}")


def _run_cell_records(wrapper, paired: list, cell: dict) -> list[dict]:
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
    return records


def _run_trace_records(wrapper, paired: list, cfg: dict, *, patch_steps: list[str]) -> list[dict]:
    records = []
    for sample in paired:
        try:
            rec = TL.patch_pair(wrapper, sample, cfg, patch_steps=patch_steps)
            rec["layer"] = -1
            rec["position_bucket"] = "generation_trace"
            records.append(rec)
        except Exception as exc:  # noqa: BLE001
            records.append({
                "id": sample.id,
                "paired_id": sample.paired_id,
                "layer": -1,
                "position_bucket": "generation_trace",
                "source_answer": str(sample.counterfactual_answer),
                "target_answer": str(sample.answer),
                "error": repr(exc),
            })
    return records


def _run_trace_latent(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    local = _cfg(cfg)
    trace_cfg = TL.cfg(cfg)
    controls = [str(v) for v in local.get(
        "controls",
        ["self_swap", "reverse_swap", "random_pair_swap"],
    )]
    control_seed = int(local.get("control_seed", local.get("seed", 260523)))
    paired = [
        sample for sample in samples
        if sample.counterfactual_image is not None and sample.counterfactual_answer is not None
    ]
    max_pairs = local.get("max_pairs", trace_cfg.get("max_pairs"))
    if max_pairs is not None:
        paired = paired[:int(max_pairs)]
    patch_steps = [str(v) for v in trace_cfg.get("patch_steps", ["last"])]
    cell = {"layer": -1, "position_bucket": "generation_trace", "trace_latent": True, "patch_steps": patch_steps}
    records = _run_trace_records(wrapper, paired, cfg, patch_steps=patch_steps)
    cells = [_cell_summary(cell, records)]

    control_cells = []
    for control in controls:
        control_paired = _control_samples(control, paired, control_seed)
        control_records = _run_trace_records(wrapper, control_paired, cfg, patch_steps=patch_steps)
        control_cells.append({
            **_cell_summary({**cell, "control": control}, control_records),
            "control": control,
        })

    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "layers": [-1],
            "position_buckets": ["generation_trace"],
            "max_pairs": max_pairs,
            "controls": controls,
            "control_seed": control_seed,
            "source": "counterfactual_generation_trace",
            "target": "clean_generation_trace",
            "primary": "paired_hidden_feedback_swap",
            "trace_latent": True,
            "patch_steps": patch_steps,
        },
        "cells": cells,
        "control_cells": control_cells,
        "samples": _flatten_sample_records(cells),
        "reduction": reduce_cells(cells),
        "n_paired": len(paired),
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    if TL.enabled(cfg, METRIC_ID):
        return _run_trace_latent(wrapper, samples, cfg, model_tag)

    local = _cfg(cfg)
    layers = [int(v) for v in local.get("layers", DEFAULT_LAYERS)]
    buckets = [str(v) for v in local.get("position_buckets", DEFAULT_POSITION_BUCKETS)]
    controls = [str(v) for v in local.get(
        "controls",
        ["self_swap", "reverse_swap", "random_pair_swap"],
    )]
    control_seed = int(local.get("control_seed", local.get("seed", 260523)))
    paired = [
        sample for sample in samples
        if sample.counterfactual_image is not None and sample.counterfactual_answer is not None
    ]
    max_pairs = local.get("max_pairs")
    if max_pairs is not None:
        paired = paired[:int(max_pairs)]

    cells = []
    for cell in patch_grid(layers, buckets):
        records = _run_cell_records(wrapper, paired, cell)
        cells.append(_cell_summary(cell, records))

    control_cells = []
    for control in controls:
        control_paired = _control_samples(control, paired, control_seed)
        for cell in patch_grid(layers, buckets):
            records = _run_cell_records(wrapper, control_paired, cell)
            control_cells.append({
                **_cell_summary({**cell, "control": control}, records),
                "control": control,
            })

    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "layers": layers,
            "position_buckets": buckets,
            "max_pairs": max_pairs,
            "controls": controls,
            "control_seed": control_seed,
            "source": "counterfactual",
            "target": "clean",
            "primary": "paired_hidden_state_swap",
        },
        "cells": cells,
        "control_cells": control_cells,
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
