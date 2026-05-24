"""PF-A relevant-vs-irrelevant corruption selectivity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...corruptions import apply_mask, irrelevant_mask, random_mask, relevant_mask
from ... import internal_metrics as IM
from . import trace_latent as TL
from ..base import MetricSpec


METRIC_ID = "pf_a_corruption_selectivity"
LEGACY_NAME = "pf_a"


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "required_sample_fields": ["image", "question", "bboxes or region_mask"],
        "scalars": ["selectivity", "relevant_kl", "irrelevant_kl", "random_kl"],
        "status": "runnable_v0",
        "trace_latent_optional": True,
    }


def reduce_samples(records: list[dict]) -> dict | None:
    valid = [
        rec for rec in records
        if rec.get("selectivity") is not None
        and rec.get("relevant_kl") is not None
        and rec.get("irrelevant_kl") is not None
    ]
    if not valid:
        return None
    out = {}
    for key in ("selectivity", "relevant_kl", "irrelevant_kl", "random_kl"):
        vals = [float(rec[key]) for rec in valid if rec.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    out["n"] = len(valid)
    return out


def _masked_attention_kl(wrapper, sample, mask, cfg: dict) -> dict:
    pf_a_cfg = cfg.get("pf_a", {})
    masked_image = apply_mask(
        sample.image,
        mask,
        fill=tuple(pf_a_cfg.get("fill", [0, 0, 0])),
        severity=float(pf_a_cfg.get("severity", 1.0)),
    )
    return IM.query_image_attention_kl_with_meta_from_sample(
        wrapper,
        sample,
        masked_image,
    )


def _mean_kl(meta: dict) -> float | None:
    curve = meta.get("curve")
    if curve is None:
        return None
    return float(IM.pf3_reduce(np.asarray(curve, dtype=float))["mean_kl"])


def _trace_latent_delta(wrapper, sample, image, cfg: dict) -> dict:
    clean_trace = TL.generate_trace(wrapper, sample, cfg, label="clean")
    masked_trace = TL.generate_trace(wrapper, sample, cfg, image=image, label="masked")
    distance = str(TL.cfg(cfg).get("distance", "l2"))
    delta = TL.latent_delta(clean_trace, masked_trace, steps=TL.cfg(cfg).get("readout_steps", ["all"]), distance=distance)
    dist = TL.state_distance(clean_trace, masked_trace, steps=TL.cfg(cfg).get("readout_steps", ["all"]))
    return {
        "curve": [delta] if delta is not None else None,
        "mean_delta": delta,
        "distance": distance,
        "trace_quality": masked_trace.get("trace_quality"),
        "clean_trace_quality": clean_trace.get("trace_quality"),
        "n_lvr_mode_steps": int(masked_trace.get("n_lvr_mode_steps") or 0),
        "n_hidden_feedback_steps": int(masked_trace.get("n_hidden_feedback_steps") or 0),
        "n_captured_latent_states": int(masked_trace.get("n_captured_latent_states") or 0),
        "captured_state_shapes": TL.state_shape_metadata(masked_trace),
        "state_distance": dist,
    }


def _masked_trace_delta(wrapper, sample, mask, cfg: dict) -> dict:
    pf_a_cfg = cfg.get("pf_a", {})
    masked_image = apply_mask(
        sample.image,
        mask,
        fill=tuple(pf_a_cfg.get("fill", [0, 0, 0])),
        severity=float(pf_a_cfg.get("severity", 1.0)),
    )
    return _trace_latent_delta(wrapper, sample, masked_image, cfg)


def _run_trace_latent(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    records = []
    pf_a_cfg = cfg.get("pf_a", {})
    seed = int(pf_a_cfg.get("seed", 0))
    for sample in samples:
        rel = relevant_mask(sample.image, bboxes=sample.bboxes, region_mask=sample.region_mask)
        irr = irrelevant_mask(sample.image, rel, seed=seed)
        rnd = random_mask(sample.image, coverage=max(rel.coverage, 0.01), seed=seed)
        base = {
            "id": sample.id,
            "relevant_coverage": rel.coverage,
            "irrelevant_coverage": irr.coverage,
            "random_coverage": rnd.coverage,
            "relevant_oracle_source": rel.oracle_source,
            "irrelevant_oracle_source": irr.oracle_source,
            "random_oracle_source": rnd.oracle_source,
            "relevant_irrelevant_overlap": int((rel.data & irr.data).sum()),
            "severity0_preserves_image": list(apply_mask(sample.image, rel, severity=0).size) == list(sample.image.size),
        }
        try:
            relevant_meta = _masked_trace_delta(wrapper, sample, rel, cfg)
            irrelevant_meta = _masked_trace_delta(wrapper, sample, irr, cfg)
            random_meta = _masked_trace_delta(wrapper, sample, rnd, cfg)
            relevant_kl = relevant_meta.get("mean_delta")
            irrelevant_kl = irrelevant_meta.get("mean_delta")
            random_kl = random_meta.get("mean_delta")
            selectivity = (
                irrelevant_kl - relevant_kl
                if relevant_kl is not None and irrelevant_kl is not None
                else None
            )
            records.append({
                **base,
                "selectivity": selectivity,
                "relevant_kl": relevant_kl,
                "irrelevant_kl": irrelevant_kl,
                "random_kl": random_kl,
                "trace_quality": relevant_meta.get("trace_quality"),
                "clean_trace_quality": relevant_meta.get("clean_trace_quality"),
                "n_lvr_mode_steps": relevant_meta.get("n_lvr_mode_steps"),
                "n_hidden_feedback_steps": relevant_meta.get("n_hidden_feedback_steps"),
                "n_captured_latent_states": relevant_meta.get("n_captured_latent_states"),
                "captured_state_shapes": relevant_meta.get("captured_state_shapes"),
                "state_distance": {
                    "relevant": relevant_meta.get("state_distance"),
                    "irrelevant": irrelevant_meta.get("state_distance"),
                    "random": random_meta.get("state_distance"),
                },
                "query_target_kind": "generation_trace_latent_state",
                "query_span": None,
                "image_span": None,
                "skip_reasons": {},
                "reduction": {
                    "selectivity": selectivity,
                    "relevant_kl": relevant_kl,
                    "irrelevant_kl": irrelevant_kl,
                    "random_kl": random_kl,
                },
            })
        except Exception as exc:  # noqa: BLE001
            records.append({**base, "error": repr(exc)})
    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "seed": seed,
            "severity": pf_a_cfg.get("severity", 1.0),
            "fill": pf_a_cfg.get("fill", [0, 0, 0]),
            "comparison": "clean_vs_region_masked_generation_trace_latent_delta",
            "trace_latent": True,
            "distance": str(TL.cfg(cfg).get("distance", "l2")),
        },
        "reduction": reduce_samples(records),
        "samples": records,
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    if TL.enabled(cfg, METRIC_ID):
        return _run_trace_latent(wrapper, samples, cfg, model_tag)

    records = []
    pf_a_cfg = cfg.get("pf_a", {})
    seed = int(pf_a_cfg.get("seed", 0))
    for sample in samples:
        rel = relevant_mask(sample.image, bboxes=sample.bboxes, region_mask=sample.region_mask)
        irr = irrelevant_mask(sample.image, rel, seed=seed)
        rnd = random_mask(sample.image, coverage=max(rel.coverage, 0.01), seed=seed)
        base = {
            "id": sample.id,
            "relevant_coverage": rel.coverage,
            "irrelevant_coverage": irr.coverage,
            "random_coverage": rnd.coverage,
            "relevant_oracle_source": rel.oracle_source,
            "irrelevant_oracle_source": irr.oracle_source,
            "random_oracle_source": rnd.oracle_source,
            "relevant_irrelevant_overlap": int((rel.data & irr.data).sum()),
            "severity0_preserves_image": list(apply_mask(sample.image, rel, severity=0).size) == list(sample.image.size),
        }
        try:
            relevant_meta = _masked_attention_kl(wrapper, sample, rel, cfg)
            irrelevant_meta = _masked_attention_kl(wrapper, sample, irr, cfg)
            random_meta = _masked_attention_kl(wrapper, sample, rnd, cfg)
            relevant_kl = _mean_kl(relevant_meta)
            irrelevant_kl = _mean_kl(irrelevant_meta)
            random_kl = _mean_kl(random_meta)
            selectivity = (
                irrelevant_kl - relevant_kl
                if relevant_kl is not None and irrelevant_kl is not None
                else None
            )
            records.append({
                **base,
                "selectivity": selectivity,
                "relevant_kl": relevant_kl,
                "irrelevant_kl": irrelevant_kl,
                "random_kl": random_kl,
                "query_target_kind": relevant_meta.get("query_target_kind"),
                "query_span": relevant_meta.get("query_span"),
                "image_span": relevant_meta.get("image_span"),
                "skip_reasons": {
                    "relevant": relevant_meta.get("skip_reasons", {}),
                    "irrelevant": irrelevant_meta.get("skip_reasons", {}),
                    "random": random_meta.get("skip_reasons", {}),
                },
                "reduction": {
                    "selectivity": selectivity,
                    "relevant_kl": relevant_kl,
                    "irrelevant_kl": irrelevant_kl,
                    "random_kl": random_kl,
                },
            })
        except Exception as exc:  # noqa: BLE001
            records.append({**base, "error": repr(exc)})
    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {
            "seed": seed,
            "severity": pf_a_cfg.get("severity", 1.0),
            "fill": pf_a_cfg.get("fill", [0, 0, 0]),
            "comparison": "clean_vs_region_masked_attention_kl",
        },
        "reduction": reduce_samples(records),
        "samples": records,
    }


def write_schema_smoke(path: str | Path) -> dict:
    payload = build_schema()
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="PF-A Corruption Selectivity",
    kind="internal_curve",
    run_fn=run,
)
