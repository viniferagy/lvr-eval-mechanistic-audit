"""PF-B native patch-alignment entry point."""
from __future__ import annotations

import numpy as np

from ... import internal_metrics as IM
from ...corruptions import apply_mask, irrelevant_mask, random_mask, relevant_mask
from ..base import MetricSpec


METRIC_ID = "pf_b_patch_alignment"
LEGACY_NAME = "pf_b"


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "scalars": ["native_alignment", "relevant_alignment", "irrelevant_alignment", "random_alignment"],
        "external_backends": {"dino": "optional_not_required"},
        "status": "runnable_native_v0",
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("pf_b", {}) or {}


def _attention_alignment(wrapper, sample, mask, cfg: dict) -> dict:
    local = _cfg(cfg)
    masked = apply_mask(
        sample.image,
        mask,
        fill=tuple(local.get("fill", [0, 0, 0])),
        severity=float(local.get("severity", 1.0)),
    )
    meta = IM.query_image_attention_kl_with_meta_from_sample(wrapper, sample, masked)
    curve = meta.get("curve")
    mean_kl = None if curve is None else float(IM.pf3_reduce(np.asarray(curve, dtype=float))["mean_kl"])
    return {
        "mean_kl": mean_kl,
        "alignment": None if mean_kl is None else float(1.0 / (1.0 + max(mean_kl, 0.0))),
        "skip_reasons": meta.get("skip_reasons", {}),
        "query_target_kind": meta.get("query_target_kind"),
        "query_span": meta.get("query_span"),
        "image_span": meta.get("image_span"),
    }


def _aggregate(records: list[dict]) -> dict | None:
    valid = [r for r in records if r.get("native_alignment") is not None]
    if not valid:
        return None
    out = {}
    for key in ("native_alignment", "relevant_alignment", "irrelevant_alignment", "random_alignment"):
        vals = [float(r[key]) for r in valid if r.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    out["n"] = len(valid)
    return out


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    local = _cfg(cfg)
    seed = int(local.get("seed", 0))
    use_dino = bool(local.get("use_dino", False))
    records = []
    for sample in samples:
        rel = relevant_mask(sample.image, bboxes=sample.bboxes, region_mask=sample.region_mask)
        irr = irrelevant_mask(sample.image, rel, seed=seed)
        rnd = random_mask(sample.image, coverage=max(rel.coverage, 0.01), seed=seed)
        base = {
            "id": sample.id,
            "relevant_coverage": rel.coverage,
            "irrelevant_coverage": irr.coverage,
            "random_coverage": rnd.coverage,
        }
        try:
            rel_meta = _attention_alignment(wrapper, sample, rel, cfg)
            irr_meta = _attention_alignment(wrapper, sample, irr, cfg)
            rnd_meta = _attention_alignment(wrapper, sample, rnd, cfg)
            native = rel_meta["alignment"]
            reduction = {
                "native_alignment": native,
                "relevant_alignment": rel_meta["alignment"],
                "irrelevant_alignment": irr_meta["alignment"],
                "random_alignment": rnd_meta["alignment"],
            }
            records.append({
                **base,
                **reduction,
                "relevant_kl": rel_meta["mean_kl"],
                "irrelevant_kl": irr_meta["mean_kl"],
                "random_kl": rnd_meta["mean_kl"],
                "query_target_kind": rel_meta.get("query_target_kind"),
                "query_span": rel_meta.get("query_span"),
                "image_span": rel_meta.get("image_span"),
                "dino": {
                    "requested": use_dino,
                    "available": False,
                    "reason": "optional_backend_not_configured" if not use_dino else "not_implemented_in_native_smoke",
                },
                "skip_reasons": {
                    "relevant": rel_meta.get("skip_reasons", {}),
                    "irrelevant": irr_meta.get("skip_reasons", {}),
                    "random": rnd_meta.get("skip_reasons", {}),
                },
                "reduction": reduction,
            })
        except Exception as exc:  # noqa: BLE001
            records.append({**base, "error": repr(exc)})

    return {
        "model": model_tag,
        "schema": build_schema(),
        "config": {"seed": seed, "use_dino": use_dino, "severity": local.get("severity", 1.0)},
        "samples": records,
        "reduction": _aggregate(records),
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="PF-B Patch Alignment",
    kind="internal_curve",
    run_fn=run,
)
