"""PF-A relevant-vs-irrelevant corruption selectivity skeleton."""
from __future__ import annotations

import json
from pathlib import Path

from ...corruptions import apply_mask, irrelevant_mask, random_mask, relevant_mask
from ..base import MetricSpec


METRIC_ID = "pf_a_corruption_selectivity"
LEGACY_NAME = "pf_a"


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "required_sample_fields": ["image", "question", "bboxes or region_mask"],
        "scalars": ["selectivity", "relevant_kl", "irrelevant_kl", "random_kl"],
        "status": "skeleton_fixture_only",
    }


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    records = []
    for sample in samples:
        rel = relevant_mask(sample.image, bboxes=sample.bboxes, region_mask=sample.region_mask)
        irr = irrelevant_mask(sample.image, rel, seed=int(cfg.get("pf_a", {}).get("seed", 0)))
        rnd = random_mask(sample.image, coverage=max(rel.coverage, 0.01), seed=0)
        records.append({
            "id": sample.id,
            "relevant_coverage": rel.coverage,
            "irrelevant_coverage": irr.coverage,
            "random_coverage": rnd.coverage,
            "relevant_irrelevant_overlap": int((rel.data & irr.data).sum()),
            "severity0_preserves_image": list(apply_mask(sample.image, rel, severity=0).size) == list(sample.image.size),
        })
    return {"model": model_tag, "schema": build_schema(), "samples": records}


def write_schema_smoke(path: str | Path) -> dict:
    payload = build_schema()
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="PF-A Corruption Selectivity",
    kind="v2_skeleton",
    run_fn=run,
)
