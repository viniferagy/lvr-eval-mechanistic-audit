"""CF-Stage early/mid/late degradation retention."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ... import internal_metrics as IM
from ..base import MetricSpec


METRIC_ID = "cf_stage_decay"
LEGACY_NAME = "cf_stage"
DEFAULT_FAMILIES = {"mask": [0.0, 0.4, 0.8]}
DEFAULT_STAGES = ("early", "mid", "late")


def build_schema() -> dict:
    return {
        "metric_id": METRIC_ID,
        "stages": list(DEFAULT_STAGES),
        "scalars": ["early_auc", "mid_auc", "late_auc", "late_retention"],
        "status": "runnable_v0",
    }


def stage_reduce(curve) -> dict[str, float]:
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("stage_reduce requires a non-empty 1D curve")
    splits = np.array_split(arr, 3)
    return {
        "early": float(splits[0].mean()),
        "mid": float(splits[1].mean()),
        "late": float(splits[2].mean()),
    }


def _curve_features(severities: list[float], values: list[float]) -> dict:
    s = np.asarray(severities, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(s)
    s, v = s[order], v[order]
    span = s[-1] - s[0]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz(v, s) / span) if span > 0 else float(v.mean())
    clean = float(v[0])
    final = float(v[-1])
    return {
        "auc": auc,
        "rel_change": float((final - clean) / (abs(clean) + 1e-9)),
        "clean": clean,
        "final": final,
    }


def _cfg(cfg: dict) -> dict:
    return cfg.get("cf_stage", {}) or {}


def _families(cfg: dict) -> dict:
    local = _cfg(cfg)
    if local.get("families"):
        return dict(local["families"])
    return dict((cfg.get("cf2", {}) or {}).get("families") or DEFAULT_FAMILIES)


def _severity_curve(wrapper, sample, family: str, severity: float, readout: str, cfg: dict) -> dict:
    if readout == "pf3":
        meta = IM.pf3_curve_with_meta_from_sample(
            wrapper,
            sample,
            corruption_mode=family,
            num_seeds=int((cfg.get("pf3", {}) or {}).get("num_seeds", 1)),
            severity=severity,
        )
        return meta
    if readout == "bf3":
        processed, _image_meta = IM.prepare_image_for_audit(wrapper, sample.image)
        image = processed if severity == 0 else IM.corrupt_image(processed, family, seed=0, severity=severity)
        return IM.bf3_curve_with_meta_from_sample(wrapper, replace(sample, image=image))
    raise ValueError(f"unknown cf_stage readout: {readout}")


def _stage_series(records: list[dict], stage: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for record in records:
        stages = record.get("stages") or {}
        if stages.get(stage) is None:
            continue
        xs.append(float(record["severity"]))
        ys.append(float(stages[stage]))
    return xs, ys


def _family_reduction(records: list[dict]) -> dict:
    out = {}
    for stage in DEFAULT_STAGES:
        xs, ys = _stage_series(records, stage)
        if len(xs) >= 2:
            feats = _curve_features(xs, ys)
            out[f"{stage}_auc"] = feats["auc"]
            out[f"{stage}_rel_change"] = feats["rel_change"]
        elif ys:
            out[f"{stage}_auc"] = ys[0]
            out[f"{stage}_rel_change"] = None
        else:
            out[f"{stage}_auc"] = None
            out[f"{stage}_rel_change"] = None
    clean_late = next((r.get("stages", {}).get("late") for r in records if float(r.get("severity", -1)) == 0.0), None)
    final_late = records[-1].get("stages", {}).get("late") if records else None
    out["late_retention"] = (
        float(final_late) / (abs(float(clean_late)) + 1e-9)
        if clean_late is not None and final_late is not None
        else None
    )
    return out


def run(wrapper, samples, cfg: dict, model_tag: str) -> dict:
    local = _cfg(cfg)
    readout = str(local.get("readout", "pf3"))
    max_samples = local.get("max_samples")
    selected = samples[:int(max_samples)] if max_samples is not None else samples
    families = _families(cfg)

    family_results = {}
    sample_rows = []
    for family, severities in families.items():
        sev_values = sorted({0.0, *[float(v) for v in severities]})
        records = []
        for severity in sev_values:
            stage_values = []
            errors = []
            for sample in selected:
                try:
                    meta = _severity_curve(wrapper, sample, family, severity, readout, cfg)
                    if meta.get("curve") is None:
                        errors.append("empty_curve")
                        continue
                    stages = stage_reduce(meta["curve"])
                    stage_values.append(stages)
                    sample_rows.append({
                        "id": sample.id,
                        "family": family,
                        "severity": severity,
                        "reduction": {
                            f"{key}_stage": value
                            for key, value in stages.items()
                        },
                    })
                except Exception as exc:  # noqa: BLE001
                    errors.append(repr(exc))
            aggregate = {
                stage: float(np.mean([row[stage] for row in stage_values]))
                for stage in DEFAULT_STAGES
            } if stage_values else {stage: None for stage in DEFAULT_STAGES}
            records.append({
                "family": family,
                "severity": severity,
                "stages": aggregate,
                "n_success": len(stage_values),
                "n_error": len(errors),
                "errors": errors[:3],
            })
        family_results[family] = {
            "severities": sev_values,
            "records": records,
            "reduction": _family_reduction(records),
        }

    reductions = [fam["reduction"] for fam in family_results.values()]
    aggregate = {}
    for key in sorted({k for reduction in reductions for k in reduction}):
        vals = [float(r[key]) for r in reductions if r.get(key) is not None]
        aggregate[key] = float(np.mean(vals)) if vals else None
    aggregate["n_families"] = len(family_results)
    return {
        "model": model_tag,
        "schema": build_schema(),
        "readout": readout,
        "families": family_results,
        "samples": sample_rows,
        "reduction": aggregate,
    }


SPEC = MetricSpec(
    metric_id=METRIC_ID,
    legacy_name=LEGACY_NAME,
    title="CF-Stage Decay",
    kind="internal_curve",
    run_fn=run,
)
