"""Sanity suite orchestration and artifact writing."""
from __future__ import annotations

import json
import os
import re
from typing import Iterable

from .bf1 import check_bf1_result
from .bf3 import check_bf3_curve
from .cf2 import check_cf2_result
from .common import FAIL, WARN
from .pf3 import check_pf3_curve
from .spans import check_span_metadata
from .v2 import (
    check_bf_conf_result,
    check_bf_patch_result,
    check_bf_swap_result,
    check_cf_stage_result,
    check_lvr_latent_patch_result,
    check_monet_latent_patch_result,
    check_pf_a_result,
    check_pf_b_result,
    check_trace_v2_result,
)


def run_sanity_suite(ablation_results: dict[str, dict],
                     decay_results: dict[str, dict],
                     cfg: dict | None = None) -> list[dict]:
    """Build sanity reports for all currently available metric outputs."""
    reports: list[dict] = []

    for tag, result in ablation_results.items():
        reports.append(check_bf1_result(result, cfg))
        reports.append(check_span_metadata(result, "bf1_latent_ablation", cfg))
        baseline = result.get("baseline", {})
        if baseline.get("bf3_curve") is not None:
            reports.append(check_bf3_curve(
                baseline["bf3_curve"], tag, cfg, source="bf1_baseline"
            ))
        if baseline.get("pf3_curve") is not None:
            reports.append(check_pf3_curve(
                baseline["pf3_curve"], tag, cfg, source="bf1_baseline"
            ))

    for tag, result in decay_results.items():
        reports.append(check_cf2_result(result, cfg))

    return reports


def run_sanity_for_metric_result(metric_id: str, result: dict,
                                 cfg: dict | None = None) -> list[dict]:
    """Build sanity reports for one metric result payload."""
    if metric_id in {"bf1_latent_ablation", "bf1_latent_ablation_legacy"}:
        reports = [
            check_bf1_result(result, cfg),
            check_span_metadata(result, metric_id, cfg),
        ]
        baseline = result.get("baseline", {})
        if baseline.get("bf3_curve") is not None:
            reports.append(check_bf3_curve(
                baseline["bf3_curve"], result.get("model", "unknown"),
                cfg, source="bf1_baseline"
            ))
        if baseline.get("pf3_curve") is not None:
            reports.append(check_pf3_curve(
                baseline["pf3_curve"], result.get("model", "unknown"),
                cfg, source="bf1_baseline"
            ))
        return reports
    if metric_id in {"bf1_layer_ablation", "bf1_layer_ablation_legacy"}:
        return [
            check_bf1_result(result, cfg),
            check_span_metadata(result, metric_id, cfg),
        ]
    if metric_id in {"cf2_pf_decay_curve", "cf2_pf_decay_curve_legacy"}:
        return [check_cf2_result(result, cfg)]
    if metric_id in {"bf3_confidence_progression", "bf3_confidence_progression_legacy"}:
        reports = [check_span_metadata(result, metric_id, cfg)]
        if result.get("curve") is not None:
            reports.append(check_bf3_curve(
                result["curve"], result.get("model", "unknown"),
                cfg, source="standalone"
            ))
        return reports
    if metric_id in {"pf3_attention_distance", "pf3_attention_distance_legacy"}:
        reports = [check_span_metadata(result, metric_id, cfg)]
        if result.get("curve") is not None:
            reports.append(check_pf3_curve(
                result["curve"], result.get("model", "unknown"),
                cfg, source="standalone"
            ))
        return reports
    if metric_id == "lvr_generation_trace":
        return [check_trace_v2_result(result, cfg)]
    if metric_id == "pf_a_corruption_selectivity":
        return [check_pf_a_result(result, cfg)]
    if metric_id == "pf_b_patch_alignment":
        return [check_pf_b_result(result, cfg)]
    if metric_id == "bf_patch_answer_transfer":
        return [check_bf_patch_result(result, cfg)]
    if metric_id == "bf_swap_latent_replacement":
        return [check_bf_swap_result(result, cfg)]
    if metric_id == "bf_conf_calibrated_progression":
        return [check_bf_conf_result(result, cfg)]
    if metric_id == "cf_stage_decay":
        return [check_cf_stage_result(result, cfg)]
    if metric_id == "lvr_latent_patch_answer_transfer":
        return [check_lvr_latent_patch_result(result, cfg)]
    if metric_id == "monet_latent_patch_answer_transfer":
        return [check_monet_latent_patch_result(result, cfg)]
    return []


def run_sanity_for_metric_results(metric_results: list[dict],
                                  cfg: dict | None = None) -> list[dict]:
    """Build sanity reports from unified metric result envelopes."""
    reports: list[dict] = []
    for envelope in metric_results:
        reports.extend(run_sanity_for_metric_result(
            str(envelope.get("metric_id")),
            envelope.get("payload", {}),
            cfg,
        ))
    return reports


def has_failed_checks(reports: Iterable[dict]) -> bool:
    return any(r.get("status") == FAIL or not r.get("passed", False) for r in reports)


def summarize_reports(reports: list[dict]) -> dict:
    total = len(reports)
    failed = sum(1 for r in reports if r.get("status") == FAIL)
    warned = sum(1 for r in reports if r.get("status") == WARN)
    passed = total - failed
    return {
        "total_reports": total,
        "passed_reports": passed,
        "warned_reports": warned,
        "failed_reports": failed,
        "overall_status": FAIL if failed else (WARN if warned else "pass"),
        "reports": [
            {
                "metric_id": r.get("metric_id"),
                "model": r.get("model"),
                "source": r.get("source"),
                "status": r.get("status"),
                "passed": r.get("passed"),
            }
            for r in reports
        ],
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def save_sanity_reports(reports: list[dict], out_dir: str) -> str:
    """Write per-report JSON files plus a summary under out_dir/sanity."""
    sanity_dir = os.path.join(out_dir, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)

    for report in reports:
        metric = _safe_name(str(report.get("metric_id", "metric")))
        model = _safe_name(str(report.get("model", "model")))
        source = _safe_name(str(report.get("source") or "result"))
        path = os.path.join(sanity_dir, f"{metric}_{model}_{source}_sanity.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    summary = summarize_reports(reports)
    summary_path = os.path.join(sanity_dir, "summary_sanity.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return sanity_dir
