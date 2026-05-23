"""Sanity checks for W2 v2 metric payloads."""
from __future__ import annotations

from .common import FAIL, PASS, WARN, is_finite_scalar, make_check, make_report


def _v2_cfg(cfg: dict | None) -> dict:
    return (((cfg or {}).get("validation") or {}).get("v2") or {})


def _min_samples(cfg: dict | None) -> int:
    return int(_v2_cfg(cfg).get("min_samples", 50))


def _min_pairs(cfg: dict | None) -> int:
    return int(_v2_cfg(cfg).get("min_pairs", 50))


def _max_error_ratio(cfg: dict | None) -> float:
    return float(_v2_cfg(cfg).get("max_error_ratio", 0.2))


def _allow_center_fallback(cfg: dict | None) -> bool:
    return bool(_v2_cfg(cfg).get("allow_center_fallback", False))


def _finite_reduction_checks(payload: dict, keys: list[str]) -> list[dict]:
    reduction = payload.get("reduction")
    checks = [make_check("reduction_present", PASS if isinstance(reduction, dict) else FAIL)]
    if not isinstance(reduction, dict):
        return checks
    for key in keys:
        checks.append(make_check(
            f"finite_{key}",
            PASS if is_finite_scalar(reduction.get(key)) else FAIL,
            key=key,
            value=reduction.get(key),
        ))
    return checks


def check_trace_v2_result(payload: dict, cfg: dict | None = None) -> dict:
    checks = []
    samples = payload.get("samples") or []
    checks.append(make_check("samples_present", PASS if samples else FAIL, n=len(samples)))
    for idx, sample in enumerate(samples):
        prefix = f"sample_{idx}"
        checks.append(make_check(
            f"{prefix}_instrumented",
            PASS if sample.get("trace_quality") == "instrumented_sparse_v0" else FAIL,
            trace_quality=sample.get("trace_quality"),
        ))
        checks.append(make_check(
            f"{prefix}_no_trace_error",
            PASS if not sample.get("trace_v2_error") else FAIL,
            trace_v2_error=sample.get("trace_v2_error"),
        ))
        checks.append(make_check(
            f"{prefix}_modules_present",
            PASS if sample.get("missing_modules") == [] else FAIL,
            missing_modules=sample.get("missing_modules"),
        ))
        checks.append(make_check(
            f"{prefix}_lvr_mode_steps",
            PASS if int(sample.get("n_lvr_mode_steps") or 0) >= 1 else FAIL,
            n_lvr_mode_steps=sample.get("n_lvr_mode_steps"),
        ))
        checks.append(make_check(
            f"{prefix}_hidden_feedback_steps",
            PASS if int(sample.get("n_hidden_feedback_steps") or 0) >= 1 else FAIL,
            n_hidden_feedback_steps=sample.get("n_hidden_feedback_steps"),
        ))
        checks.append(make_check(
            f"{prefix}_lm_head_called",
            PASS if sample.get("lm_head_called") is True else FAIL,
            lm_head_called=sample.get("lm_head_called"),
        ))
    return make_report("lvr_generation_trace", payload.get("model", "unknown"), checks, "trace_v2")


def check_pf_a_result(payload: dict, cfg: dict | None = None) -> dict:
    checks = _finite_reduction_checks(payload, ["selectivity", "relevant_kl", "irrelevant_kl", "random_kl"])
    reduction = payload.get("reduction") or {}
    checks.append(make_check("min_samples", PASS if int(reduction.get("n") or 0) >= _min_samples(cfg) else FAIL,
                             n=reduction.get("n"), min_samples=_min_samples(cfg)))
    for record in payload.get("samples") or []:
        checks.append(make_check(
            "mask_overlap",
            PASS if int(record.get("relevant_irrelevant_overlap") or 0) == 0 else FAIL,
            sample_id=record.get("id"),
            overlap=record.get("relevant_irrelevant_overlap"),
        ))
        oracle = record.get("relevant_oracle_source")
        if oracle == "center_fallback" and not _allow_center_fallback(cfg):
            status = FAIL
        elif oracle == "center_fallback":
            status = WARN
        else:
            status = PASS if oracle else FAIL
        checks.append(make_check("mask_oracle_source", status, sample_id=record.get("id"), oracle_source=oracle))
    return make_report("pf_a_corruption_selectivity", payload.get("model", "unknown"), checks, "v2")


def check_pf_b_result(payload: dict, cfg: dict | None = None) -> dict:
    checks = _finite_reduction_checks(
        payload,
        ["native_alignment", "relevant_alignment", "irrelevant_alignment", "random_alignment"],
    )
    reduction = payload.get("reduction") or {}
    checks.append(make_check("min_samples", PASS if int(reduction.get("n") or 0) >= _min_samples(cfg) else FAIL,
                             n=reduction.get("n"), min_samples=_min_samples(cfg)))
    use_dino = bool((payload.get("config") or {}).get("use_dino", False))
    for record in payload.get("samples") or []:
        dino = record.get("dino") or {}
        checks.append(make_check(
            "dino_policy",
            PASS if (dino.get("available") or not use_dino) else FAIL,
            sample_id=record.get("id"),
            requested=dino.get("requested"),
            available=dino.get("available"),
            reason=dino.get("reason"),
        ))
        oracle = record.get("relevant_oracle_source")
        if oracle == "center_fallback" and not _allow_center_fallback(cfg):
            status = FAIL
        elif oracle == "center_fallback":
            status = WARN
        else:
            status = PASS if oracle else FAIL
        checks.append(make_check("mask_oracle_source", status, sample_id=record.get("id"), oracle_source=oracle))
    return make_report("pf_b_patch_alignment", payload.get("model", "unknown"), checks, "v2")


def _check_cells(metric_id: str, payload: dict, cfg: dict | None = None) -> dict:
    checks = []
    n_paired = int(payload.get("n_paired") or 0)
    checks.append(make_check("min_pairs", PASS if n_paired >= _min_pairs(cfg) else FAIL,
                             n_paired=n_paired, min_pairs=_min_pairs(cfg)))
    cells = payload.get("cells") or []
    total_success = sum(int(cell.get("n_success") or 0) for cell in cells)
    total_error = sum(int(cell.get("n_error") or 0) for cell in cells)
    checks.append(make_check("cell_success", PASS if total_success > 0 else FAIL,
                             total_success=total_success))
    checks.append(make_check(
        "cell_error_ratio",
        PASS if total_error <= total_success * _max_error_ratio(cfg) else FAIL,
        total_error=total_error,
        total_success=total_success,
        max_error_ratio=_max_error_ratio(cfg),
    ))
    valid_scoring = 0
    for cell in cells:
        for record in cell.get("records") or []:
            if record.get("source_answer_token_ids") and record.get("target_answer_token_ids"):
                if is_finite_scalar(record.get("clean_margin")) and is_finite_scalar(record.get("patched_margin")):
                    valid_scoring += 1
    checks.append(make_check("answer_sequence_scoring", PASS if valid_scoring > 0 else FAIL,
                             valid_scoring=valid_scoring))
    return make_report(metric_id, payload.get("model", "unknown"), checks, "v2")


def check_bf_patch_result(payload: dict, cfg: dict | None = None) -> dict:
    return _check_cells("bf_patch_answer_transfer", payload, cfg)


def check_bf_swap_result(payload: dict, cfg: dict | None = None) -> dict:
    report = _check_cells("bf_swap_latent_replacement", payload, cfg)
    checks = list(report.get("checks") or [])
    control_cells = payload.get("control_cells") or []
    controls = {cell.get("control") for cell in control_cells}
    for required in ("self_swap", "reverse_swap", "random_pair_swap"):
        checks.append(make_check(
            f"{required}_recorded",
            PASS if required in controls else FAIL,
            controls=sorted(str(c) for c in controls if c),
        ))
    self_shifts = [
        abs(float(cell.get("swap_margin_shift")))
        for cell in control_cells
        if cell.get("control") == "self_swap" and is_finite_scalar(cell.get("swap_margin_shift"))
    ]
    tolerance = float(_v2_cfg(cfg).get("self_swap_max_abs_shift", 0.05))
    checks.append(make_check(
        "self_swap_near_zero",
        PASS if self_shifts and max(self_shifts) <= tolerance else FAIL,
        max_abs_shift=max(self_shifts) if self_shifts else None,
        tolerance=tolerance,
    ))
    return make_report("bf_swap_latent_replacement", payload.get("model", "unknown"), checks, "v2")


def check_bf_conf_result(payload: dict, cfg: dict | None = None) -> dict:
    checks = _finite_reduction_checks(payload, ["gold_logit_slope"])
    min_layers = int((((cfg or {}).get("validation") or {}).get("bf3") or {}).get("min_layers", 20))
    for sample in payload.get("samples") or []:
        curve = sample.get("curve") or []
        checks.append(make_check("curve_length", PASS if len(curve) >= min_layers else FAIL,
                                 sample_id=sample.get("id"), length=len(curve), min_layers=min_layers))
        checks.append(make_check("text_only_control_recorded", PASS if "text_only_control" in sample else FAIL,
                                 sample_id=sample.get("id")))
    return make_report("bf_conf_calibrated_progression", payload.get("model", "unknown"), checks, "v2")


def check_cf_stage_result(payload: dict, cfg: dict | None = None) -> dict:
    checks = _finite_reduction_checks(payload, ["late_delta"])
    reduction = payload.get("reduction") or {}
    if is_finite_scalar(reduction.get("late_retention")):
        checks.append(make_check(
            "late_retention_diagnostic_only",
            PASS,
            late_retention=reduction.get("late_retention"),
            unstable_near_zero_baseline=reduction.get("late_retention_unstable_near_zero_baseline"),
        ))
    for family, family_result in (payload.get("families") or {}).items():
        records = family_result.get("records") or []
        checks.append(make_check("family_min_severities", PASS if len(records) >= 2 else FAIL,
                                 family=family, n_severities=len(records)))
        for stage in ("early", "mid", "late"):
            any_finite = any(is_finite_scalar((rec.get("stages") or {}).get(stage)) for rec in records)
            checks.append(make_check(f"{family}_{stage}_finite", PASS if any_finite else FAIL))
    return make_report("cf_stage_decay", payload.get("model", "unknown"), checks, "v2")
