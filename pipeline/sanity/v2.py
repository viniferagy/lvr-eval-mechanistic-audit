"""Sanity checks for W2 v2 metric payloads."""
from __future__ import annotations

import math

from .common import FAIL, PASS, WARN, is_finite_scalar, make_check, make_report


ACCEPTED_TRACE_QUALITIES = {"instrumented_sparse_v0", "monet_vllm_latent_v0"}


def _v2_cfg(cfg: dict | None) -> dict:
    return (((cfg or {}).get("validation") or {}).get("v2") or {})


def _min_samples(cfg: dict | None) -> int:
    return int(_v2_cfg(cfg).get("min_samples", 50))


def _min_pairs(cfg: dict | None) -> int:
    return int(_v2_cfg(cfg).get("min_pairs", 50))


def _max_error_ratio(cfg: dict | None) -> float:
    return float(_v2_cfg(cfg).get("max_error_ratio", 0.2))


def _parsed_fallback_warn_ratio(cfg: dict | None) -> float:
    return float(_v2_cfg(cfg).get("parsed_fallback_warn_ratio", 0.2))


def _parsed_fallback_fail_ratio(cfg: dict | None) -> float:
    return float(_v2_cfg(cfg).get("parsed_fallback_fail_ratio", 0.5))


def _w3_cfg(cfg: dict | None) -> dict:
    return (((cfg or {}).get("validation") or {}).get("w3") or {})


def _w3_min_pairs(cfg: dict | None) -> int:
    return int(_w3_cfg(cfg).get("min_pairs", 50))


def _w3_max_error_ratio(cfg: dict | None) -> float:
    return float(_w3_cfg(cfg).get("max_error_ratio", 0.2))


def _w4_cfg(cfg: dict | None) -> dict:
    return (((cfg or {}).get("validation") or {}).get("w4") or {})


def _w4_min_pairs(cfg: dict | None) -> int:
    return int(_w4_cfg(cfg).get("min_pairs", _w3_min_pairs(cfg)))


def _w4_min_steps(cfg: dict | None) -> int:
    return int(_w4_cfg(cfg).get("min_steps", 2))


def _monet_cfg(cfg: dict | None) -> dict:
    return (((cfg or {}).get("validation") or {}).get("monet") or {})


def _monet_min_pairs(cfg: dict | None) -> int:
    return int(_monet_cfg(cfg).get("min_pairs", 50))


def _monet_max_error_ratio(cfg: dict | None) -> float:
    return float(_monet_cfg(cfg).get("max_error_ratio", 0.2))


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
            PASS if sample.get("trace_quality") in ACCEPTED_TRACE_QUALITIES else FAIL,
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
    continuous_margin_records = 0
    parsed_margin_records = 0
    trace_records = 0
    for cell in cells:
        for record in cell.get("records") or []:
            if record.get("source_answer_token_ids") and record.get("target_answer_token_ids"):
                if is_finite_scalar(record.get("clean_margin")) and is_finite_scalar(record.get("patched_margin")):
                    valid_scoring += 1
            margin_source = record.get("margin_source")
            if margin_source in {"generation_scores", "generation_scores_aligned_first_token", "latent_logit_lens"}:
                continuous_margin_records += 1
            elif margin_source == "parsed_answer_fallback":
                parsed_margin_records += 1
            if (
                record.get("trace_quality") in ACCEPTED_TRACE_QUALITIES
                and record.get("source_trace_quality") in ACCEPTED_TRACE_QUALITIES
                and int(record.get("n_patch_applied") or 0) >= 1
            ):
                trace_records += 1
    checks.append(make_check("answer_sequence_scoring", PASS if valid_scoring > 0 else FAIL,
                             valid_scoring=valid_scoring))
    if (payload.get("config") or {}).get("trace_latent"):
        score_diag_records = 0
        score_failure_reason_counts: dict[str, int] = {}
        for cell in cells:
            diag = cell.get("generation_score_diagnostics") or {}
            score_diag_records += int(diag.get("diagnostic_records") or 0)
            for reason, count in (diag.get("generation_score_failure_reason_counts") or {}).items():
                score_failure_reason_counts[str(reason)] = score_failure_reason_counts.get(str(reason), 0) + int(count)
        if trace_records > 0 and valid_scoring == 0:
            checks[-1]["status"] = WARN
            checks[-1]["values"]["reason"] = "continuous_margin_unavailable"
        margin_total = continuous_margin_records + parsed_margin_records
        parsed_ratio = (
            float(parsed_margin_records) / float(margin_total)
            if margin_total else math.nan
        )
        warn_ratio = _parsed_fallback_warn_ratio(cfg)
        fail_ratio = _parsed_fallback_fail_ratio(cfg)
        if margin_total == 0:
            parsed_status = WARN
        elif parsed_ratio > fail_ratio:
            parsed_status = FAIL
        elif parsed_ratio > warn_ratio:
            parsed_status = WARN
        else:
            parsed_status = PASS
        checks.append(make_check(
            "continuous_margin_source",
            PASS if continuous_margin_records > 0 else WARN,
            continuous_margin_records=continuous_margin_records,
            parsed_answer_fallback_records=parsed_margin_records,
        ))
        checks.append(make_check(
            "parsed_fallback_ratio",
            parsed_status,
            parsed_fallback_ratio=None if math.isnan(parsed_ratio) else parsed_ratio,
            warn_ratio=warn_ratio,
            fail_ratio=fail_ratio,
            continuous_margin_records=continuous_margin_records,
            parsed_answer_fallback_records=parsed_margin_records,
        ))
        checks.append(make_check(
            "generation_score_diagnostics_recorded",
            PASS if score_diag_records > 0 else WARN,
            diagnostic_records=score_diag_records,
            failure_reason_counts=score_failure_reason_counts,
        ))
        checks.append(make_check(
            "trace_latent_patch_records",
            PASS if trace_records > 0 else FAIL,
            trace_records=trace_records,
        ))
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
    trace_latent = bool((payload.get("config") or {}).get("trace_latent"))
    has_continuous_margin = any(
        (record.get("margin_source") in {"generation_scores", "generation_scores_aligned_first_token", "latent_logit_lens"})
        for cell in (payload.get("cells") or [])
        for record in (cell.get("records") or [])
        if record.get("error") is None
    )
    checks.append(make_check(
        "self_swap_near_zero",
        PASS if self_shifts and max(self_shifts) <= tolerance else (WARN if trace_latent and (not self_shifts or has_continuous_margin) else FAIL),
        max_abs_shift=max(self_shifts) if self_shifts else None,
        tolerance=tolerance,
        reason="continuous_margin_available_but_self_swap_cell_missing" if trace_latent and not self_shifts and has_continuous_margin else None,
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


def check_lvr_latent_patch_result(payload: dict, cfg: dict | None = None) -> dict:
    checks = _finite_reduction_checks(payload, ["latent_answer_transfer_rate"])
    reduction = payload.get("reduction") or {}
    n_paired = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
    min_pairs = _w4_min_pairs(cfg) if _w4_cfg(cfg) else _w3_min_pairs(cfg)
    checks.append(make_check(
        "min_pairs",
        PASS if n_paired >= min_pairs else FAIL,
        n_paired=n_paired,
        min_pairs=min_pairs,
    ))
    n_success = int(reduction.get("n_success") or 0)
    n_error = int(reduction.get("n_error") or 0)
    n_patch_applied = int(reduction.get("n_patch_applied") or 0)
    n_with_lvr_mode = int(reduction.get("n_with_lvr_mode") or 0)
    n_with_captured_state = int(reduction.get("n_with_captured_state") or 0)
    checks.append(make_check("success_present", PASS if n_success > 0 else FAIL, n_success=n_success))
    checks.append(make_check(
        "error_ratio",
        PASS if n_error <= max(1, n_success) * _w3_max_error_ratio(cfg) else FAIL,
        n_success=n_success,
        n_error=n_error,
        max_error_ratio=_w3_max_error_ratio(cfg),
    ))
    checks.append(make_check(
        "patch_applied_min_pairs",
        PASS if n_patch_applied >= min_pairs else FAIL,
        n_patch_applied=n_patch_applied,
        min_pairs=min_pairs,
    ))
    checks.append(make_check(
        "lvr_mode_min_pairs",
        PASS if n_with_lvr_mode >= min_pairs else FAIL,
        n_with_lvr_mode=n_with_lvr_mode,
        min_pairs=min_pairs,
    ))
    checks.append(make_check(
        "captured_state_min_pairs",
        PASS if n_with_captured_state >= min_pairs else FAIL,
        n_with_captured_state=n_with_captured_state,
        min_pairs=min_pairs,
    ))

    valid_trace = 0
    answer_records = 0
    shape_records = 0
    for record in payload.get("samples") or []:
        if record.get("error") is not None:
            continue
        if (
            record.get("trace_quality") in ACCEPTED_TRACE_QUALITIES
            and record.get("source_trace_quality") in ACCEPTED_TRACE_QUALITIES
            and not record.get("trace_v2_error")
            and record.get("missing_modules") == []
        ):
            valid_trace += 1
        if record.get("clean_answer") is not None and record.get("patched_answer") is not None:
            answer_records += 1
        shapes = record.get("captured_state_shapes") or []
        patched_shapes = record.get("patched_captured_state_shapes") or []
        if shapes and patched_shapes and shapes[-1] and patched_shapes[-1] and shapes[-1][-1] == patched_shapes[-1][-1]:
            shape_records += 1
    checks.append(make_check("instrumented_trace_records", PASS if valid_trace >= min_pairs else FAIL,
                             valid_trace=valid_trace, min_pairs=min_pairs))
    checks.append(make_check("generated_answers_recorded", PASS if answer_records >= min_pairs else FAIL,
                             answer_records=answer_records, min_pairs=min_pairs))
    checks.append(make_check("captured_shape_match", PASS if shape_records >= min_pairs else FAIL,
                             shape_records=shape_records, min_pairs=min_pairs))
    min_steps = _w4_min_steps(cfg)
    n_steps = int(reduction.get("n_steps_evaluated") or 0)
    if n_steps >= min_steps or _w4_cfg(cfg):
        checks.append(make_check(
            "step_sweep_min_steps",
            PASS if n_steps >= min_steps else FAIL,
            n_steps_evaluated=n_steps,
            min_steps=min_steps,
        ))
        for key in ("best_step_transfer_rate", "step_transfer_auc", "last_step_transfer_rate"):
            checks.append(make_check(
                f"finite_{key}",
                PASS if is_finite_scalar(reduction.get(key)) else FAIL,
                key=key,
                value=reduction.get(key),
            ))
        checks.append(make_check(
            "best_step_index_present",
            PASS if reduction.get("best_step_index") is not None else FAIL,
            best_step_index=reduction.get("best_step_index"),
        ))
        per_step = reduction.get("per_step") or []
        checks.append(make_check(
            "per_step_records_present",
            PASS if len(per_step) >= min_steps else FAIL,
            n_per_step=len(per_step),
            min_steps=min_steps,
        ))
        step_result_records = 0
        for record in payload.get("samples") or []:
            if record.get("error") is not None:
                continue
            step_results = record.get("step_results") or []
            if len(step_results) >= min_steps:
                step_result_records += 1
        checks.append(make_check(
            "sample_step_results_present",
            PASS if step_result_records >= min_pairs else FAIL,
            step_result_records=step_result_records,
            min_pairs=min_pairs,
        ))
    return make_report("lvr_latent_patch_answer_transfer", payload.get("model", "unknown"), checks, "w3")


def check_monet_latent_patch_result(payload: dict, cfg: dict | None = None) -> dict:
    checks = _finite_reduction_checks(payload, ["latent_answer_transfer_rate"])
    reduction = payload.get("reduction") or {}
    n_paired = int(payload.get("n_paired") or reduction.get("n_paired") or 0)
    min_pairs = _monet_min_pairs(cfg)
    checks.append(make_check(
        "min_pairs",
        PASS if n_paired >= min_pairs else FAIL,
        n_paired=n_paired,
        min_pairs=min_pairs,
    ))
    n_success = int(reduction.get("n_success") or 0)
    n_error = int(reduction.get("n_error") or 0)
    n_patch_applied = int(reduction.get("n_patch_applied") or 0)
    n_with_captured_state = int(reduction.get("n_with_captured_state") or 0)
    checks.append(make_check("success_min_pairs", PASS if n_success >= min_pairs else FAIL,
                             n_success=n_success, min_pairs=min_pairs))
    checks.append(make_check(
        "error_ratio",
        PASS if n_error <= max(1, n_success) * _monet_max_error_ratio(cfg) else FAIL,
        n_success=n_success,
        n_error=n_error,
        max_error_ratio=_monet_max_error_ratio(cfg),
    ))
    checks.append(make_check(
        "patch_applied_min_pairs",
        PASS if n_patch_applied >= min_pairs else FAIL,
        n_patch_applied=n_patch_applied,
        min_pairs=min_pairs,
    ))
    checks.append(make_check(
        "captured_state_min_pairs",
        PASS if n_with_captured_state >= min_pairs else FAIL,
        n_with_captured_state=n_with_captured_state,
        min_pairs=min_pairs,
    ))

    trace_records = 0
    answer_records = 0
    shape_records = 0
    for record in payload.get("samples") or []:
        if record.get("error") is not None:
            continue
        if (
            record.get("trace_quality") == "monet_transformers_latent_mode_v0"
            and record.get("source_trace_quality") == "monet_transformers_latent_mode_v0"
            and record.get("target_trace_quality") == "monet_transformers_latent_mode_v0"
        ):
            trace_records += 1
        if record.get("clean_answer") is not None and record.get("patched_answer") is not None:
            answer_records += 1
        shapes = record.get("captured_state_shapes") or []
        target_shapes = record.get("target_captured_state_shapes") or []
        patch_shapes = record.get("patch_state_shapes") or []
        if shapes and target_shapes and patch_shapes and shapes[-1] and target_shapes[-1] and patch_shapes[-1]:
            if shapes[-1][-1] == target_shapes[-1][-1] == patch_shapes[-1][-1]:
                shape_records += 1
    checks.append(make_check("monet_latent_trace_records", PASS if trace_records >= min_pairs else FAIL,
                             trace_records=trace_records, min_pairs=min_pairs))
    checks.append(make_check("candidate_answers_recorded", PASS if answer_records >= min_pairs else FAIL,
                             answer_records=answer_records, min_pairs=min_pairs))
    checks.append(make_check("captured_shape_match", PASS if shape_records >= min_pairs else FAIL,
                             shape_records=shape_records, min_pairs=min_pairs))
    return make_report("monet_latent_patch_answer_transfer", payload.get("model", "unknown"), checks, "w13_monet")


def check_output_accuracy_result(payload: dict, cfg: dict | None = None) -> dict:
    checks = _finite_reduction_checks(payload, ["accuracy"])
    reduction = payload.get("reduction") or {}
    min_samples = int((((cfg or {}).get("validation") or {}).get("output_accuracy") or {}).get(
        "min_samples",
        _min_samples(cfg),
    ))
    n = int(reduction.get("n") or 0)
    n_error = int(reduction.get("n_error") or 0)
    max_error_ratio = float((((cfg or {}).get("validation") or {}).get("output_accuracy") or {}).get(
        "max_error_ratio",
        _max_error_ratio(cfg),
    ))
    checks.append(make_check("min_samples", PASS if n >= min_samples else FAIL, n=n, min_samples=min_samples))
    checks.append(make_check(
        "error_ratio",
        PASS if n_error <= max(1, n) * max_error_ratio else FAIL,
        n=n,
        n_error=n_error,
        max_error_ratio=max_error_ratio,
    ))
    prediction_records = 0
    for record in payload.get("samples") or []:
        if record.get("error") is None and record.get("prediction") is not None:
            prediction_records += 1
    checks.append(make_check(
        "predictions_recorded",
        PASS if prediction_records >= min_samples else FAIL,
        prediction_records=prediction_records,
        min_samples=min_samples,
    ))
    return make_report("output_accuracy_sanity", payload.get("model", "unknown"), checks, "output_accuracy")
