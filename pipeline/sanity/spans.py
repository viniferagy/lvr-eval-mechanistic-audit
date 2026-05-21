"""Sanity checks for adapter-declared audit spans."""
from __future__ import annotations

from .common import FAIL, PASS, WARN, make_check, make_report


DEFAULTS = {
    "min_valid_rate": 0.8,
}


def _span_len(span) -> int | None:
    if not isinstance(span, list) or len(span) != 2:
        return None
    return int(span[1]) - int(span[0])


def _fully_overlap(a, b) -> bool:
    if not isinstance(a, list) or not isinstance(b, list):
        return False
    return int(a[0]) >= int(b[0]) and int(a[1]) <= int(b[1])


def check_span_metadata(result: dict, metric_id: str,
                        cfg: dict | None = None,
                        source: str = "span_metadata") -> dict:
    thresholds = dict(DEFAULTS)
    thresholds.update(((cfg or {}).get("validation", {}).get("spans", {})))

    model = str(result.get("model", "unknown"))
    baseline = result.get("baseline") if isinstance(result.get("baseline"), dict) else {}
    meta = result.get("span_metadata") or baseline.get("span_metadata") or {}
    checks = []
    details = {"thresholds": thresholds, "span_metadata": meta}

    checks.append(make_check(
        "span_metadata_present",
        PASS if isinstance(meta, dict) and bool(meta) else WARN,
    ))
    if not meta:
        return make_report(metric_id, model, checks, source, details)

    valid_rate = float(meta.get("valid_rate", 0.0))
    checks.append(make_check(
        "valid_rate",
        PASS if valid_rate >= float(thresholds["min_valid_rate"]) else WARN,
        valid_rate=valid_rate,
        min_valid_rate=float(thresholds["min_valid_rate"]),
    ))

    counts = meta.get("query_target_counts", {})
    is_qwen = "qwen" in model.lower() and "lvr" not in model.lower()
    is_lvr = "lvr" in model.lower()
    if is_qwen:
        checks.append(make_check(
            "qwen_uses_answer_probe",
            PASS if counts.get("answer_probe_pos", 0) > 0 else WARN,
            "Qwen baseline should not report LVR latent tokens",
            query_target_counts=counts,
        ))
    if is_lvr:
        audit_cfg = (cfg or {}).get("audit", {}) if isinstance(cfg or {}, dict) else {}
        if audit_cfg.get("mode", "teacher_forced") == "teacher_forced":
            allow_fallback = bool(audit_cfg.get("allow_lvr_fallback_to_answer_probe", False))
            checks.append(make_check(
                "lvr_teacher_forced_uses_lvr_placeholder",
                PASS if counts.get("lvr_placeholder_tokens", 0) > 0 else (WARN if allow_fallback else FAIL),
                "LVR teacher-forced audit must use <|lvr|> placeholder tokens",
                query_target_counts=counts,
                allow_fallback=allow_fallback,
            ))
        else:
            checks.append(make_check(
                "lvr_uses_lvr_placeholder_when_available",
                PASS if counts.get("lvr_placeholder_tokens", 0) > 0 else WARN,
                "LVR should prefer <|lvr|> placeholder tokens when present",
                query_target_counts=counts,
            ))

    for idx, rec in enumerate(meta.get("examples", [])):
        query_span = rec.get("query_span")
        image_span = rec.get("image_span")
        q_len = _span_len(query_span)
        i_len = _span_len(image_span)
        checks.append(make_check(
            f"example_{idx}.span_lengths",
            PASS if (q_len is not None and q_len > 0 and i_len is not None and i_len > 0) else FAIL,
            query_span=query_span,
            image_span=image_span,
        ))
        checks.append(make_check(
            f"example_{idx}.query_not_inside_image",
            PASS if not _fully_overlap(query_span, image_span) else WARN,
            "query span should not be fully inside image span for text/LVR audit",
            query_span=query_span,
            image_span=image_span,
        ))
        notes = rec.get("adapter_notes", {})
        if notes.get("adapter") == "qwen_vl":
            checks.append(make_check(
                f"example_{idx}.qwen_no_latent",
                PASS if notes.get("latent_tokens") == "none_for_qwen_baseline" else WARN,
                adapter_notes=notes,
            ))

    return make_report(metric_id, model, checks, source, details)
