#!/usr/bin/env python
"""Validate trace-latent patch margin provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PATCH_METRICS = {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}
CONTINUOUS_MARGIN_SOURCES = {"generation_scores", "generation_scores_aligned_first_token", "latent_logit_lens"}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def _load_payloads(run_dir: Path) -> dict[str, list[dict]]:
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        fail(f"missing metrics dir: {metrics_dir}")
    out: dict[str, list[dict]] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        env = json.loads(path.read_text(encoding="utf-8"))
        metric_id = str(env.get("metric_id") or "")
        if metric_id in PATCH_METRICS:
            out.setdefault(metric_id, []).append(env.get("payload") or {})
    return out


def _quality(payload: dict) -> dict:
    total = 0
    continuous = 0
    parsed = 0
    missing = 0
    sources: dict[str, int] = {}
    diag_records = 0
    missing_diag_records = 0
    reason_counts: dict[str, int] = {}
    clean_ok = 0
    patched_ok = 0
    both_ok = 0
    for cell in payload.get("cells") or []:
        diag = cell.get("generation_score_diagnostics") or {}
        diag_records += int(diag.get("diagnostic_records") or 0)
        missing_diag_records += int(diag.get("missing_diagnostic_records") or 0)
        clean_ok += int(diag.get("clean_generation_scores_ok") or 0)
        patched_ok += int(diag.get("patched_generation_scores_ok") or 0)
        both_ok += int(diag.get("both_generation_scores_ok") or 0)
        for reason, count in (diag.get("generation_score_failure_reason_counts") or {}).items():
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(count)
        for record in cell.get("records") or []:
            if record.get("error") is not None:
                continue
            total += 1
            source = str(record.get("margin_source") or "missing")
            sources[source] = sources.get(source, 0) + 1
            if source in CONTINUOUS_MARGIN_SOURCES:
                continuous += 1
            elif source == "parsed_answer_fallback":
                parsed += 1
            else:
                missing += 1
    return {
        "total": total,
        "continuous": continuous,
        "parsed": parsed,
        "missing": missing,
        "parsed_ratio": float(parsed) / float(total) if total else 1.0,
        "sources": sources,
        "diagnostic_records": diag_records,
        "missing_diagnostic_records": missing_diag_records,
        "clean_generation_scores_ok": clean_ok,
        "patched_generation_scores_ok": patched_ok,
        "both_generation_scores_ok": both_ok,
        "generation_score_failure_reason_counts": reason_counts,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--max-parsed-fallback-ratio", type=float, default=0.2)
    ap.add_argument("--min-records", type=int, default=1)
    ap.add_argument("--allow-missing-patch-metrics", action="store_true")
    ap.add_argument("--require-score-diagnostics", action="store_true")
    ap.add_argument(
        "--compat-allow-legacy-margin",
        action="store_true",
        help="Allow pre-provenance artifacts that have records but no margin_source fields.",
    )
    args = ap.parse_args(argv)

    by_metric = _load_payloads(Path(args.run_dir))
    if not by_metric and not args.allow_missing_patch_metrics:
        fail("no patch metric payloads found")

    report = {}
    for metric_id, payloads in sorted(by_metric.items()):
        if len(payloads) != 1:
            fail(f"{metric_id} expected exactly one payload, got {len(payloads)}")
        quality = _quality(payloads[0])
        report[metric_id] = quality
        if quality["total"] < args.min_records:
            fail(f"{metric_id} margin records too small: {quality['total']} < {args.min_records}")
        legacy_missing_only = (
            args.compat_allow_legacy_margin
            and quality["continuous"] == 0
            and quality["parsed"] == 0
            and quality["missing"] == quality["total"]
        )
        if quality["continuous"] <= 0 and not legacy_missing_only:
            fail(f"{metric_id} has no continuous margin-source records")
        if args.require_score_diagnostics and quality["diagnostic_records"] <= 0:
            fail(f"{metric_id} missing generation-score diagnostic records")
        if quality["parsed_ratio"] > args.max_parsed_fallback_ratio:
            fail(
                f"{metric_id} parsed fallback ratio too high: "
                f"{quality['parsed_ratio']:.3f} > {args.max_parsed_fallback_ratio:.3f}"
            )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("TRACE MARGIN QUALITY VALIDATION PASSED")


if __name__ == "__main__":
    main()
