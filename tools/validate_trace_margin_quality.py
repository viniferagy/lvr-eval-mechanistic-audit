#!/usr/bin/env python
"""Validate trace-latent patch margin provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PATCH_METRICS = {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}
CONTINUOUS_MARGIN_SOURCES = {"generation_scores", "latent_logit_lens"}


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
    for cell in payload.get("cells") or []:
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
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--max-parsed-fallback-ratio", type=float, default=0.2)
    ap.add_argument("--min-records", type=int, default=1)
    ap.add_argument("--allow-missing-patch-metrics", action="store_true")
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
        if quality["continuous"] <= 0:
            fail(f"{metric_id} has no continuous margin-source records")
        if quality["parsed_ratio"] > args.max_parsed_fallback_ratio:
            fail(
                f"{metric_id} parsed fallback ratio too high: "
                f"{quality['parsed_ratio']:.3f} > {args.max_parsed_fallback_ratio:.3f}"
            )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("TRACE MARGIN QUALITY VALIDATION PASSED")


if __name__ == "__main__":
    main()
