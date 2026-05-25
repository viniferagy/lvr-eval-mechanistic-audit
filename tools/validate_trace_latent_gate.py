#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_REQUIRED_METRICS = {
    "pf_a_corruption_selectivity",
    "pf_b_patch_alignment",
    "bf_patch_answer_transfer",
    "bf_swap_latent_replacement",
    "bf_conf_calibrated_progression",
    "cf_stage_decay",
}
PRIMARY_SCALARS = {
    "pf_a_corruption_selectivity": "selectivity",
    "pf_b_patch_alignment": "native_alignment",
    "bf_patch_answer_transfer": "logprob_margin_shift",
    "bf_swap_latent_replacement": "swap_margin_shift",
    "bf_conf_calibrated_progression": "gold_logit_slope",
    "cf_stage_decay": "late_delta",
}
PATCH_METRICS = {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}
TRANSFER_SCALARS = {
    "bf_patch_answer_transfer": "answer_transfer_rate",
    "bf_swap_latent_replacement": "swap_answer_transfer_rate",
}
CONTINUOUS_MARGIN_SOURCES = {"generation_scores", "latent_logit_lens"}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def _load_metric_files(run_dir: Path) -> list[dict]:
    metric_files = sorted((run_dir / "metrics").glob("*.json"))
    if not metric_files:
        fail(f"no metric files under {run_dir}/metrics")
    out = []
    for path in metric_files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["_path"] = str(path)
        out.append(obj)
    return out


def _valid_trace_records(payload: dict) -> int:
    count = 0
    for record in payload.get("samples") or []:
        if record.get("error") is not None:
            continue
        if record.get("trace_quality") == "instrumented_sparse_v0" or record.get("source_trace_quality") == "instrumented_sparse_v0":
            count += 1
    for cell in payload.get("cells") or []:
        for record in cell.get("records") or []:
            if record.get("error") is not None:
                continue
            if record.get("trace_quality") == "instrumented_sparse_v0" or record.get("source_trace_quality") == "instrumented_sparse_v0":
                count += 1
    return count


def _patch_success(payload: dict) -> int:
    total = 0
    for cell in payload.get("cells") or []:
        for record in cell.get("records") or []:
            if record.get("error") is None and int(record.get("n_patch_applied") or 0) >= 1:
                total += 1
    return total


def _patch_margin_quality(payload: dict) -> dict:
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
        "parsed_ratio": (float(parsed) / float(total)) if total else 1.0,
        "sources": sources,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-pairs", type=int, default=50)
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument(
        "--allow-disabled-metrics",
        action="store_true",
        help="Only require metrics enabled in config_snapshot.yaml; used for n=1000 light trace-latent runs.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional explicit trace-latent metric subset to validate, useful for entry smoke runs.",
    )
    ap.add_argument("--max-parsed-fallback-ratio", type=float, default=0.2)
    ap.add_argument(
        "--compat-allow-legacy-margin",
        action="store_true",
        help="Do not require margin_source / transfer CI rows; for validating pre-margin-fix runs.",
    )
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    required_metrics = set(DEFAULT_REQUIRED_METRICS)
    if args.metrics:
        required_metrics = {str(metric) for metric in args.metrics}
    if args.allow_disabled_metrics:
        config_path = run_dir / "config_snapshot.yaml"
        if config_path.is_file():
            import yaml

            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            metrics_cfg = cfg.get("metrics") or {}
            required_metrics = {
                metric for metric in DEFAULT_REQUIRED_METRICS
                if bool((metrics_cfg.get(metric) or {}).get("enabled", True))
            }
    metrics = _load_metric_files(run_dir)
    by_metric: dict[str, list[dict]] = {}
    for envelope in metrics:
        metric_id = envelope.get("metric_id")
        payload = envelope.get("payload") or {}
        by_metric.setdefault(str(metric_id), []).append(payload)

    missing = required_metrics - set(by_metric)
    if missing:
        fail(f"missing trace-latent metrics: {sorted(missing)}")

    for metric_id in sorted(required_metrics):
        payloads = by_metric.get(metric_id) or []
        if len(payloads) != 1:
            fail(f"{metric_id} expected exactly one LVR payload, found {len(payloads)}")
        payload = payloads[0]
        config = payload.get("config") or {}
        reduction = payload.get("reduction") or {}
        scalar = PRIMARY_SCALARS[metric_id]
        print(f"{metric_id}: trace_latent={config.get('trace_latent')} reduction={reduction}")
        if config.get("trace_latent") is not True:
            fail(f"{metric_id} did not run in trace_latent mode")
        if reduction.get(scalar) is None:
            fail(f"{metric_id} missing primary scalar {scalar}")
        trace_records = _valid_trace_records(payload)
        threshold = args.min_pairs if metric_id in PATCH_METRICS else args.min_samples
        if trace_records < threshold:
            fail(f"{metric_id} trace records too small: {trace_records} < {threshold}")
        if metric_id in PATCH_METRICS:
            n_paired = int(payload.get("n_paired") or 0)
            if n_paired < args.min_pairs:
                fail(f"{metric_id} n_paired too small: {n_paired} < {args.min_pairs}")
            if _patch_success(payload) < args.min_pairs:
                fail(f"{metric_id} patch successes too small")
            quality = _patch_margin_quality(payload)
            print(f"{metric_id}: margin_quality={quality}")
            if not args.compat_allow_legacy_margin:
                if quality["total"] < args.min_pairs:
                    fail(f"{metric_id} margin-quality records too small: {quality['total']} < {args.min_pairs}")
                if quality["continuous"] <= 0:
                    fail(f"{metric_id} has no continuous margin-source records")
                if quality["parsed_ratio"] > args.max_parsed_fallback_ratio:
                    fail(
                        f"{metric_id} parsed fallback ratio too high: "
                        f"{quality['parsed_ratio']:.3f} > {args.max_parsed_fallback_ratio:.3f}"
                    )
        if metric_id == "bf_swap_latent_replacement":
            controls = {cell.get("control") for cell in payload.get("control_cells") or []}
            missing_controls = {"self_swap", "reverse_swap", "random_pair_swap"} - controls
            if missing_controls:
                fail(f"bf_swap missing controls: {sorted(missing_controls)}")

    ci_path = run_dir / "summary_with_ci.json"
    if not ci_path.is_file():
        fail("missing summary_with_ci.json")
    ci = json.loads(ci_path.read_text(encoding="utf-8"))
    primary_ci = {
        (row.get("metric_id"), row.get("model"), row.get("scalar"))
        for row in ci
        if int(row.get("n") or 0) > 0
    }
    for metric_id, scalar in PRIMARY_SCALARS.items():
        if metric_id not in required_metrics:
            continue
        if (metric_id, "lvr_7b", scalar) not in primary_ci:
            fail(f"summary_with_ci missing primary row: {metric_id}/lvr_7b/{scalar}")
    if not args.compat_allow_legacy_margin:
        for metric_id, scalar in TRANSFER_SCALARS.items():
            if metric_id not in required_metrics:
                continue
            if (metric_id, "lvr_7b", scalar) not in primary_ci:
                fail(f"summary_with_ci missing transfer row: {metric_id}/lvr_7b/{scalar}")

    sanity_path = run_dir / "sanity" / "summary_sanity.json"
    if not sanity_path.is_file():
        fail("missing sanity/summary_sanity.json")
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    if sanity.get("overall_status") != "pass":
        fail(f"sanity overall_status={sanity.get('overall_status')}")

    plot_dir = run_dir / "metric_plots"
    if not plot_dir.is_dir() or not any(plot_dir.glob("*trace_latent*.png")):
        fail("missing trace-latent visualization PNGs under metric_plots")
    print(f"summary_with_ci rows={len(ci)}")
    print("TRACE LATENT GATE VALIDATION PASSED")


if __name__ == "__main__":
    main()
