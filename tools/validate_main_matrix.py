#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_METRICS = {
    "pf_a_corruption_selectivity",
    "pf_b_patch_alignment",
    "bf_patch_answer_transfer",
    "bf_swap_latent_replacement",
    "bf_conf_calibrated_progression",
    "cf_stage_decay",
}
NON_PAIRED_METRICS = {
    "pf_a_corruption_selectivity",
    "pf_b_patch_alignment",
    "bf_conf_calibrated_progression",
    "cf_stage_decay",
}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _task_for_run(run_dir: Path) -> str:
    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.is_file():
        return "unknown"
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return str((cfg.get("data") or {}).get("source_type") or "unknown")


def _load_envelopes(run_root: Path) -> list[dict]:
    out = []
    for path in sorted(run_root.glob("**/metrics/*.json")):
        if "/merged/" in str(path):
            continue
        env = _read_json(path)
        if not isinstance(env, dict) or not env.get("metric_id"):
            continue
        run_dir = path.parent.parent
        env["task"] = _task_for_run(run_dir)
        env["source_run_dir"] = str(run_dir)
        out.append(env)
    return out


def _valid_n(payload: dict) -> int:
    reduction = payload.get("reduction") or {}
    for key in ("n", "n_paired"):
        if reduction.get(key) is not None:
            return int(reduction.get(key) or 0)
    if payload.get("n_paired") is not None:
        return int(payload.get("n_paired") or 0)
    samples = payload.get("samples") or []
    if samples:
        return sum(1 for row in samples if row.get("error") is None)
    return 0


def _ci_rows(run_root: Path) -> list[dict]:
    candidates = [
        run_root / "summary_with_ci.json",
        run_root / "merged" / "summary_with_ci.json",
    ]
    for path in candidates:
        if path.is_file():
            data = _read_json(path)
            return data if isinstance(data, list) else []
    return []


def validate_accuracy(args, envelopes: list[dict]) -> None:
    rows = [env for env in envelopes if env.get("metric_id") == "output_accuracy_sanity"]
    if not rows:
        fail("missing output_accuracy_sanity rows")
    for env in rows:
        payload = env.get("payload") or {}
        n = _valid_n(payload)
        if n < args.min_samples:
            fail(f"{env.get('model')} accuracy samples too small: {n} < {args.min_samples}")
        acc = (payload.get("reduction") or {}).get("accuracy")
        if acc is None:
            fail(f"{env.get('model')} missing accuracy")
    print(f"ACCURACY MATRIX VALIDATION PASSED rows={len(rows)}")


def validate_main(args, run_root: Path, envelopes: list[dict]) -> None:
    by = {}
    for env in envelopes:
        by[(env.get("task"), env.get("model"), env.get("metric_id"))] = env
    models = [item for item in args.models.split(",") if item]
    tasks = [item for item in args.tasks.split(",") if item]
    missing = []
    for task in tasks:
        expected_metrics = PRIMARY_METRICS if task == "spd_faith" else NON_PAIRED_METRICS
        for model in models:
            for metric in expected_metrics:
                if (task, model, metric) not in by:
                    missing.append((task, model, metric))
    if missing:
        fail(f"missing task/model/metric rows: {missing[:10]}")
    for (task, model, metric), env in by.items():
        if task not in tasks or model not in models:
            continue
        payload = env.get("payload") or {}
        n = _valid_n(payload)
        threshold = args.bf_min_pairs if metric in {"bf_patch_answer_transfer", "bf_swap_latent_replacement"} else args.min_samples
        if n < threshold:
            fail(f"{task}/{model}/{metric} n too small: {n} < {threshold}")
    ci = _ci_rows(run_root)
    if not ci:
        fail("missing summary_with_ci.json; run tools/merge_main_matrix.py first")
    print(f"MAIN MATRIX VALIDATION PASSED rows={len(envelopes)} ci_rows={len(ci)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--tasks", default="maze,spd_faith,blink")
    ap.add_argument("--models", default="qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b")
    ap.add_argument("--min-samples", type=int, default=800)
    ap.add_argument("--bf-min-pairs", type=int, default=300)
    ap.add_argument("--accuracy-only", action="store_true")
    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    envelopes = _load_envelopes(run_root)
    if not envelopes:
        fail(f"no metric envelopes found under {run_root}")
    if args.accuracy_only:
        validate_accuracy(args, envelopes)
    else:
        validate_main(args, run_root, envelopes)


if __name__ == "__main__":
    main()
