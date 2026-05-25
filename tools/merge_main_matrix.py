#!/usr/bin/env python
"""Merge sharded W16 main-matrix metric outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.analysis import run_analysis
from pipeline.sanity.report import run_sanity_for_metric_results, save_sanity_reports


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _task_for_run(run_dir: Path) -> str:
    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.is_file():
        return "unknown"
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return str((cfg.get("data") or {}).get("source_type") or "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_metric_results_all(run_root: Path) -> list[dict]:
    out = []
    for path in sorted(run_root.glob("**/metrics/*.json")):
        if "/merged/" in str(path):
            continue
        envelope = _read_json(path)
        if not isinstance(envelope, dict) or not envelope.get("metric_id"):
            continue
        run_dir = path.parent.parent
        task = _task_for_run(run_dir)
        payload = envelope.get("payload") or {}
        if isinstance(payload, dict):
            payload.setdefault("task", task)
            payload.setdefault("source_run_dir", str(run_dir))
            config = payload.setdefault("config", {})
            if isinstance(config, dict):
                config.setdefault("task", task)
        envelope["source_run_dir"] = str(run_dir)
        envelope["task"] = task
        out.append(envelope)
    return out


def _copy_metric_files(metric_results: list[dict], out_dir: Path) -> None:
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for envelope in metric_results:
        metric_id = str(envelope.get("metric_id"))
        model = str(envelope.get("model"))
        task = envelope.get("task") or "merged"
        name = f"{task}_{metric_id}_{model}.json".replace("/", "_")
        (metrics_dir / name).write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--out", default=None, help="Defaults to RUN_ROOT/merged")
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    out_dir = Path(args.out) if args.out else run_root / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_results = _load_metric_results_all(run_root)
    if not metric_results:
        raise SystemExit(f"FAIL: no metric results found under {run_root}")
    _copy_metric_files(metric_results, out_dir)
    if not args.skip_sanity:
        reports = run_sanity_for_metric_results(metric_results, cfg={"validation": {"v2": {"min_samples": 1, "min_pairs": 0, "allow_center_fallback": True}, "output_accuracy": {"min_samples": 1}}})
        save_sanity_reports(reports, str(out_dir))
    run_analysis({}, {}, str(out_dir), metric_results=metric_results)
    (out_dir / "merge_manifest.json").write_text(json.dumps({
        "run_root": str(run_root),
        "n_metric_results": len(metric_results),
        "metrics": sorted({str(row.get("metric_id")) for row in metric_results}),
        "models": sorted({str(row.get("model")) for row in metric_results}),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MERGED_MAIN_MATRIX {out_dir} n_metric_results={len(metric_results)}")


if __name__ == "__main__":
    main()
