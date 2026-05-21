#!/usr/bin/env python
"""
merge_and_analyze.py
====================
sharded 跑完后, 把分散的 bf1_*.json / cf2_*.json 或 metrics/*.json
收集到一个目录, 统一出图 + radar + summary。

  python merge_and_analyze.py --dir runs/collected
"""
from __future__ import annotations

import argparse

from pipeline.analysis import run_analysis
from pipeline.results import load_metric_results, split_metric_results
from pipeline.sanity import has_failed_checks, run_sanity_for_metric_results, save_sanity_reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()
    metric_results = load_metric_results(args.dir, include_legacy=True)
    ablation, decay = split_metric_results(metric_results)
    print(f"loaded BF-1 models: {list(ablation)}")
    print(f"loaded CF-2 models: {list(decay)}")
    if not args.no_sanity:
        reports = run_sanity_for_metric_results(metric_results, cfg={})
        sanity_dir = save_sanity_reports(reports, args.dir)
        status = "FAILED" if has_failed_checks(reports) else "ok"
        print(f"sanity reports: {sanity_dir} ({status})")
    run_analysis(ablation, decay, args.dir, metric_results=metric_results)


if __name__ == "__main__":
    main()
