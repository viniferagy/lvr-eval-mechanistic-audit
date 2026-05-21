# Validation Report

Date: 2026-05-21

## Scope

This repository contains the LVR-Eval source code, configuration, documentation,
and GPU reservation utilities. Large local model weights and historical run
outputs are intentionally excluded from git and must be restored under
`models/` before running full GPU evaluations.

## Static And Smoke Checks

The source project passed:

```bash
../.venv/bin/python -m py_compile run_all.py smoke_test.py merge_and_analyze.py pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py pipeline/adapters/*.py
../.venv/bin/python smoke_test.py
```

Smoke coverage includes:

- corruption operators
- BF-3/PF-3 curve reductions
- CF-2 curve features
- JSONL `data.field_map`
- sanity report generation
- unified `MetricResult` analysis output

## Full Server Run

The full 4-GPU sharded run was executed through the GPU reservation wrapper:

```bash
bash /home/pengguangyue/workspace/proj/run_and_hold.sh 0,1,2,3 launch_sharded.sh
```

Run root:

```text
runs/sharded_20260521_154014
```

Completed jobs:

- `qwen2_5_vl_7b` / `bf1_latent_ablation`
- `qwen2_5_vl_7b` / `cf2_pf_decay_curve`
- `lvr_7b` / `bf1_latent_ablation`
- `lvr_7b` / `cf2_pf_decay_curve`

The sharded command exited with status `0`.

## Merge And Analysis

The merged analysis was executed through the same GPU reservation wrapper:

```bash
bash /home/pengguangyue/workspace/proj/run_and_hold.sh 0,1,2,3 ../.venv/bin/python merge_and_analyze.py --dir runs/sharded_20260521_154014
```

Expected outputs were produced:

- `summary.json`
- `rank_correlation.json`
- `radar_4metric.png`
- `bf1_layerwise_bf3.png`
- `bf1_layerwise_pf3.png`
- `cf2_decay_mask.png`
- `cf2_decay_gaussian_blur.png`
- `metric_results_summary.json`
- `metric_plots/*.png`

## Sanity Status

Merged sanity summary:

- total reports: 8
- passed reports: 6
- failed reports: 2

The two failures are BF-3 baseline monotonicity checks:

- `lvr_7b`: monotonicity Spearman `-0.2167`
- `qwen2_5_vl_7b`: monotonicity Spearman `0.2972`

All structural checks passed. BF-1, PF-3, and CF-2 sanity reports passed.
The BF-3 failures are numerical sanity flags on the monotonicity heuristic, not
runtime or artifact-generation failures.

## GPU Holder

`tools/run_and_hold.sh` launches commands through a GPU reservation wrapper.
`tools/hold_gpu.py` now auto-expands ballast allocation periodically, so when
other processes release memory, the holder can claim newly available GPU memory
on the next expansion cycle.

