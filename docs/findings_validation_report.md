# Findings Gate Validation Report

Date: 2026-05-24

This report defines the Findings-level gate after W8. It does not relabel W8 as paper-ready. The gate requires:

- W3 true LVR hidden-feedback last-step patch, `n>=50`.
- W4 true LVR hidden-feedback step sweep, `n>=50`.
- W6 best-step replication, `n>=50`.
- SPD-Faith regression matrix across `qwen2_5_vl_3b`, `qwen2_5_vl_7b`, and `lvr_7b`.
- MazePlanning subset across the same three models.

## Local State

The local repo has validated W3-W8/SPD artifacts under `runs/` and a real MazePlanning artifact at `runs/w9_maze_findings_m0_m1_m2_n200_bbox`. The Findings gate is therefore runnable and strict-pass locally.

The canonical Maze input is the public Latent Sketchpad test set, `huanyu112/MazePlanning-Test`. Convert it into the local `source_type=maze` manifest with:

```bash
./venv/bin/python tools/prepare_maze_planning_hf.py \
  --out data/maze_planning
```

Expected local outputs:

```text
data/maze_planning/manifest.jsonl
data/maze_planning/prepare_stats.json
data/maze_planning/images/
```

## Commands

CPU checks:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python -m py_compile \
  run_all.py smoke_test.py merge_and_analyze.py \
  pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py \
  pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py \
  pipeline/adapters/*.py pipeline/stats/*.py tools/*.py

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python smoke_test.py
```

Findings SPD scale-up:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.findings_spd_scale.yaml \
  --models qwen2_5_vl_3b qwen2_5_vl_7b lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w9_spd_findings_m0_m1_m2_n500
```

Findings Maze subset:

```bash
./venv/bin/python tools/prepare_maze_planning_hf.py \
  --out data/maze_planning

bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.maze.findings.yaml \
  --models qwen2_5_vl_3b qwen2_5_vl_7b lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w9_maze_findings_m0_m1_m2_n200
```

Gate validation:

```bash
./venv/bin/python tools/validate_findings_gate.py
```

Blocked-mode planning pack:

```bash
./venv/bin/python tools/build_findings_pack.py --allow-missing-maze
```

## Recorded Maze Artifact

```text
run_dir: runs/w9_maze_findings_m0_m1_m2_n200_bbox
samples: 200
models: qwen2_5_vl_3b, qwen2_5_vl_7b, lvr_7b
metrics: pf_a_corruption_selectivity, pf_b_patch_alignment, bf_conf_calibrated_progression, cf_stage_decay
sanity: 12/12 pass, 0 warn, 0 fail
ci_rows: 51
oracle: bbox generated from non-white MazePlanning image extent
```

Primary reductions:

| metric | scalar | qwen2_5_vl_3b | qwen2_5_vl_7b | lvr_7b |
|---|---|---:|---:|---:|
| `pf_a_corruption_selectivity` | `selectivity` | -0.654818 | -0.884300 | -0.626778 |
| `pf_b_patch_alignment` | `native_alignment` | 0.476241 | 0.391128 | 0.462267 |
| `bf_conf_calibrated_progression` | `gold_logit_slope` | 0.189739 | 0.277957 | 0.030699 |
| `cf_stage_decay` | `late_delta` | 1.486417 | 2.000213 | 1.517284 |

## Acceptance

`tools/validate_findings_gate.py` must print:

```text
FINDINGS GATE VALIDATION PASSED
```

The validator intentionally fails when the Maze run is absent. This prevents a SPD-only artifact set from being described as Findings-ready.

## Boundary

Findings-level claims should be limited to a reproducible causal audit toolkit, SPD-Faith paired evidence, and true LVR hidden-feedback patch gates on constrained answers. Main/Spotlight claims still require broader tasks, larger `n`, additional LVR paradigms, mixed-effects analysis, and layer/position localization.
