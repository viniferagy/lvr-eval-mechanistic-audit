# W18 Full Local Main Matrix Results

Generated from real local artifacts on 2026-05-26.

Run root: `runs/w18_main_matrix_full_local`

Merged artifacts: `runs/w18_main_matrix_full_local/merged`

Main matrix validation:

```text
tools/validate_main_matrix.py runs/w18_main_matrix_full_local --tasks maze,spd_faith,blink --min-samples 450 --bf-min-pairs 300
MAIN MATRIX VALIDATION PASSED rows=42 ci_rows=180

tools/validate_main_paper_readiness.py runs/w18_main_matrix_full_local/merged --mode full_main_matrix
MAIN MATRIX READINESS VALIDATION PASSED
```

## Scope And Boundaries

W18 completed the local thousand-scale main matrix where staged data permitted it: SPD-Faith and BLINK ran at `n=1000`; Maze ran the full local/upstream test set at `n=500`; SPD paired BF metrics used the preregistered paired subset cap `n=500`. BLINK has no bbox metadata locally, so PF-A/PF-B are weak-oracle center-fallback diagnostics. V*Bench/VStar and VSI-Bench are not reported as model results because their local manifests contain zero usable visual samples.

This matrix is a standard-forward/query-span cross-model matrix for Qwen and LVR. The real LVR generation-trace latent scale gate is reported separately in the trace-latent section below.

## Data Staging

| Dataset | Usable local n | W18 status | Boundary |
|---|---|---|---|
| T2 SPD-Faith | 2996 | n=1000 run; BF paired subset n=500 | paired bbox/region oracle |
| T1 Maze | 500 | full local set n=500; data-limited below 800 | no paired BF metrics |
| T3 BLINK | 1000 | n=1000 run; weak_oracle=1000 | center fallback, no strong bbox claim |
| V*Bench/VStar | 0 | blocked: missing images=191, missing bbox=191 | code path present, not a completed visual run |
| T4 VSI-Bench | 0 | blocked: skipped_no_answer=100 | needs frame-grid/images and answer mapping |

## Coverage

| Task | Models | Primary metrics | CI n values | Validation |
|---|---|---|---|---|
| T2 SPD-Faith | Qwen2.5-VL-3B, Qwen2.5-VL-7B, LVR-7B | 6 | 500, 1000 | pass |
| T1 Maze | Qwen2.5-VL-3B, Qwen2.5-VL-7B, LVR-7B | 4 | 500 | pass |
| T3 BLINK | Qwen2.5-VL-3B, Qwen2.5-VL-7B, LVR-7B | 4 | 1000 | pass |

## Combined Line Plot

![W18 combined primary metric lines](figures/w18_full_local_matrix/w18_primary_metric_lines_combined.png)

## Primary Results

### PF-A selectivity

| Task | Qwen2.5-VL-3B | Qwen2.5-VL-7B | LVR-7B |
|---|---|---|---|
| T2 SPD-Faith | 0.2105 [0.1893, 0.2309] | 0.2161 [0.1966, 0.2348] | 0.1962 [0.1807, 0.2106] |
| T1 Maze | -0.6553 [-0.6605, -0.6498] | -0.8876 [-0.8975, -0.8778] | -0.6335 [-0.6432, -0.6242] |
| T3 BLINK | 0.2279 [0.2124, 0.2431] | 0.1960 [0.1809, 0.2108] | 0.2927 [0.2786, 0.3070] |

![pf_a_corruption_selectivity](figures/w18_full_local_matrix/w18_line_pf_a_corruption_selectivity.png)

### PF-B native alignment

| Task | Qwen2.5-VL-3B | Qwen2.5-VL-7B | LVR-7B |
|---|---|---|---|
| T2 SPD-Faith | 0.7174 [0.7071, 0.7280] | 0.6987 [0.6883, 0.7092] | 0.7964 [0.7880, 0.8047] |
| T1 Maze | 0.4752 [0.4739, 0.4764] | 0.3912 [0.3899, 0.3924] | 0.4622 [0.4604, 0.4641] |
| T3 BLINK | 0.5746 [0.5691, 0.5802] | 0.5411 [0.5361, 0.5464] | 0.6203 [0.6169, 0.6238] |

![pf_b_patch_alignment](figures/w18_full_local_matrix/w18_line_pf_b_patch_alignment.png)

### BF-Patch continuous margin

| Task | Qwen2.5-VL-3B | Qwen2.5-VL-7B | LVR-7B |
|---|---|---|---|
| T2 SPD-Faith | -0.0025 [-0.0100, 0.0050] | -0.0028 [-0.0130, 0.0075] | -0.0047 [-0.0190, 0.0091] |

![bf_patch_answer_transfer](figures/w18_full_local_matrix/w18_line_bf_patch_answer_transfer.png)

### BF-Swap continuous margin

| Task | Qwen2.5-VL-3B | Qwen2.5-VL-7B | LVR-7B |
|---|---|---|---|
| T2 SPD-Faith | -0.0050 [-0.0125, 0.0028] | -0.0052 [-0.0153, 0.0053] | -0.0229 [-0.0382, -0.0077] |

![bf_swap_latent_replacement](figures/w18_full_local_matrix/w18_line_bf_swap_latent_replacement.png)

### BF-Conf gold-logit slope

| Task | Qwen2.5-VL-3B | Qwen2.5-VL-7B | LVR-7B |
|---|---|---|---|
| T2 SPD-Faith | 0.1939 [0.1927, 0.1951] | 0.3469 [0.3446, 0.3493] | 0.0707 [0.0698, 0.0716] |
| T1 Maze | 0.1903 [0.1896, 0.1910] | 0.2787 [0.2774, 0.2800] | 0.0305 [0.0303, 0.0307] |
| T3 BLINK | -0.0771 [-0.0789, -0.0754] | 0.0760 [0.0728, 0.0793] | 0.0318 [0.0287, 0.0348] |

![bf_conf_calibrated_progression](figures/w18_full_local_matrix/w18_line_bf_conf_calibrated_progression.png)

### CF-Stage late delta

| Task | Qwen2.5-VL-3B | Qwen2.5-VL-7B | LVR-7B |
|---|---|---|---|
| T2 SPD-Faith | 1.7846 [1.7655, 1.8031] | 2.1856 [2.1650, 2.2056] | 1.3096 [1.2934, 1.3261] |
| T1 Maze | 1.5133 [1.4894, 1.5375] | 1.9995 [1.9787, 2.0197] | 1.4992 [1.4818, 1.5174] |
| T3 BLINK | 1.6062 [1.5850, 1.6263] | 1.8960 [1.8745, 1.9167] | 1.6506 [1.6305, 1.6717] |

![cf_stage_decay](figures/w18_full_local_matrix/w18_line_cf_stage_decay.png)


## SPD Answer-Transfer Rates

These rows are secondary binary transfer readouts for the SPD paired BF metrics. The paper-facing continuous BF scalar is `continuous_margin_shift`; transfer rate is reported as an interpretable companion.

| Metric | Model | Transfer rate [95% CI] |
|---|---|---|
| BF-Patch continuous margin | Qwen2.5-VL-3B | 0.7760 [0.7380, 0.8120] |
| BF-Patch continuous margin | Qwen2.5-VL-7B | 0.2700 [0.2320, 0.3100] |
| BF-Patch continuous margin | LVR-7B | 0.4300 [0.3860, 0.4740] |
| BF-Swap continuous margin | Qwen2.5-VL-3B | 0.7800 [0.7440, 0.8160] |
| BF-Swap continuous margin | Qwen2.5-VL-7B | 0.2640 [0.2260, 0.3040] |
| BF-Swap continuous margin | LVR-7B | 0.4180 [0.3740, 0.4640] |

## LVR Real Trace-Latent n=1000 Light Gate

This is a separate evidence layer from W18. It uses `runs/w17_lvr_trace_latent_spd_n1000_light/merged`, `audit.mode=generation_trace`, and `trace_latent.enabled=true`; BF-Patch/BF-Swap are intentionally disabled in the light config.

Validator:

```text
tools/validate_trace_latent_gate.py runs/w17_lvr_trace_latent_spd_n1000_light/merged --allow-disabled-metrics --metrics pf_a_corruption_selectivity pf_b_patch_alignment bf_conf_calibrated_progression cf_stage_decay --min-samples 800
TRACE LATENT GATE VALIDATION PASSED
```

| Metric | Scalar | n | Mean [95% CI] |
|---|---|---|---|
| BF-Conf gold-logit slope | gold_logit_slope | 1000 | 0.1400 [0.1287, 0.1517] |
| CF-Stage late delta | late_delta | 1000 | -3.8448 [-4.1683, -3.5254] |
| PF-A selectivity | selectivity | 1000 | -0.0553 [-0.0629, -0.0477] |
| PF-B native alignment | native_alignment | 1000 | 0.8355 [0.8305, 0.8403] |

## Mixed-Effects/Fallback Regression

Current artifact uses the preregistered dependency-light fallback `value ~ model + task`, not a true random-effects MixedLM. This is a readiness artifact and should be upgraded to sample-level mixed effects before final Main submission.

| Metric | Method | Rows | R2 |
|---|---|---|---|
| BF-Conf gold-logit slope | ols_fixed_effects_fallback | 9 | 0.8230 |
| BF-Patch continuous margin | ols_fixed_effects_fallback | 3 | 1.0000 |
| BF-Swap continuous margin | ols_fixed_effects_fallback | 3 | 1.0000 |
| CF-Stage late delta | ols_fixed_effects_fallback | 9 | 0.7902 |
| PF-A selectivity | ols_fixed_effects_fallback | 9 | 0.9867 |
| PF-B native alignment | ols_fixed_effects_fallback | 9 | 0.9836 |

## Reproduction Commands

```bash
PYTHON_BIN=./venv/bin/python MPLCONFIGDIR=/tmp/matplotlib-lvr-eval HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.main_spd_n1000.yaml,config.main_maze_full500.yaml,config.main_blink_n1000.yaml \
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \
  --metrics all \
  --gpus 0,1,2,3 \
  --run-root runs/w18_main_matrix_full_local

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python tools/merge_main_matrix.py runs/w18_main_matrix_full_local
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python tools/summarize_w18_results.py
```
