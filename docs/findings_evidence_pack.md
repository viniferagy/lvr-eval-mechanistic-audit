# Findings Evidence Pack

This pack is generated from local run artifacts. It is intended as the Findings-level review gate, not a Main/Spotlight claim-finalizing package.

## Gate Status

| component | status | run dir |
|---|---|---|
| W3 true latent patch | pass | `runs/w3_lvr_latent_patch_n50` |
| W4 latent step sweep | pass | `runs/w4_lvr_latent_stepsweep_n50_s8` |
| W6 best-step replication | pass | `runs/w6_lvr_beststep4_s8_n50` |
| W7 SPD regression matrix | pass | `runs/w7_spd_scale_m0_m1_m2_n200` |
| W9 Maze Findings subset | pass | `runs/w9_maze_findings_m0_m1_m2_n200_bbox` |

## True LVR Hidden-Feedback Gates

| gate | key reductions |
|---|---|
| W3 last-step | `{"latent_answer_transfer_rate": 0.52, "n_paired": 50, "n_success": 50}` |
| W4 step sweep | `{"best_step_index": 4, "best_step_transfer_rate": 0.44, "n_steps_evaluated": 6, "step_transfer_auc": 0.42400000000000004}` |
| W6 best-step | `{"best_step_index": 4, "latent_answer_transfer_rate": 0.6, "n_paired": 50, "n_success": 50}` |

## SPD-Faith Findings Matrix

| metric | scalar | qwen2_5_vl_3b | qwen2_5_vl_7b | lvr_7b |
|---|---|---:|---:|---:|
| `pf_a_corruption_selectivity` | `selectivity` | 0.211966 | 0.192499 | 0.179774 |
| `pf_b_patch_alignment` | `native_alignment` | 0.71581 | 0.68812 | 0.785818 |
| `bf_patch_answer_transfer` | `logprob_margin_shift` | -0.0025 | -0.00374999 | -0.00679687 |
| `bf_swap_latent_replacement` | `swap_margin_shift` | 0.003125 | -0.0125 | -0.0188281 |
| `bf_conf_calibrated_progression` | `gold_logit_slope` | 0.194958 | 0.343618 | 0.0699619 |
| `cf_stage_decay` | `late_delta` | 1.80408 | 2.2159 | 1.31834 |

## Maze Findings Matrix

| metric | scalar | qwen2_5_vl_3b | qwen2_5_vl_7b | lvr_7b |
|---|---|---:|---:|---:|
| `pf_a_corruption_selectivity` | `selectivity` | -0.654818 | -0.8843 | -0.626778 |
| `pf_b_patch_alignment` | `native_alignment` | 0.476241 | 0.391128 | 0.462267 |
| `bf_conf_calibrated_progression` | `gold_logit_slope` | 0.189739 | 0.277957 | 0.0306987 |
| `cf_stage_decay` | `late_delta` | 1.48642 | 2.00021 | 1.51728 |

## Boundary Statement

Findings-level claims should be limited to a reproducible causal audit toolkit, SPD-Faith paired evidence, and true LVR hidden-feedback patch gates on constrained answers. Main/Spotlight claims still require broader tasks, larger n, additional LVR paradigms, and layer/position localization.
