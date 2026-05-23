# W5-W8 Evidence Pack

This file is generated from local run artifacts. It summarizes gate-level evidence, not final paper-scale claims.

## Latent Intervention Gates

| gate | run_dir | sanity | key reductions |
|---|---|---|---|
| W3 last-step | `runs/w3_lvr_latent_patch_n50` | pass | `{"latent_answer_transfer_rate": 0.52, "n_paired": 50, "n_success": 50}` |
| W4 step sweep | `runs/w4_lvr_latent_stepsweep_n50_s8` | pass | `{"best_step_index": 4, "best_step_transfer_rate": 0.44, "last_step_transfer_rate": 0.42, "latent_answer_transfer_rate": 0.42, "n_paired": 50, "n_steps_evaluated": 6, "n_success": 50, "step_transfer_auc": 0.42400000000000004}` |
| W6 best-step | `runs/w6_lvr_beststep4_s8_n50` | pass | `{"best_step_index": 4, "best_step_transfer_rate": 0.6, "last_step_transfer_rate": 0.6, "latent_answer_transfer_rate": 0.6, "n_paired": 50, "n_steps_evaluated": 1, "n_success": 50, "step_transfer_auc": 0.6}` |

## W5 Capacity Sweep

| run_dir | sanity | key reductions |
|---|---|---|
| `runs/w5_lvr_capacity_s2_n50` | pass | `{"best_step_index": 3, "best_step_transfer_rate": 0.44, "last_step_transfer_rate": 0.44, "n_paired": 50, "n_steps_evaluated": 2, "step_transfer_auc": 0.42000000000000004}` |
| `runs/w5_lvr_capacity_s4_n50` | pass | `{"best_step_index": 4, "best_step_transfer_rate": 0.52, "last_step_transfer_rate": 0.42, "n_paired": 50, "n_steps_evaluated": 4, "step_transfer_auc": 0.4666666666666666}` |
| `runs/w5_lvr_capacity_s8_n50` | pass | `{"best_step_index": 4, "best_step_transfer_rate": 0.44, "last_step_transfer_rate": 0.42, "n_paired": 50, "n_steps_evaluated": 6, "step_transfer_auc": 0.42400000000000004}` |
| `runs/w5_lvr_capacity_s16_n50` | pass | `{"best_step_index": 4, "best_step_transfer_rate": 0.48, "last_step_transfer_rate": 0.46, "n_paired": 50, "n_steps_evaluated": 10, "step_transfer_auc": 0.4444444444444444}` |

## W7 SPD Regression Scale-Up

- Run dir: `runs/w7_spd_scale_m0_m1_m2_n200`
- Sanity: `pass`
- CI rows: `66`

## Boundary Statement

W3-W6 are true inference-time LVR hidden-feedback intervention gates on SPD-Faith constrained answers. W7 remains query-span regression evidence across Qwen/LVR weights. Broader-task and layer-level localization remain future work.
