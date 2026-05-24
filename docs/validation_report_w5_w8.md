# W5-W8 Validation Report

Date: 2026-05-24

This report records the planned W5-W8 validation gates. It should be filled from run artifacts after each GPU stage.

## Commands

CPU:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python -m py_compile \
  run_all.py smoke_test.py merge_and_analyze.py \
  pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py \
  pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py \
  pipeline/adapters/*.py pipeline/stats/*.py tools/*.py

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python smoke_test.py
```

W5 capacity sweep:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep_s2.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w5_lvr_capacity_s2_n50

bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep_s4.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w5_lvr_capacity_s4_n50

bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w5_lvr_capacity_s8_n50

bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep_s16.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w5_lvr_capacity_s16_n50

./venv/bin/python tools/validate_capacity_sweep.py \
  runs/w5_lvr_capacity_s2_n50 \
  runs/w5_lvr_capacity_s4_n50 \
  runs/w5_lvr_capacity_s8_n50 \
  runs/w5_lvr_capacity_s16_n50 \
  --min-pairs 50 --min-steps 1 \
  --expected-steps 2 4 6 10
```

W6 best-step replication:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.beststep_s8.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w6_lvr_beststep4_s8_n50

./venv/bin/python tools/validate_w3_latent.py \
  runs/w6_lvr_beststep4_s8_n50 \
  --min-pairs 50
```

W7 SPD regression scale-up:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.spd_faith.week7_scale.yaml \
  --models qwen2_5_vl_3b qwen2_5_vl_7b lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w7_spd_scale_m0_m1_m2_n200

./venv/bin/python tools/validate_spd_range.py \
  runs/w7_spd_scale_m0_m1_m2_n200 \
  --min-pairs 200
```

W8 evidence pack:

```bash
./venv/bin/python tools/build_evidence_pack.py
```

Environment and capacity comparison helpers:

```bash
./venv/bin/python tools/check_lvr_env.py \
  --config config.lvr_latent_patch.stepsweep.yaml

./venv/bin/python tools/compare_capacity_sweep.py \
  runs/w5_lvr_capacity_s2_n50 \
  runs/w5_lvr_capacity_s4_n50 \
  runs/w5_lvr_capacity_s8_n50 \
  runs/w5_lvr_capacity_s16_n50
```

## Results

### W5 Capacity Sweep

All four capacity runs passed sanity and validator checks.

| run | n_paired | n_success | n_steps_evaluated | best_step_index | best_step_transfer_rate | step_transfer_auc | last_step_transfer_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `w5_lvr_capacity_s2_n50` | 50 | 50 | 2 | 3 | 0.44 | 0.420000 | 0.44 |
| `w5_lvr_capacity_s4_n50` | 50 | 50 | 4 | 4 | 0.52 | 0.466667 | 0.42 |
| `w5_lvr_capacity_s8_n50` | 50 | 50 | 6 | 4 | 0.44 | 0.424000 | 0.42 |
| `w5_lvr_capacity_s16_n50` | 50 | 50 | 10 | 4 | 0.48 | 0.444444 | 0.46 |

Capacity sweep validator:

```text
CAPACITY SWEEP VALIDATION PASSED
```

Capacity comparison artifacts:

```text
docs/capacity_sweep/capacity_summary.json
docs/capacity_sweep/capacity_summary.md
```

These report adjacent paired-bootstrap differences across capacity budgets. They are descriptive and do not hard-fail on monotonicity.

### W6 Best-Step Replication

Run dir: `runs/w6_lvr_beststep4_s8_n50`

| field | value |
|---|---:|
| n_paired | 50 |
| n_success | 50 |
| n_error | 0 |
| patch step | 4 |
| latent_answer_transfer_rate | 0.60 |
| best_step_transfer_rate | 0.60 |
| step_transfer_auc | 0.60 |
| sanity overall_status | pass |

Validator:

```text
W3 LATENT VALIDATION PASSED
```

### W7 SPD Regression Scale-Up

Run dir: `runs/w7_spd_scale_m0_m1_m2_n200`

Artifacts:

| artifact | result |
|---|---:|
| metric JSON files | 18 |
| sanity reports | 18 pass, 0 warn, 0 fail |
| summary_with_ci rows | 66 |
| paired BF-Patch n_success/n_error | 200/0 per model |
| paired BF-Swap n_success/n_error | 200/0 per model |

Primary reductions:

| metric | qwen2_5_vl_3b | qwen2_5_vl_7b | lvr_7b |
|---|---:|---:|---:|
| PF-A selectivity | 0.211966 | 0.192499 | 0.179774 |
| PF-B native_alignment | 0.715810 | 0.688120 | 0.785818 |
| BF-Patch logprob_margin_shift | -0.002500 | -0.003750 | -0.006797 |
| BF-Swap swap_margin_shift | 0.003125 | -0.012500 | -0.018828 |
| BF-Conf gold_logit_slope | 0.194958 | 0.343618 | 0.069962 |
| CF-Stage late_delta | 1.804084 | 2.215903 | 1.318340 |

Validator:

```text
SPD RANGE VALIDATION PASSED
```

### W8 Evidence Pack

Generated: `docs/evidence_pack_w5_w8.md`

Builder status:

```text
EVIDENCE PACK VALIDATION PASSED
```

The evidence pack summarizes W3-W7 gates and restates the scope boundary: W3-W6 are true inference-time LVR hidden-feedback interventions on constrained SPD-Faith answers; W7 is query-span regression evidence across Qwen/LVR weights.

The builder now hard-fails by default on missing latent metric JSON, missing reductions, non-pass sanity summaries, invalid W5 capacity artifacts, missing W7 metric JSON files, or missing/empty W7 primary CI rows. `--allow-missing` is reserved for local drafting only.
