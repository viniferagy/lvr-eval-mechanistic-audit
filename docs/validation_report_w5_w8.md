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
  --min-pairs 50 --min-steps 1
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
./venv/bin/python tools/build_evidence_pack.py \
  --w5 runs/w5_lvr_capacity_s2_n50 runs/w5_lvr_capacity_s4_n50 \
       runs/w5_lvr_capacity_s8_n50 runs/w5_lvr_capacity_s16_n50 \
  --w6 runs/w6_lvr_beststep4_s8_n50 \
  --w7 runs/w7_spd_scale_m0_m1_m2_n200
```

## Results

Results will be populated from the W5-W8 run artifacts after execution.
