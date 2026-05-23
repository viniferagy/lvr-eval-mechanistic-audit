# W4 Latent Step Localization Validation Report

Date: 2026-05-24

This report records the Week 4 LVR generation-time latent step-localization gate. W4 builds on W3 by patching each captured hidden-feedback step separately. It remains a localization gate, not a final paper-scale result.

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

Latent step sweep:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w4_lvr_latent_stepsweep_n50_s8

./venv/bin/python tools/validate_w4_stepsweep.py \
  runs/w4_lvr_latent_stepsweep_n50_s8 \
  --min-pairs 50 \
  --min-steps 2
```

## Data

Source: `data/spd_faith_hf/manifest.jsonl`

Prepare stats: `data/spd_faith_hf/prepare_stats.json`

Images: `data/spd_faith_hf/images`

The gate samples 50 paired SPD-Faith examples with region/bbox metadata.

## Result

Run dir: `runs/w4_lvr_latent_stepsweep_n50_s8`

Config: `config.lvr_latent_patch.stepsweep.yaml`

Model: `lvr_7b`

Trace tensor capture policy:

- Capture `output_last_position_hidden_state` for each source LVR hidden-feedback step.
- Inject one captured state at the matching target `forward_pre.last_position_hidden_state` step.
- Keep tensors in memory; write only metadata, per-step answer results, and reductions.
- `trace_v2.required=true` and `trace_v2.forbid_fallback=true`.

Primary reductions:

| field | value |
|---|---:|
| n_paired | 50 |
| n_success | 50 |
| n_error | 0 |
| n_patch_applied | 50 |
| n_with_lvr_mode | 50 |
| n_with_captured_state | 50 |
| n_steps_evaluated | 6 |
| best_step_index | 4 |
| best_step_transfer_rate | 0.44 |
| step_transfer_auc | 0.424000 |
| last_step_transfer_rate | 0.42 |
| sanity overall_status | pass |

Per-step transfer rates:

| step_index | n | transfer_rate |
|---:|---:|---:|
| 2 | 50 | 0.42 |
| 3 | 50 | 0.40 |
| 4 | 50 | 0.44 |
| 5 | 50 | 0.42 |
| 6 | 50 | 0.44 |
| 7 | 50 | 0.42 |

Primary CI:

| metric/model/scalar | n | mean | ci_low | ci_high |
|---|---:|---:|---:|---:|
| LVR latent step sweep/lvr_7b/best_step_transfer_rate | 50 | 0.46 | 0.32 | 0.60 |
| LVR latent step sweep/lvr_7b/step_transfer_auc | 50 | 0.424000 | 0.292000 | 0.564000 |
| LVR latent step sweep/lvr_7b/last_step_transfer_rate | 50 | 0.42 | 0.28 | 0.56 |

## Notes

- W4 is the first step-localization gate for true inference-time LVR hidden-feedback intervention.
- `lvr_steps=8` is a localization budget, not a full-depth sweep.
- Generated answer parsing remains constrained to SPD-Faith `original` / `modified` prompts.
- Layer-level localization and broader-task replication remain future work.
