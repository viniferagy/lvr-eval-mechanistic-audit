# W3 Latent Intervention Validation Report

Date: 2026-05-23

This report records the Week 3 LVR generation-time latent-state intervention gate and the follow-up SPD-Faith scale-up regression. It is stronger than W2 because it patches real hidden-feedback tensors in the LVR generation loop, but it remains a gate-level result rather than final paper evidence.

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

Latent intervention gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.range.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w3_lvr_latent_patch_n50

./venv/bin/python tools/validate_w3_latent.py \
  runs/w3_lvr_latent_patch_n50 \
  --min-pairs 50
```

SPD scale-up regression:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.spd_faith.week3_scale.yaml \
  --models qwen2_5_vl_3b qwen2_5_vl_7b lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w3_spd_scale_m0_m1_m2_n100

./venv/bin/python tools/validate_spd_range.py \
  runs/w3_spd_scale_m0_m1_m2_n100 \
  --min-pairs 100
```

## Data

Source: `data/spd_faith_hf/manifest.jsonl`

Prepare stats: `data/spd_faith_hf/prepare_stats.json`

Images: `data/spd_faith_hf/images`

The SPD-Faith prepared manifest contains 2996 written records. Region/bbox filtering leaves 2831 usable pairs. Week 3 latent gate sampled 50 pairs; scale-up sampled 100 pairs.

## Latent Patch Gate

Run dir: `runs/w3_lvr_latent_patch_n50`

Config: `config.lvr_latent_patch.range.yaml`

Model: `lvr_7b`

Trace tensor capture policy:

- Capture `output_last_position_hidden_state` from source generation when LVR mode is active.
- Inject captured state into target generation at `forward_pre.last_position_hidden_state`.
- Keep tensors in memory for intervention; write only metadata and reductions to JSON.
- `trace_v2.required=true` and `trace_v2.forbid_fallback=true`.

Primary result:

| field | value |
|---|---:|
| n_paired | 50 |
| n_success | 50 |
| n_error | 0 |
| n_patch_applied | 50 |
| n_with_lvr_mode | 50 |
| n_with_captured_state | 50 |
| latent_answer_transfer_rate | 0.52 |
| sanity overall_status | pass |

Primary CI:

| metric/model/scalar | n | mean | ci_low | ci_high |
|---|---:|---:|---:|---:|
| LVR latent patch/lvr_7b/latent_answer_transfer_rate | 50 | 0.52 | 0.38 | 0.66 |

## SPD Scale-Up Regression

Run dir: `runs/w3_spd_scale_m0_m1_m2_n100`

Config: `config.spd_faith.week3_scale.yaml`

Models: `qwen2_5_vl_3b`, `qwen2_5_vl_7b`, `lvr_7b`

This remains query-span/runnable-v0 regression evidence, not true latent-state proof.

Artifacts:

| artifact | result |
|---|---:|
| metric JSON files | 18 |
| sanity reports | 18 pass, 0 warn, 0 fail |
| summary_with_ci rows | 66 |
| paired BF-Patch n_success/n_error | 100/0 per model |
| paired BF-Swap n_success/n_error | 100/0 per model |

Primary reductions:

| metric | qwen2_5_vl_3b | qwen2_5_vl_7b | lvr_7b |
|---|---:|---:|---:|
| PF-A selectivity | 0.229771 | 0.244202 | 0.225360 |
| PF-B native_alignment | 0.751852 | 0.722500 | 0.819741 |
| BF-Patch logprob_margin_shift | 0.013750 | -0.005000 | -0.012969 |
| BF-Swap swap_margin_shift | 0.006250 | 0.000000 | -0.028047 |
| BF-Conf gold_logit_slope | 0.195249 | 0.347482 | 0.069941 |
| CF-Stage late_delta | 1.809719 | 2.199738 | 1.304567 |

## Notes

- W3 latent patch is the first repository gate that intervenes on real inference-time LVR hidden-feedback state.
- The answer parser is intentionally constrained to SPD-Faith `original` / `modified` prompts.
- `lvr_steps=2` is a gate budget, not a final full-depth latent reasoning sweep.
- Full paper-grade evidence still requires larger sample sizes, multiple latent steps, broader tasks, and layer/step localization.
