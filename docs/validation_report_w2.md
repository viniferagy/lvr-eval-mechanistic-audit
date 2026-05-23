# W2 Final Gate Validation Report

Date: 2026-05-23

This report records the reviewer-gated W2-final cleanup validation. It is a runnable-v0 gate result, not a paper-grade causal conclusion.

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

SPD-Faith prepare smoke:

```bash
./venv/bin/python tools/prepare_spd_faith_hf.py \
  --max-per-split 2 \
  --out /tmp/spd_faith_prepare_smoke
```

Trace v2 gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.trace_v2.yaml \
  --models lvr_7b \
  --only lvr_generation_trace \
  --device cuda:0 \
  --run-name w2_final_trace_v2_lvr_n3
```

SPD-Faith range gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.spd_faith.range.yaml \
  --models qwen2_5_vl_3b qwen2_5_vl_7b lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w2_final_spd_range_m0_m1_m2_n50
```

Post-run validator:

```bash
./venv/bin/python tools/validate_spd_range.py \
  runs/w2_final_spd_range_m0_m1_m2_n50 \
  --min-pairs 50
```

## Data

Public source: `Jackson-Lv/SPD-Faith-Bench`

Local prepared source: `data/spd_faith_hf/manifest.jsonl`

Prepare stats: `data/spd_faith_hf/prepare_stats.json`

Images: `data/spd_faith_hf/images`

`prepare_stats.json`: `n_written=2996`, answer policy `clean=original, counterfactual=modified`.

The final range config requires a real region oracle (`require_region_or_bbox: true`): 2831 usable pairs, 165 records skipped for missing region/bbox annotation, 50 sampled pairs.

The prepare smoke wrote 8 records under `/tmp/spd_faith_prepare_smoke`, including clean/counterfactual image files, answers, bboxes, and `prepare_stats.json`.

## Trace Gate

Run dir: `runs/w2_final_trace_v2_lvr_n3`

Config: `config.trace_v2.yaml`

Model: `lvr_7b`

Trace hard requirements:

```yaml
trace_v2:
  required: true
  forbid_fallback: true
```

Result:

| field | value |
|---|---:|
| n_samples | 3 |
| n_instrumented | 3 |
| n_with_lvr_mode | 3 |
| n_with_hidden_feedback | 3 |
| instrumented_rate | 1.0 |
| sanity overall_status | pass |

Per-sample trace records had `trace_quality=instrumented_sparse_v0`, `missing_modules=[]`, `lm_head_called=true`, and no `trace_v2_error`.

## SPD Range Gate

Run dir: `runs/w2_final_spd_range_m0_m1_m2_n50`

Config: `config.spd_faith.range.yaml`

Models: `qwen2_5_vl_3b`, `qwen2_5_vl_7b`, `lvr_7b`

This is a paired SPD query-span intervention range gate. The `lvr_7b` rows are not true inference-time latent-state intervention evidence.

Artifacts:

| artifact | result |
|---|---:|
| metric JSON files | 18 |
| sanity reports | 18 pass, 0 warn, 0 fail |
| summary_with_ci rows | 66 |
| paired BF-Patch n_success/n_error | 50/0 per model |
| paired BF-Swap n_success/n_error | 50/0 per model |
| BF-Swap controls | self-swap, reverse-swap, random-pair swap |

Primary reductions:

| metric | qwen2_5_vl_3b | qwen2_5_vl_7b | lvr_7b |
|---|---:|---:|---:|
| PF-A selectivity | 0.256129 | 0.247687 | 0.250040 |
| PF-B native_alignment | 0.740718 | 0.702077 | 0.806338 |
| BF-Patch logprob_margin_shift | 0.015000 | 0.010000 | -0.048828 |
| BF-Swap swap_margin_shift | 0.010000 | -0.002500 | -0.034141 |
| BF-Conf gold_logit_slope | 0.197275 | 0.344212 | 0.068481 |
| CF-Stage late_delta | 1.829121 | 2.253331 | 1.307643 |

BF-Swap control reductions:

| control | qwen2_5_vl_3b | qwen2_5_vl_7b | lvr_7b |
|---|---:|---:|---:|
| self_swap | 0.000000 | 0.000000 | 0.000000 |
| reverse_swap | 0.007500 | -0.010000 | 0.021953 |
| random_pair_swap | 0.012500 | -0.015000 | -0.016094 |

Primary CI rows (`summary_with_ci.json`, grouped by `paired_id` fallback `id`, 5000 bootstrap resamples, seed 260523):

| metric/model/scalar | n | mean | ci_low | ci_high |
|---|---:|---:|---:|---:|
| PF-A/qwen2_5_vl_3b/selectivity | 50 | 0.256129 | 0.183538 | 0.323213 |
| PF-A/qwen2_5_vl_7b/selectivity | 50 | 0.247687 | 0.182228 | 0.310932 |
| PF-A/lvr_7b/selectivity | 50 | 0.250040 | 0.205558 | 0.294823 |
| PF-B/qwen2_5_vl_3b/native_alignment | 50 | 0.740718 | 0.695961 | 0.784680 |
| PF-B/qwen2_5_vl_7b/native_alignment | 50 | 0.702077 | 0.660738 | 0.744787 |
| PF-B/lvr_7b/native_alignment | 50 | 0.806338 | 0.774283 | 0.838802 |
| BF-Patch/qwen2_5_vl_3b/logprob_margin_shift | 50 | 0.015000 | -0.007500 | 0.040000 |
| BF-Patch/qwen2_5_vl_7b/logprob_margin_shift | 50 | 0.010000 | -0.015000 | 0.035000 |
| BF-Patch/lvr_7b/logprob_margin_shift | 50 | -0.048828 | -0.083594 | -0.013281 |
| BF-Swap/qwen2_5_vl_3b/swap_margin_shift | 50 | 0.010000 | -0.012500 | 0.030000 |
| BF-Swap/qwen2_5_vl_7b/swap_margin_shift | 50 | -0.002500 | -0.032500 | 0.027500 |
| BF-Swap/lvr_7b/swap_margin_shift | 50 | -0.034141 | -0.078205 | 0.008365 |
| BF-Conf/qwen2_5_vl_3b/gold_logit_slope | 50 | 0.197275 | 0.191914 | 0.202321 |
| BF-Conf/qwen2_5_vl_7b/gold_logit_slope | 50 | 0.344212 | 0.331840 | 0.355731 |
| BF-Conf/lvr_7b/gold_logit_slope | 50 | 0.068481 | 0.063958 | 0.072880 |
| CF-Stage/qwen2_5_vl_3b/late_delta | 50 | 1.829121 | 1.745603 | 1.928477 |
| CF-Stage/qwen2_5_vl_7b/late_delta | 50 | 2.253331 | 2.156904 | 2.352151 |
| CF-Stage/lvr_7b/late_delta | 50 | 1.307643 | 1.248247 | 1.370573 |

## Notes

- `prereg/manifest.yaml` uses `manifest_version: 0.2-w2-final-gate` and CF-Stage primary scalar `late_delta`.
- `summary_with_ci.json` now groups derived sample rows by `paired_id`, falling back to `id`, before bootstrapping.
- BF-Swap now records self-swap, reverse-swap, and random-pair swap controls. The final validator requires all three controls and self-swap near zero.
- `apply_mask()` now has continuous severity semantics: 0 is clean, 1 is full replacement, and intermediate values alpha-blend the fill color.
- CF-Stage `late_retention` remains present as a diagnostic because near-zero clean late-stage baselines can make it very large. It is no longer the preregistered primary scalar.
- DINO remains optional and disabled for this gate; PF-B reports native attention-proxy alignment.
