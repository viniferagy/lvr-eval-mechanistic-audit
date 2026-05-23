# W2 Final Gate Validation Report

Date: 2026-05-23

This report records the reviewer-gated W2-alpha -> W2-final validation run. It is a runnable-v0 gate result, not a paper-grade causal conclusion.

## Commands

CPU checks:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python -m py_compile run_all.py smoke_test.py merge_and_analyze.py pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py pipeline/adapters/*.py pipeline/stats/*.py tools/validate_spd_range.py
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python smoke_test.py
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
  --only pf_a_corruption_selectivity pf_b_patch_alignment bf_patch_answer_transfer bf_swap_latent_replacement bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w2_final_spd_range_m0_m1_m2_n50
```

Post-run validator:

```bash
./venv/bin/python tools/validate_spd_range.py runs/w2_final_spd_range_m0_m1_m2_n50 --min-pairs 50
```

## Data

Source: `data/spd_faith_hf/manifest.jsonl`

Prepare stats: `data/spd_faith_hf/prepare_stats.json`

Images: `data/spd_faith_hf/images`

`prepare_stats.json`: dataset `Jackson-Lv/SPD-Faith-Bench`, `n_written=2996`, answer policy `clean=original, counterfactual=modified`.

Final range config required a real region oracle (`require_region_or_bbox: true`): 2831 usable pairs, 165 records skipped for missing region/bbox annotation, 50 sampled pairs.

## Trace Gate

Run dir: `runs/w2_final_trace_v2_lvr_n3`

Config: `config.trace_v2.yaml`

Model: `lvr_7b`

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

Artifacts:

| artifact | result |
|---|---:|
| metric JSON files | 18 |
| sanity reports | 18 pass, 0 warn, 0 fail |
| summary_with_ci rows | 63 |
| paired BF-Patch n_success/n_error | 50/0 per model |
| paired BF-Swap n_success/n_error | 50/0 per model |

Primary reductions:

| metric | qwen2_5_vl_3b | qwen2_5_vl_7b | lvr_7b |
|---|---:|---:|---:|
| PF-A selectivity | 0.256129 | 0.247687 | 0.250040 |
| PF-B native_alignment | 0.740718 | 0.702077 | 0.806338 |
| BF-Patch logprob_margin_shift | 0.015000 | 0.010000 | -0.048828 |
| BF-Swap swap_margin_shift | 0.010000 | -0.002500 | -0.034141 |
| BF-Conf gold_logit_slope | 0.197275 | 0.344212 | 0.068481 |
| CF-Stage late_retention | 1829121260.841688 | 2253330912.854936 | 1307643178.436491 |

Primary CI rows (`summary_with_ci.json`, 5000 bootstrap resamples, seed 260523):

| metric/model/scalar | n | mean | ci_low | ci_high |
|---|---:|---:|---:|---:|
| PF-A/qwen2_5_vl_3b/selectivity | 50 | 0.256129 | 0.183051 | 0.324153 |
| PF-A/qwen2_5_vl_7b/selectivity | 50 | 0.247687 | 0.181326 | 0.308651 |
| PF-A/lvr_7b/selectivity | 50 | 0.250040 | 0.205447 | 0.294718 |
| PF-B/qwen2_5_vl_3b/native_alignment | 50 | 0.740718 | 0.694881 | 0.785948 |
| PF-B/qwen2_5_vl_7b/native_alignment | 50 | 0.702077 | 0.659535 | 0.745873 |
| PF-B/lvr_7b/native_alignment | 50 | 0.806338 | 0.773008 | 0.838605 |
| BF-Patch/qwen2_5_vl_3b/logprob_margin_shift | 50 | 0.015000 | -0.007500 | 0.037500 |
| BF-Patch/qwen2_5_vl_7b/logprob_margin_shift | 50 | 0.010000 | -0.017500 | 0.035000 |
| BF-Patch/lvr_7b/logprob_margin_shift | 50 | -0.048828 | -0.084377 | -0.014297 |
| BF-Swap/qwen2_5_vl_3b/swap_margin_shift | 50 | 0.010000 | -0.010000 | 0.030000 |
| BF-Swap/qwen2_5_vl_7b/swap_margin_shift | 50 | -0.002500 | -0.032500 | 0.027500 |
| BF-Swap/lvr_7b/swap_margin_shift | 50 | -0.034141 | -0.078438 | 0.010080 |
| BF-Conf/qwen2_5_vl_3b/gold_logit_slope | 50 | 0.197275 | 0.191920 | 0.202327 |
| BF-Conf/qwen2_5_vl_7b/gold_logit_slope | 50 | 0.344212 | 0.332231 | 0.355746 |
| BF-Conf/lvr_7b/gold_logit_slope | 50 | 0.068481 | 0.063791 | 0.073092 |
| CF-Stage/qwen2_5_vl_3b/late_retention | 50 | 1829121260.841688 | 1742996867.063145 | 1922794570.120672 |
| CF-Stage/qwen2_5_vl_7b/late_retention | 50 | 2253330912.854936 | 2155290780.630377 | 2359334859.828155 |
| CF-Stage/lvr_7b/late_retention | 50 | 1307643178.436491 | 1248655374.965734 | 1370836147.122913 |

## Notes

- Trace v2 initially failed correctly under hard-fail sanity when full-resolution LVR samples OOMed and attempted fallback. The final trace gate uses a bounded 384px image budget and passes with true sparse instrumentation.
- The first SPD range attempt correctly failed PF sanity when random sampling included 4 samples without bbox/region masks. The final config requires region/bbox oracle before sampling.
- DINO remains optional and disabled for this gate; PF-B reports native attention-proxy alignment.
- CF-Stage `late_retention` values are very large because the clean late-stage baseline is near zero under this runnable-v0 proxy. Treat CF-Stage as a gate artifact requiring semantic calibration before main-paper use.
