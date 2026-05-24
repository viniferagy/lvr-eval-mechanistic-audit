# W14-W15 Trace-Latent Validation Report

Date: 2026-05-25

## Scope

W14-W15 validates the Main-track scientific red line for LVR-7B: the six v2 primary metrics can now run on real LVR generation-time hidden-feedback states instead of teacher-forced/query-span placeholders.

This report covers LVR only. Monet's W13 gate remains a separate Transformers latent-mode causal patch; modified-vLLM Monet tracing is still future work.

## Artifacts

```text
config: config.lvr_trace_latent.w14_w15.yaml
run_dir: runs/w14_w15_lvr_trace_latent_n50_v2
validator: tools/validate_trace_latent_gate.py
sanity: runs/w14_w15_lvr_trace_latent_n50_v2/sanity/summary_sanity.json
ci: runs/w14_w15_lvr_trace_latent_n50_v2/summary_with_ci.json
plots: runs/w14_w15_lvr_trace_latent_n50_v2/metric_plots/
```

Key implementation files:

```text
pipeline/metrics/v2/trace_latent.py
pipeline/adapters/lvr_qwen.py
pipeline/metrics/v2/pf_a_corruption_selectivity.py
pipeline/metrics/v2/pf_b_patch_alignment.py
pipeline/metrics/v2/bf_patch_answer_transfer.py
pipeline/metrics/v2/bf_swap_latent_replacement.py
pipeline/metrics/v2/bf_conf_calibrated_progression.py
pipeline/metrics/v2/cf_stage_decay.py
pipeline/analysis.py
pipeline/sanity/v2.py
```

## Results

Run summary:

```text
n: 50 SPD-Faith pairs
model: lvr_7b
trace_quality: instrumented_sparse_v0
sanity overall_status: pass
summary_with_ci rows: 20
trace-latent plots: 32
validator: TRACE LATENT GATE VALIDATION PASSED
```

Primary scalar CIs:

| metric | scalar | n | mean | ci_low | ci_high |
|---|---|---:|---:|---:|---:|
| `pf_a_corruption_selectivity` | `selectivity` | 50 | -0.0382235 | -0.0721962 | -0.00607026 |
| `pf_b_patch_alignment` | `native_alignment` | 50 | 0.845704 | 0.822772 | 0.866863 |
| `bf_patch_answer_transfer` | `logprob_margin_shift` | 50 | -0.120000 | -0.280000 | 0.040000 |
| `bf_swap_latent_replacement` | `swap_margin_shift` | 50 | -0.120000 | -0.280000 | 0.000000 |
| `bf_conf_calibrated_progression` | `gold_logit_slope` | 50 | 0.110406 | 0.0545209 | 0.167702 |
| `cf_stage_decay` | `late_delta` | 50 | -4.046318 | -5.252559 | -2.870815 |

Additional reductions:

| metric | additional scalar | value |
|---|---|---:|
| `bf_patch_answer_transfer` | `answer_transfer_rate` | 0.46 |
| `bf_swap_latent_replacement` | `swap_answer_transfer_rate` | 0.46 |
| `cf_stage_decay` | `late_retention` | 0.920282 |
| `pf_a_corruption_selectivity` | `relevant_kl` | 0.194820 |
| `pf_a_corruption_selectivity` | `irrelevant_kl` | 0.156597 |
| `pf_b_patch_alignment` | `irrelevant_alignment` | 0.868674 |

## Visualizations

Representative generated plots:

```text
runs/w14_w15_lvr_trace_latent_n50_v2/metric_plots/bf_patch_answer_transfer_lvr_7b_trace_latent_patch.png
runs/w14_w15_lvr_trace_latent_n50_v2/metric_plots/bf_swap_latent_replacement_lvr_7b_trace_latent_patch.png
runs/w14_w15_lvr_trace_latent_n50_v2/metric_plots/bf_conf_calibrated_progression_lvr_7b_curve.png
runs/w14_w15_lvr_trace_latent_n50_v2/metric_plots/trace_latent_matrix_logprob_margin_shift.png
runs/w14_w15_lvr_trace_latent_n50_v2/metric_plots/trace_latent_matrix_late_delta.png
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

Environment check:

```bash
./venv/bin/python tools/check_lvr_env.py \
  --config config.lvr_trace_latent.w14_w15.yaml
```

GPU gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_trace_latent.w14_w15.yaml \
  --models lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w14_w15_lvr_trace_latent_n50_v2
```

Validation:

```bash
./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w14_w15_lvr_trace_latent_n50_v2 \
  --min-pairs 50 \
  --min-samples 50
```

Validator output:

```text
summary_with_ci rows=20
TRACE LATENT GATE VALIDATION PASSED
```

## Interpretation

This closes the W14/W15 LVR-side trace upgrade: the same six primary metric IDs can now be evaluated against real generation-time hidden feedback when `trace_latent.enabled=true`.

The `n=50` result is still a gate, not a final Main-scale matrix. It establishes that the path is runnable, sanity-checked, and visualized. Larger `n`, cross-task replication, and Monet scheduler-native tracing remain necessary for paper-grade Main evidence.
