# W16 Main Matrix Validation Report

Date: 2026-05-25

## Scope

This report records the W16 implementation state for scaling from gate-level evidence toward the Main-track audit matrix. It adds reusable configs, data loaders, output-accuracy sanity, matrix sharding, merge, and validation tooling.

The LVR trace-latent six-metric `n=500` scale gate is complete. The follow-up
W17 `n=1000` light trace-latent gate is also complete for PF-A, PF-B, BF-Conf,
and CF-Stage. On 2026-05-25, a real hundred-scale checkpoint was also run to
verify the full W16 path end to end: LVR trace-latent SPD `n=100` and T1/T2/T3
main matrix `n=100`.

## Configs

```text
config.lvr_trace_latent.spd_n500.yaml
config.lvr_trace_latent.spd_n1000_light.yaml
config.main_spd_n1000.yaml
config.main_maze_n1000.yaml
config.main_blink_n1000.yaml
config.main_vsi_accuracy.yaml
```

## Commands

Data preparation:

```bash
./venv/bin/python tools/prepare_blink_hf.py \
  --configs all \
  --split val \
  --out data/blink

./venv/bin/python tools/prepare_vsi_bench_hf.py \
  --split test \
  --image-root /path/to/local/vsi/frame_grids \
  --out data/vsi_bench
```

BLINK smoke preparation was verified against the public `Art_Style` config. VSI-Bench exposes question metadata on Hugging Face but requires a local image/frame-grid root for runnable VLM samples.

LVR trace-latent `n=500` six-metric scale gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_trace_latent.spd_n500.yaml \
  --models lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w16_lvr_trace_latent_spd_n500

./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w16_lvr_trace_latent_spd_n500 \
  --min-pairs 500 \
  --min-samples 500
```

LVR trace-latent `n=1000` light gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.lvr_trace_latent.spd_n1000_light.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w17_lvr_trace_latent_spd_n1000_light

./venv/bin/python tools/merge_main_matrix.py \
  runs/w17_lvr_trace_latent_spd_n1000_light

./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w17_lvr_trace_latent_spd_n1000_light/merged \
  --min-pairs 0 \
  --min-samples 800 \
  --allow-disabled-metrics
```

T1/T2/T3 main matrix:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.main_maze_n1000.yaml,config.main_spd_n1000.yaml,config.main_blink_n1000.yaml \
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \
  --metrics all \
  --gpus 0,1,2,3 \
  --run-root runs/w16_main_matrix_t1_t2_t3

./venv/bin/python tools/merge_main_matrix.py runs/w16_main_matrix_t1_t2_t3

./venv/bin/python tools/validate_main_matrix.py \
  runs/w16_main_matrix_t1_t2_t3 \
  --tasks maze,spd_faith,blink \
  --min-samples 800 \
  --bf-min-pairs 300
```

T4 VSI accuracy sanity:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.main_vsi_accuracy.yaml \
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b,monet_7b \
  --metrics output_accuracy_sanity \
  --gpus 0,1,2,3 \
  --run-root runs/w16_vsi_accuracy_sanity

./venv/bin/python tools/merge_main_matrix.py runs/w16_vsi_accuracy_sanity

./venv/bin/python tools/validate_main_matrix.py \
  runs/w16_vsi_accuracy_sanity \
  --accuracy-only \
  --min-samples 200
```

## Implementation Validation

Required local checks:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python -m py_compile \
  run_all.py smoke_test.py merge_and_analyze.py \
  pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py \
  pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py \
  pipeline/adapters/*.py pipeline/stats/*.py tools/*.py

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python smoke_test.py
```

Completed implementation checks:

```text
py_compile: pass
smoke_test.py: pass
launch_main_matrix.sh --dry-run: pass
BLINK real HF smoke: Art_Style / val / n=2 converted successfully
LVR trace-latent GPU entry smoke: runs/w16_trace_latent_entry_smoke, PF-A n=50, sanity pass
trace-latent subset validator: pass on runs/w16_trace_latent_entry_smoke
```

Current data staging state:

```text
SPD-Faith local manifest: 2996 records, enough for n=1000
MazePlanning local manifest: 500 records, below the W16 n=800 validator target
BLINK: Art_Style val n=100 staged at data/blink_art_style_n100; full BLINK staging remains future work
VSI-Bench: public HF question table verified, but local scene/frame-grid images are required
```

## Recorded Hundred-Scale Runs

## Recorded LVR Trace-Latent n=500 Scale Gate

LVR trace-latent SPD `n=500`, sharded by metric across four GPUs through
`tools/run_and_hold.sh`:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.lvr_trace_latent.spd_n500.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_patch_answer_transfer|bf_swap_latent_replacement|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w16_lvr_trace_latent_spd_n500_sharded

./venv/bin/python tools/merge_main_matrix.py \
  runs/w16_lvr_trace_latent_spd_n500_sharded

./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w16_lvr_trace_latent_spd_n500_sharded/merged \
  --min-pairs 500 \
  --min-samples 500
```

Validation result:

```text
TRACE LATENT GATE VALIDATION PASSED
sanity: 6/6 pass
summary_with_ci rows: 20
```

Primary LVR trace-latent SPD `n=500` results:

| Metric | n | value | 95% bootstrap CI |
|---|---:|---:|---:|
| PF-A selectivity | 500 | -0.044502 | [-0.054109, -0.035044] |
| PF-B native_alignment | 500 | 0.842298 | [0.835494, 0.848849] |
| BF-Patch logprob_margin_shift | 500 | -0.004000 | [-0.040000, 0.032000] |
| BF-Patch answer_transfer_rate | 500 | 0.462000 | not bootstrapped in current secondary summary |
| BF-Swap swap_margin_shift | 500 | -0.012000 | [-0.048000, 0.024000] |
| BF-Swap swap_answer_transfer_rate | 500 | 0.462000 | not bootstrapped in current secondary summary |
| BF-Conf gold_logit_slope | 500 | 0.150825 | [0.134188, 0.167482] |
| CF-Stage late_delta | 500 | -3.372306 | [-3.852121, -2.902451] |

## Recorded W17 LVR Trace-Latent n=1000 Light Gate

LVR trace-latent SPD `n=1000`, sharded by metric across four GPUs through
`tools/run_and_hold.sh`:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.lvr_trace_latent.spd_n1000_light.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w17_lvr_trace_latent_spd_n1000_light

./venv/bin/python tools/merge_main_matrix.py \
  runs/w17_lvr_trace_latent_spd_n1000_light

./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w17_lvr_trace_latent_spd_n1000_light/merged \
  --allow-disabled-metrics \
  --min-samples 800 \
  --min-pairs 0

./venv/bin/python tools/validate_main_paper_readiness.py \
  runs/w17_lvr_trace_latent_spd_n1000_light/merged
```

Validation result:

```text
TRACE LATENT GATE VALIDATION PASSED
MAIN PAPER READINESS VALIDATION PASSED
sanity: 4/4 pass
summary_with_ci rows: 16
```

Primary LVR trace-latent SPD `n=1000` light results:

| Metric | n | value | 95% bootstrap CI |
|---|---:|---:|---:|
| PF-A selectivity | 1000 | -0.055257 | [-0.062864, -0.047651] |
| PF-B native_alignment | 1000 | 0.835482 | [0.830548, 0.840310] |
| BF-Conf gold_logit_slope | 1000 | 0.139953 | [0.128714, 0.151739] |
| CF-Stage late_delta | 1000 | -3.844755 | [-4.168295, -3.525368] |

This is a light gate. BF-Patch and BF-Swap are disabled by config and remain
represented at scale by the W16 `n=500` all-six run.

Trace-latent n=500 visualization:

```text
runs/w16_lvr_trace_latent_spd_n500_sharded/merged/metric_plots/w16_trace_latent_n500_primary_heatmap.png
```

LVR trace-latent SPD `n=100`, sharded across four GPUs through
`tools/run_and_hold.sh`:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs /tmp/lvr_trace_latent_spd_n100.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_patch_answer_transfer|bf_swap_latent_replacement|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w16_lvr_trace_latent_spd_n100_sharded

./venv/bin/python tools/merge_main_matrix.py \
  runs/w16_lvr_trace_latent_spd_n100_sharded

./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w16_lvr_trace_latent_spd_n100_sharded/merged \
  --min-pairs 100 \
  --min-samples 100
```

Validation result:

```text
TRACE LATENT GATE VALIDATION PASSED
sanity: 6/6 pass
summary_with_ci rows: 20
```

Primary LVR trace-latent SPD results:

| Metric | n | value | 95% bootstrap CI |
|---|---:|---:|---:|
| PF-A selectivity | 100 | -0.045730 | [-0.067747, -0.024061] |
| PF-B native_alignment | 100 | 0.845153 | [0.830550, 0.859188] |
| BF-Patch logprob_margin_shift | 100 | 0.060000 | [-0.020000, 0.160000] |
| BF-Patch answer_transfer_rate | 100 | 0.460000 | not bootstrapped in current secondary summary |
| BF-Swap swap_margin_shift | 100 | 0.000000 | [-0.100000, 0.100000] |
| BF-Swap swap_answer_transfer_rate | 100 | 0.450000 | not bootstrapped in current secondary summary |
| BF-Conf gold_logit_slope | 100 | 0.149496 | [0.112923, 0.188046] |
| CF-Stage late_delta | 100 | -3.885112 | [-4.934226, -2.796776] |

Trace-latent visualization:

```text
runs/w16_lvr_trace_latent_spd_n100_sharded/merged/metric_plots/w16_trace_latent_n100_primary_heatmap.png
```

T1/T2/T3 main matrix `n=100`:

```bash
./venv/bin/python tools/prepare_blink_hf.py \
  --configs Art_Style \
  --split val \
  --max-samples 100 \
  --out data/blink_art_style_n100

bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs /tmp/main_maze_n100.yaml,/tmp/main_spd_n100.yaml,/tmp/main_blink_n100.yaml \
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \
  --metrics all \
  --gpus 0,1,2,3 \
  --run-root runs/w16_main_matrix_t1_t2_t3_n100

./venv/bin/python tools/merge_main_matrix.py \
  runs/w16_main_matrix_t1_t2_t3_n100

./venv/bin/python tools/validate_main_matrix.py \
  runs/w16_main_matrix_t1_t2_t3_n100 \
  --tasks maze,spd_faith,blink \
  --min-samples 100 \
  --bf-min-pairs 100
```

Validation result:

```text
MAIN MATRIX VALIDATION PASSED rows=42 ci_rows=168
sanity: 42/42 pass, overall warn because BLINK Art_Style has weak-oracle center fallback for PF metrics
```

T1 Maze `n=100`:

| Metric | Qwen3B | Qwen7B | LVR-7B |
|---|---:|---:|---:|
| PF-A selectivity | -0.660587 | -0.898330 | -0.644481 |
| PF-B native_alignment | 0.475198 | 0.390295 | 0.458739 |
| BF-Conf gold_logit_slope | 0.190406 | 0.278682 | 0.030451 |
| CF-Stage late_delta | 1.465018 | 1.974893 | 1.516935 |

T2 SPD-Faith `n=100`:

| Metric | Qwen3B | Qwen7B | LVR-7B |
|---|---:|---:|---:|
| PF-A selectivity | 0.176829 | 0.192988 | 0.206082 |
| PF-B native_alignment | 0.684626 | 0.672777 | 0.769803 |
| BF-Patch logprob_margin_shift | 0.008750 | 0.020000 | -0.017578 |
| BF-Swap swap_margin_shift | 0.006250 | 0.026250 | -0.031289 |
| BF-Conf gold_logit_slope | 0.192592 | 0.349234 | 0.070154 |
| CF-Stage late_delta | 1.845391 | 2.195018 | 1.387603 |

T3 BLINK Art_Style `n=100`:

| Metric | Qwen3B | Qwen7B | LVR-7B |
|---|---:|---:|---:|
| PF-A selectivity | 0.304636 | 0.243137 | 0.278320 |
| PF-B native_alignment | 0.543969 | 0.529590 | 0.634386 |
| BF-Conf gold_logit_slope | -0.103736 | 0.069303 | 0.048056 |
| CF-Stage late_delta | 1.725590 | 1.948871 | 1.565728 |

Main-matrix visualization:

```text
runs/w16_main_matrix_t1_t2_t3_n100/merged/metric_plots/w16_main_matrix_t1_t2_t3_n100_primary_heatmap.png
```

T4 VSI-Bench was checked but not run as accuracy sanity. The public HF table
provides `scene_name`, `question`, and `ground_truth`, but no image or frame-grid
payload. After adding `ground_truth` answer support, a two-row preparation check
still wrote zero records with `n_skipped_no_image=2`, confirming that the blocker
is visual data. A real T4 run requires local scene frames or prebuilt frame-grid
images.
