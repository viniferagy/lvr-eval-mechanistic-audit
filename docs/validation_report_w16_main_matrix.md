# W16 Main Matrix Validation Report

Date: 2026-05-25

## Scope

This report records the W16 implementation state for scaling from gate-level evidence toward the Main-track audit matrix. It adds reusable configs, data loaders, output-accuracy sanity, matrix sharding, merge, and validation tooling.

Full `n=500` / `n=1000` GPU results should be appended here after the runs finish.

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
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_trace_latent.spd_n1000_light.yaml \
  --models lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w16_lvr_trace_latent_spd_n1000_light

./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w16_lvr_trace_latent_spd_n1000_light \
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
BLINK: public HF access verified; full data still needs staging into data/blink
VSI-Bench: public HF question table verified, but local scene/frame-grid images are required
```
