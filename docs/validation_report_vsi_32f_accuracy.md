# VSI-Bench 32-Frame Accuracy Sanity Report

Date: 2026-05-27

## Summary

The T4 VSI-Bench `scannetpp` visual subset is now completed as a three-model output-accuracy sanity run under a 32-frame static grid protocol. This is not a causal audit metric and is not directly comparable to native video-LLM VSI-Bench scores, but it is now a merged, validated paper-readiness artifact for external task accuracy sanity.

Merged artifact:

```text
runs/main_vsi_accuracy_scannetpp_32f_paper/merged/vsi_accuracy_summary.json
```

Sanity status:

| Check | Result |
|---|---|
| required models present | pass |
| per-model samples | 1000 each |
| per-model runtime errors | 0 each |
| merged sanity reports | 3 pass, 0 warn, 0 fail |

## Data Boundary

The run uses the official `nyu-visionx/VSI-Bench` `scannetpp.zip` visual source. The videos were converted into static frame-grid JPEGs, and the annotation manifest contains 1458 usable QA rows from 50 scene grids. The evaluation config caps the run at 1000 shuffled samples with seed `260721`.

Frame-grid protocol:

| Field | Value |
|---|---:|
| source | `nyu-visionx/VSI-Bench` `scannetpp.zip` |
| visual protocol | 32-frame 4x8 static image grid |
| thumbnail size | 256 |
| scene grids written | 50 |
| missing visuals | 0 |
| decode failures | 0 |
| usable QA rows | 1458 |
| evaluated rows | 1000 |

## Results

| Model | Accuracy | 95% bootstrap CI | Hits | n | Errors |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 0.196 | [0.172, 0.222] | 196 | 1000 | 0 |
| Qwen2.5-VL-7B | 0.191 | [0.168, 0.215] | 191 | 1000 | 0 |
| LVR-7B | 0.110 | [0.091, 0.130] | 110 | 1000 | 0 |

The LVR-7B result comes from the fixed rerun:

```text
runs/main_vsi_accuracy_scannetpp_32f_paper/config.main_vsi_accuracy_32f_lvr_7b_output_accuracy_sanity_fixed
```

The earlier 32-frame LVR attempt was terminated because it loaded the model but did not reach sample progress. The fixed run reached `1000/1000` in 16m32s and wrote `n_error=0`.

## LVR Adapter Fix

The failed LVR attempt exposed an adapter mismatch: `LVRQwenAdapter` inherited the baseline Qwen generate path and did not pass official LVR generation arguments. The adapter now implements an LVR-specific `generate()` method that calls `wrapper.model.generate()` with:

```text
decoding_strategy="steps"
lvr_steps=[16] * batch_size
do_sample=False
```

This keeps T4 output accuracy on the official LVR generation path while avoiding the no-progress behavior seen in the first 32-frame attempt.

## Commands

The completed LVR fixed rerun used:

```bash
bash tools/run_and_hold.sh 0,1,2,3 \
  bash -lc 'CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python run_all.py --config config.main_vsi_accuracy_32f.yaml --models lvr_7b --only output_accuracy_sanity --device cuda:0 --output-root runs/main_vsi_accuracy_scannetpp_32f_paper --run-name config.main_vsi_accuracy_32f_lvr_7b_output_accuracy_sanity_fixed --no-analysis > runs/main_vsi_accuracy_scannetpp_32f_paper/log_config.main_vsi_accuracy_32f_lvr_7b_output_accuracy_sanity_fixed.txt 2>&1'
```

The final merge used:

```bash
./venv/bin/python tools/merge_vsi_accuracy.py \
  runs/main_vsi_accuracy_scannetpp_32f_paper \
  --out runs/main_vsi_accuracy_scannetpp_32f_paper/merged \
  --min-samples 800
```

Merge output:

```text
MERGED_VSI_ACCURACY runs/main_vsi_accuracy_scannetpp_32f_paper/merged models=lvr_7b,qwen2_5_vl_3b,qwen2_5_vl_7b
```

## Interpretation

These numbers should be used as T4 output-accuracy sanity evidence only. They show that the staged VSI visual subset can be evaluated reproducibly across the three model families with finite scores, adequate sample count, and zero runtime errors. They do not measure latent-state availability, causal usage, or retention, and they should not be compared as published VSI-Bench video scores because the input is a static 32-frame image grid.
