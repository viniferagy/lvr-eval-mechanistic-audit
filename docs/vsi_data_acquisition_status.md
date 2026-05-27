# VSI-Bench Visual Data Acquisition Status

Date: 2026-05-27

## Summary

T4 visual data is now staged locally from the official Hugging Face VSI-Bench repository. The staged subset is the official `nyu-visionx/VSI-Bench` `scannetpp` video source, converted into static frame-grid JPEGs for `output_accuracy_sanity`.

This is not the full three-source VSI visual set yet. It is a paper-grade T4 accuracy sanity subset because it provides 1458 usable QA rows, above the `n >= 800` validation target. The 32-frame three-model accuracy run is now complete and merged at `runs/main_vsi_accuracy_scannetpp_32f_paper/merged`.

## HF Source Check

| Source | Result |
|---|---|
| `mmaaz60/VSI_Bench` | `datasets.load_dataset()` exposes 5130 annotation rows but no visible `image`, `images`, `frames`, or `frame_grid` payload fields in this environment. |
| `nyu-visionx/VSI-Bench` | `datasets.load_dataset()` exposes 5130 annotation rows. Visual payloads are repository files: `arkitscenes.zip`, `scannet.zip`, and `scannetpp.zip`. |

Official source distribution:

| Dataset source | QA rows |
|---|---:|
| `arkitscenes` | 1601 |
| `scannet` | 2071 |
| `scannetpp` | 1458 |

## Local Staging

Frame-grid generation for the paper-facing 32-frame run:

```bash
./venv/bin/python tools/prepare_vsi_frame_grids_from_hf.py \
  --sources scannetpp \
  --cache-dir data/vsi_raw/hf_cache \
  --extract-root data/vsi_raw/videos \
  --out data/vsi_raw/frame_grids_32f \
  --n-frames 32 \
  --thumb 256 \
  --grid-cols 8 \
  --quality 90
```

Manifest preparation:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
./venv/bin/python tools/prepare_vsi_bench_hf.py \
  --dataset nyu-visionx/VSI-Bench \
  --split test \
  --image-root data/vsi_raw/frame_grids_32f \
  --out data/vsi_bench_32f \
  --max-samples 5130
```

## Current Stats

Frame-grid stats:

| Field | Value |
|---|---:|
| source zip | `scannetpp.zip` |
| frame-grid protocol | 32-frame 4x8 static image grid |
| scene grids written | 50 |
| missing visual scenes | 0 |
| decode failures | 0 |
| covered QA rows | 1458 |

Prepared manifest stats:

| Field | Value |
|---|---:|
| source annotation rows | 5130 |
| written rows | 1458 |
| skipped no image | 3672 |
| skipped no answer | 0 |
| unique external images | 50 |
| prepared dataset coverage | `scannetpp: 1458` |

Prepared files:

```text
data/vsi_raw/frame_grids_32f/scannetpp/
data/vsi_raw/frame_grids_32f/frame_grid_prepare_stats.json
data/vsi_bench_32f/manifest.jsonl
data/vsi_bench_32f/prepare_stats.json
data/vsi_bench_32f/images/
```

## Completed Accuracy Gate

The three-model 32-frame accuracy run and merge are complete:

| Model | Accuracy | 95% bootstrap CI | Hits | n | Errors |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 0.196 | [0.172, 0.222] | 196 | 1000 | 0 |
| Qwen2.5-VL-7B | 0.191 | [0.168, 0.215] | 191 | 1000 | 0 |
| LVR-7B | 0.110 | [0.091, 0.130] | 110 | 1000 | 0 |

Merged artifact:

```text
runs/main_vsi_accuracy_scannetpp_32f_paper/merged/vsi_accuracy_summary.json
```

Merge command:

```bash
./venv/bin/python tools/merge_vsi_accuracy.py \
  runs/main_vsi_accuracy_scannetpp_32f_paper \
  --out runs/main_vsi_accuracy_scannetpp_32f_paper/merged \
  --min-samples 800
```

The merged sanity summary reports 3 pass, 0 warn, and 0 fail. The LVR row uses the fixed rerun at `config.main_vsi_accuracy_32f_lvr_7b_output_accuracy_sanity_fixed`; the first 32-frame LVR attempt was stopped after loading but making no sample progress.

Optional paper-ready validation against the W18 T1/T2/T3 matrix:

```bash
./venv/bin/python tools/validate_main_paper_readiness.py \
  runs/w18_main_matrix_full_local/merged \
  --mode paper_ready \
  --t4-accuracy-summary runs/main_vsi_accuracy_scannetpp_32f_paper/merged/vsi_accuracy_summary.json \
  --t4-min-samples 800
```
