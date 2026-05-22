# Week 1 Daily Log

### 2026-05-23
- Implemented: froze the six existing proxy metrics as registry-addressable legacy specs under `pipeline/metrics/legacy/`.
- Commands: `./venv/bin/python smoke_test.py`; `./venv/bin/python -m py_compile ...`.
- Passed: legacy aliases `bf3_legacy`, `pf3_legacy`, `bf1_legacy`, `bf1_layer_legacy`, `cf2_legacy`, and `lvr_trace_legacy` resolve and expose runnable specs.
- Failed / skipped: none.
- Artifacts: `/tmp/legacy_registry_test.json`.
- Next blocker: none.

### 2026-05-24
- Implemented: preregistration manifest and per-run lock writing.
- Commands: `./venv/bin/python smoke_test.py`; GPU `run_all.py` smoke through `tools/run_and_hold.sh`.
- Passed: `runs/gpu_week1_m0_m1_legacy_smoke/prereg.lock.json` records manifest version `0.1-draft` and SHA `39ef744e2fad944efa13f021e1df2bd7c3c3e3f92730e343009a0c0d2f730127`.
- Failed / skipped: none.
- Artifacts: `prereg/manifest.yaml`, `pipeline/preregistration.py`, `runs/gpu_week1_m0_m1_legacy_smoke/prereg.lock.json`.
- Next blocker: manifest remains draft until primary metrics are implemented.

### 2026-05-25
- Implemented: SPD-Faith and Maze loader interfaces plus optional paired/metadata fields on `ProbeSample`.
- Commands: `./venv/bin/python smoke_test.py`.
- Passed: fixture loaders validate paired counterfactual image/answer fields, Maze step metadata, missing-image skip behavior, and stable sampling through existing `load_probe_set()`.
- Failed / skipped: real SPD-Faith/Maze datasets were not present locally, so only fixture-level loading was verified.
- Artifacts: fixture `data_loader_smoke_summary.json` under a temporary smoke directory.
- Next blocker: add real dataset paths once the datasets are staged.

### 2026-05-26
- Implemented: sparse LVR trace v2 adapter with hooks for model forward mode, input embeddings, visual merger, 5 sampled decoder layers, final norm, and LM head.
- Commands: `bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py --config /tmp/lvr_gpu_trace_v2_fixed3.yaml --models lvr_7b --only lvr_generation_trace --device cuda:0`.
- Passed: real LVR-7B trace emits `trace_quality=instrumented_sparse_v0`, `sampled_layers=[0,7,14,21,27]`, `last_hidden_state_shape=[1,1,3584]`, `missing_modules=[]`, `n_embed_calls=69`, and `n_lvr_mode_steps=2`.
- Failed / skipped: initial trace v2 run missed `embed_tokens`; fixed by registering `model.language_model.embed_tokens` and a top-level `lvr_mode_switch` hook.
- Artifacts: `runs/gpu_week1_lvr_trace_v2_fixed3/metrics/lvr_generation_trace_lvr_7b.json`.
- Next blocker: for Week 2, decide how much of `events` to persist or shard before full trace dumps grow large.

### 2026-05-27
- Implemented: paired bootstrap helper, mixed-effects placeholder, and analysis output `summary_with_ci.json`.
- Commands: `./venv/bin/python smoke_test.py`; GPU baseline smoke through `tools/run_and_hold.sh`.
- Passed: `summary_with_ci.json` is produced for legacy BF3/PF3 sample reductions.
- Failed / skipped: mixed-effects fitting is intentionally a Week 2+ placeholder.
- Artifacts: `pipeline/stats/bootstrap.py`, `pipeline/stats/mixed_effects.py`, `runs/gpu_week1_m0_m1_legacy_smoke/summary_with_ci.json`.
- Next blocker: primary v2 metrics need sample-level scalar outputs before CI becomes scientifically useful.

### 2026-05-28
- Implemented: GPU baseline replay for M0/M1 legacy BF3/PF3 on 1 real sample with low-memory settings.
- Commands: `bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py --config /tmp/lvr_gpu_m0_m1_legacy_smoke.yaml --models qwen2_5_vl_7b lvr_7b --only bf3_legacy pf3_legacy --device cuda:0`.
- Passed: `runs/gpu_week1_m0_m1_legacy_smoke/sanity/summary_sanity.json` reports 8 total reports, 8 passed, 0 warnings, 0 failures.
- Failed / skipped: full 20-50 sample replay was not run in this pass; the checked run is a GPU code-path smoke.
- Artifacts: four metric JSON files under `runs/gpu_week1_m0_m1_legacy_smoke/metrics/`, metric plots, sanity summary, prereg lock, and CI summary.
- Next blocker: scale sample count once GPU wall-clock budget is allocated.

### 2026-05-29
- Implemented: Week 2 entry skeletons for PF-A corruption selectivity and BF-Patch answer transfer, plus region corruption utilities.
- Commands: `./venv/bin/python smoke_test.py`; `./venv/bin/python -m py_compile ...`.
- Passed: PF-A fixture validates relevant/irrelevant mask separation and severity-0 identity; BF-Patch fixture validates the 5 x 3 layer-position grid and answer-transfer rate calculation.
- Failed / skipped: no real activation patching forward pass yet.
- Artifacts: `/tmp/v2_metric_schema_smoke.json`, `pipeline/corruptions.py`, `pipeline/metrics/v2/`.
- Next blocker: implement Week 2 primary metric logic on top of these skeletons.
