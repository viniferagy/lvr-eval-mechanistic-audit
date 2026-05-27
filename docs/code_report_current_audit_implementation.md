# Code Report: Current Latent Visual Faithfulness Audit Implementation

Date: 2026-05-27

This report explains the implementation in the order a reader should understand it: data representation, model adapters, generation tracing, metrics, orchestration, merging, validation, and known boundaries. The completed experiment paths have passed syntax, smoke, sanity, and gate validators. Some planned Main-paper components are intentionally not implemented yet; those are listed explicitly so they are not mistaken for silent code errors.

## 1. Execution Flow

The main entry point is `run_all.py`.

The flow is:

1. Read a YAML config.
2. Validate required config sections: `models`, `data`, `inference`, and `output`.
3. Create a run directory and write `config_snapshot.yaml`.
4. Write a preregistration lock from `prereg/manifest.yaml` unless disabled.
5. Load the probe set through `pipeline.data.load_probe_set`.
6. Resolve metric IDs through `pipeline.metrics.registry`.
7. Load each selected model through `pipeline.model_utils.load_model`.
8. Run each selected metric and write metric envelopes under `metrics/*.json`.
9. Run sanity reports through `pipeline.sanity.report`.
10. Run analysis and visualization through `pipeline.analysis.run_analysis`, unless `--no-analysis` is set.

This means each result directory has enough local state to be audited later: config snapshot, prereg lock, raw metric JSON, sanity JSON, summary files, and plots.

Correctness checks in this layer:

- `validate_config` fails early if a config is structurally incomplete.
- `selected_metric_ids` normalizes aliases such as `pf_a`, `bf_patch`, and `output_accuracy`.
- Model execution preserves the resolved `--models` order; there is no longer any LVR-specific scheduling or download-wait branch.
- After each model, the wrapper is deleted and `torch.cuda.empty_cache()` is called.

## 2. Data Representation and Loaders

All datasets are converted into `pipeline.data.ProbeSample`. This dataclass is the contract between loaders and metrics. Important fields include:

| Field | Purpose |
|---|---|
| `image`, `question`, `answer` | core VQA input and target |
| `counterfactual_image`, `counterfactual_answer` | paired BF-Patch and BF-Swap source/target comparisons |
| `bboxes`, `region_mask` | PF-A/PF-B visual oracle information |
| `rationale` | optional reasoning trace or explanation |
| `paired_id` | grouping key for paired bootstrap and summaries |
| `task_metadata` | source-specific metadata such as weak-oracle flags |

Implemented loaders:

- `pipeline/data_spd_faith.py`: loads SPD-Faith paired counterfactual records. It requires clean and counterfactual images for paired BF metrics and can require region/bbox metadata for PF metrics. This is the T2 main paired task.
- `pipeline/data_maze.py`: loads MazePlanning image, question, answer, rationale, and optional bboxes. This is the T1 planning/retention task.
- `pipeline/data_blink.py`: loads BLINK manifest rows and appends answer choices to the prompt when present. If no bboxes exist, it records `weak_oracle=True`; PF validators can either fail or warn depending on config. W18 uses the all-config staged local BLINK manifest at `n=1000`, all weak-oracle.
- `pipeline/data_vsi.py`: loads VSI-Bench-style static frame-grid image records for output accuracy sanity. The local T4 staging now uses official `nyu-visionx/VSI-Bench` annotations plus frame-grid JPEGs generated from the official `scannetpp.zip` video file.
- `pipeline/data_vstar.py`: loads V*Bench high-resolution bbox-localization records prepared from the full HF repository snapshot. It requires bbox metadata by default, records high-resolution and bbox-area metadata, appends answer choices to prompts, and is intended as a small-n spotlight task rather than a Main matrix task.

The loader layer is structurally sound for completed runs. BLINK is staged at `n=1000`, but all rows are weak-oracle center-fallback cases because reliable region boxes are absent locally. VSI-Bench visual staging is now present for the official `scannetpp` subset: `tools/prepare_vsi_frame_grids_from_hf.py` converts the official HF MP4 scenes into static frame-grid JPEGs, and `tools/prepare_vsi_bench_hf.py` matches them through dataset-scoped paths. V*Bench/VStar is no longer blocked: `tools/prepare_vstar_local.py` uses `huggingface_hub.snapshot_download` to fetch the full `craigwu/vstar_bench` repository, because `datasets.load_dataset` / parquet / Data Studio omit the per-image JSON bbox files. The W20 preparation path reads `direct_attributes/sa_XXXXX.json` and `relative_position/sa_XXXXX.json`, converts V*Bench `[x, y, w, h]` boxes to absolute `xyxy`, and writes 191 usable samples with zero missing bbox records.

VSI-Bench staging outputs:

```text
data/vsi_raw/frame_grids/scannetpp/
data/vsi_raw/frame_grids/frame_grid_prepare_stats.json
data/vsi_bench/manifest.jsonl
data/vsi_bench/prepare_stats.json
data/vsi_bench/images/
```

The current VSI visual staging uses `nyu-visionx/VSI-Bench` `scannetpp.zip`: 50 scene grids written, zero missing visuals, zero decode failures, 1458 usable QA rows, zero missing answers, and 50 unique external frame-grid images. The paper-facing T4 accuracy artifact now uses the upgraded 32-frame 4x8 static grids at `data/vsi_raw/frame_grids_32f` and `data/vsi_bench_32f`; the earlier 16-frame 4x4 staging remains a superseded local preparation artifact. These are still static grids, not native video inputs, so the result should not be compared directly to published video-LLM VSI-Bench numbers. The conversion script records the requested frame count and performs an explicit first-frame decoder sanity check so OpenCV/H.264 failures cannot silently produce empty grids. The full official annotation table has 5130 rows; rows from `arkitscenes` and `scannet` are intentionally skipped until their corresponding zips are converted, so the current paper-facing T4 boundary must say "VSI-Bench scannetpp 32-frame static-grid visual subset" rather than full VSI-Bench.

V*Bench staging outputs:

```text
data/vstar/manifest.jsonl
data/vstar/prepare_stats.json
data/vstar/images/
```

The current prepared stats are `n_seen=191`, `n_written=191`, `attribute_recognition=115`, `spatial_relationship_reasoning=76`, and median bbox area ratio `0.000770`. This confirms that W20 is a high-resolution small-region localization stress test, not a generic VQA table.

## 3. Model Wrappers and Adapters

`pipeline/model_utils.py` defines `VLMWrapper`, which stores the loaded model, processor, adapter, decoder layers, final norm, LM head, device, image token ID, and config. Metrics use the wrapper instead of depending on one model class.

Adapters live under `pipeline/adapters/`:

- `qwen_vl.py`: standard Qwen2.5-VL/Qwen3-VL adapter for baseline models.
- `lvr_qwen.py`: official LVR model adapter. It supports teacher-forced LVR span audits, T4 output-accuracy generation with explicit LVR `decoding_strategy`/`lvr_steps`, and a legacy approximate `generate_with_trace` path based on generated token IDs.
- `lvr_qwen_traced.py`: traced LVR adapter used by W14-W17 trace-latent experiments. This is the real generation-time path for current LVR trace-latent evidence.
- `monet_qwen.py`: Monet adapter for the official Transformers latent-mode path. It intentionally does not implement modified-vLLM scheduler-native generation tracing.

Important adapter distinction:

`LVRQwenAdapter.generate_with_trace` in `pipeline/adapters/lvr_qwen.py` returns `trace_quality="approx_from_generated_token_ids"`. This is not sufficient for W14-W17 trace-latent gates. The trace-latent configs use `arch: lvr_qwen2_5_vl_traced`, which routes to `TracedLVRQwenAdapter` and `TraceRecorder`. Validators accept instrumented trace qualities such as `instrumented_sparse_v0` and the reserved future `monet_vllm_latent_v0`, so accidental fallback to the approximate path is caught.

## 4. LVR Trace Recorder

`pipeline/adapters/lvr_qwen_traced.py` implements the sparse hook instrumentation.

The core class is `TraceRecorder`. During generation, it registers hooks on:

- model forward pre-hook;
- model forward output hook;
- input embedding module;
- final norm;
- visual merger;
- sampled decoder layers;
- LM head.

It records:

- whether LVR mode was active;
- `last_position_hidden_state` shape;
- generated hidden-feedback steps;
- sparse decoder-layer events;
- captured hidden-state tensors when `capture_tensors=True`;
- patch counts when replacement states are injected.

For trace-latent patching, the forward pre-hook can replace `last_position_hidden_state` with a source tensor. Shape policy is configurable as `strict` or `slice`; W16/W17 use strict shape matching. The recorder stores captured tensors as CPU float32 by default to avoid holding unnecessary GPU memory after generation.

The trace summary sets:

```text
trace_quality = instrumented_sparse_v0
```

This is the quality marker enforced by `tools/validate_trace_latent_gate.py`.

## 5. Metrics

Metrics are registered in `pipeline/metrics/registry.py`. The current primary v2 metric files are under `pipeline/metrics/v2/`.

### PF-A: `pf_a_corruption_selectivity`

This metric compares model-internal changes after masking relevant, irrelevant, and random visual regions. In standard/query-span mode it uses prompt-side internal readouts. In trace-latent mode it calls `trace_latent.generate_trace` on clean and masked images, then computes latent-state distances over captured generation states.

The image-corruption path now supports optional bbox dilation through `mask_dilate_px`. This is passed from PF-A/PF-B configs into `pipeline/corruptions.relevant_mask`. The preregistered W20 V*Bench run keeps dilation at `0` so the reported spotlight result reflects the native JSON boxes; future sensitivity checks can set a positive value when tiny boxes are too small to perturb the model.

Primary scalar:

```text
selectivity = irrelevant_delta - relevant_delta
```

A negative value therefore means relevant-region corruption moved the latent state more than irrelevant-region corruption under the current sign convention.

### PF-B: `pf_b_patch_alignment`

This metric reports native visual alignment between hidden/latent representations and region-derived visual signals. It also now has an optional DINOv2 external image-backbone sensitivity backend. Native PF-B remains the model-facing primary scalar; DINO rows are an external masked-image sensitivity check over the same clean/relevant/irrelevant/random images and should not replace the native model-alignment claim.

Like PF-A, PF-B can pass `mask_dilate_px` into the region-mask builder. W20 V*Bench uses the same no-dilation setting as PF-A and keeps the 1024px audit image budget for localization-sensitive readouts.

When `pf_b.use_dino=true`, `pipeline/metrics/v2/dino_backend.py` loads `facebook/dinov2-small` through Transformers, embeds clean and masked images, caches embeddings under `.cache/dino_pf_b`, and writes `dino_alignment`, `dino_relevant_alignment`, `dino_irrelevant_alignment`, `dino_random_alignment`, and `dino_region_selectivity`. `pipeline/sanity/v2.py` treats missing DINO output as a failure when DINO is requested. Smoke tests use the deterministic `pf_b.dino_backend=mock` path so CI does not require network or model weights.

Primary scalar:

```text
native_alignment
```

### BF-Patch: `bf_patch_answer_transfer`

This metric is the main causal usage test. For paired samples, it selects a source/counterfactual sample and a target/clean sample. It captures source latent states, reruns the target while injecting the source state, and measures whether the answer or answer margin moves toward the source answer.

In trace-latent mode, the intervention site is:

```text
forward_pre.last_position_hidden_state
```

For generation-score margins, the code now uses an aligned diagnostic path rather than the old `scores[-1]` assumption. `score_margin_with_diagnostics()` derives generated token ids from the adapter trace, checks whether the source/target candidates are single-token answers, finds the generated candidate token, and uses the matching score index only when the token-score alignment is explicit. Otherwise it records a machine-readable failure reason such as `missing_scores`, `multi_token_candidate`, `decision_index_mismatch`, `score_generated_length_mismatch`, `bad_score_shape`, or `nonfinite_score`.

When aligned generation scores are unavailable, the trace-latent path falls back to a continuous latent logit-lens margin computed from the final captured hidden-feedback state through `final_norm` and `lm_head`. Only if both continuous paths fail does it use the parsed constrained-answer fallback. Patch records expose `clean_score_diagnostic`, `patched_score_diagnostic`, `score_failure_reasons`, `margin_source`, `clean_margin_source`, `patched_margin_source`, `generation_score_margin_shift`, `continuous_margin_shift`, `effective_margin_shift`, and `latent_logit_margin_shift`. Validators can fail runs that rely too heavily on parsed-answer fallback, and can also require diagnostic summaries for new patch/swap artifacts.

In standard/query-span BF-Patch runs, the implementation computes forced candidate-sequence log probabilities for the source and target answer strings. That legacy value is a continuous sequence-logprob margin, so `patch_one_pair()` now also writes it into `sequence_logprob_margin_shift`, `effective_margin_shift`, and `continuous_margin_shift`, with `margin_source="candidate_sequence_logprob"`. `pipeline.analysis.build_summary_with_ci()` and `pipeline.stats.mixed_effects.fit_mixed_effects()` also provide a compatibility alias from standard `logprob_margin_shift` / `swap_margin_shift` into `continuous_margin_shift` when reading older non-trace artifacts. This alias is disabled for trace-latent payloads so parsed-answer fallback cannot silently enter the Main-facing continuous scalar.

Primary scalar:

```text
continuous_margin_shift
```

Important secondary scalar:

```text
answer_transfer_rate
```

### BF-Swap: `bf_swap_latent_replacement`

This metric performs controlled latent replacement. It shares much of the trace-latent intervention logic with BF-Patch but also records controls:

- self swap;
- reverse swap;
- random pair swap.

Validators require these control cells for full trace-latent gates.

Primary scalar:

```text
continuous_margin_shift
```

Important secondary scalar:

```text
swap_answer_transfer_rate
```

Compatibility scalar:

```text
swap_margin_shift
```

### BF Usage Diagnostics Layer

`pipeline/stats/bf_usage_diagnostics.py` adds an offline diagnostic layer for BF-Patch and BF-Swap artifacts. It reads `cells[*].records` and, for BF-Swap, `control_cells[*].records`; it does not run model inference. The analyzer reports:

- directional margin accuracy: `P(continuous_margin_shift > 0)`;
- signed and absolute continuous margin shift;
- clean-margin boundary bins: `near |clean_margin| < 0.1`, `medium < 0.5`, and `far >= 0.5`;
- agreement between binary answer transfer and continuous shift direction;
- margin-source stratification, including legacy candidate-sequence logprob rows and parsed-answer fallback rows;
- BF-Swap control-normalized signed and absolute effects against self, reverse, and random-pair controls.

`tools/summarize_bf_usage_diagnostics.py` writes the diagnostic artifact bundle:

```text
runs/bf_usage_diagnostics_w18_w16/bf_usage_diagnostics.json
runs/bf_usage_diagnostics_w18_w16/bf_usage_diagnostics_summary.csv
runs/bf_usage_diagnostics_w18_w16/bf_usage_diagnostics_records.csv
runs/bf_usage_diagnostics_w18_w16/bf_usage_diagnostics_boundary_bins.csv
runs/bf_usage_diagnostics_w18_w16/bf_usage_diagnostics_margin_sources.csv
runs/bf_usage_diagnostics_w18_w16/bf_usage_diagnostics_control_normalized.csv
docs/validation_report_bf_usage_diagnostics.md
docs/figures/bf_usage_diagnostics/
```

This layer resolves the W18 interpretation issue where transfer rates are nonzero but `continuous_margin_shift` is close to zero. In W18 standard SPD BF artifacts, continuous margin coverage is 100%, so the null signed shift is not caused by missing scores. The diagnostic shows nonzero absolute movement but weak or wrong-direction signed movement, plus little random-control specificity for BF-Swap. In W16 trace-latent BF n=500 artifacts, the analyzer marks rows as `parsed_answer_fallback_legacy` and reports zero continuous-margin coverage, preventing those rows from being silently pooled into Main continuous-margin claims.

### BF-Conf: `bf_conf_calibrated_progression`

This metric tracks answer confidence progression. In standard mode it uses layerwise internal curves and can include text-only controls. In trace-latent mode it reads answer-logit curves from captured LVR generation states.

The implementation includes a low-memory image-budget override for high-resolution spotlight runs. `config.main_vstar_n191.yaml` keeps PF-A/PF-B at the 1024px audit budget, but sets BF-Conf to `max_image_side=512`, `max_image_pixels=262144`, and `text_only_control=false`. The metric also captures layerwise answer logits in a single image+text forward where possible, avoiding the 1024px OOM path seen in the first V*Bench pilot on 24GB GPUs. This makes BF-Conf an auxiliary W20 trajectory readout; it does not change the primary PF-A/PF-B localization budget.

Primary scalar:

```text
gold_logit_slope
```

### CF-Stage: `cf_stage_decay`

This metric estimates retention by comparing early, middle, and late stages under perturbations. In standard mode the readout is usually PF-style. In trace-latent mode the readout comes from captured generation hidden states.

Primary scalar:

```text
late_delta
```

### W3-W6 LVR latent patch metric

`pipeline/metrics/v2/lvr_latent_patch_answer_transfer.py` implements the earlier W3-W6 hidden-feedback patch experiments. It captures LVR source states and patches target generation at selected steps. It supports:

- last-step patch;
- per-step sweep with `patch_steps=["each"]`;
- best-step reduction;
- step-transfer AUC.

This is the code behind W3, W4, W5, and W6.

### W13 Monet latent patch metric

`pipeline/metrics/v2/monet_latent_patch_answer_transfer.py` implements the Monet causal range gate. It uses `MonetQwenAdapter.capture_latent_state` to capture `ce_patch_pos` and `ce_patch_vec` from Monet's official Transformers `latent_mode`, then scores constrained candidates after injecting source vectors into target latent positions.

The metric explicitly records:

```text
latent_mode_path = transformers_ce_patch_vec
vllm_scheduler_native = false
```

This prevents the result from being mislabeled as Monet modified-vLLM runtime tracing.

### Output accuracy sanity

`pipeline/metrics/v2/output_accuracy_sanity.py` calls `wrapper.generate()` and compares normalized predictions to normalized answers. It is intended for T4 VSI-Bench and optional sanity runs; it is not causal evidence.

The metric now supports structured numeric answers emitted by `tools/prepare_vsi_bench_hf.py`, for example:

```json
{"answer": "2.1", "numeric_value": 2.1, "abs_tolerance": null, "rel_tolerance": null, "unit": null}
```

For numeric targets, `answer_hit()` extracts numbers from the model response and checks absolute or relative tolerance when provided. Otherwise it falls back to normalized text containment. The reduction records both `n` and `n_total`, so T4 validators can distinguish valid scored samples from model/runtime errors.

`tools/merge_vsi_accuracy.py` merges separate T4 accuracy runs into `vsi_accuracy_summary.json`, copies the metric envelopes into a merged directory, writes `summary_with_ci.json`, and validates that the required three models are present with finite accuracy values in `[0, 1]`, adequate sample count, and acceptable error ratio.

The completed 32-frame T4 artifact is `runs/main_vsi_accuracy_scannetpp_32f_paper/merged`: Qwen2.5-VL-3B accuracy `0.196` with 95% bootstrap CI `[0.172, 0.222]`, Qwen2.5-VL-7B accuracy `0.191` with CI `[0.168, 0.215]`, and LVR-7B accuracy `0.110` with CI `[0.091, 0.130]`; all three runs score `n=1000` with `n_error=0` and merged sanity status `pass`.

## 6. Analysis, CI, and Visualization

`pipeline/analysis.py` writes generic metric plots and bootstrap summaries.

Key behavior:

- `plot_generic_metric_results` creates per-metric plots when payloads expose curves, layers, families, or trace-latent cells.
- `plot_trace_latent_metric_matrix` creates heatmaps for trace-latent scalar reductions.
- `build_summary_with_ci` extracts numeric per-sample reductions and computes bootstrap confidence intervals through `pipeline.stats.bootstrap.paired_bootstrap`.
- Results are written to `summary_with_ci.json`, `summary.json`, `metric_results_summary.json`, and `metric_plots/`.
- For standard non-trace BF-Patch/BF-Swap payloads, `build_summary_with_ci` aliases the continuous candidate-sequence margin into `continuous_margin_shift` if the artifact predates that explicit field. Trace-latent payloads must provide `continuous_margin_shift` themselves.

The n=500 LVR trace-latent run produced:

```text
runs/w16_lvr_trace_latent_spd_n500_sharded/merged/summary_with_ci.json
runs/w16_lvr_trace_latent_spd_n500_sharded/merged/metric_plots/w16_trace_latent_n500_primary_heatmap.png
```

The W17 `n=1000` light run produced:

```text
runs/w17_lvr_trace_latent_spd_n1000_light/merged/summary_with_ci.json
runs/w17_lvr_trace_latent_spd_n1000_light/merged/mixed_effects_summary.json
runs/w17_lvr_trace_latent_spd_n1000_light/merged/metric_plots/
```

`pipeline/stats/mixed_effects.py` now writes a minimal Main-readiness regression artifact. It records the intended mixed-effects formula but implements a dependency-light fixed-effect fallback, `value ~ model + task`, so merged matrices can produce `mixed_effects_summary.json` offline. A full random-effect model remains a future paper-polish task rather than a completed claim.

The W18 full local matrix produced:

```text
runs/w18_main_matrix_full_local/merged/summary_with_ci.json
runs/w18_main_matrix_full_local/merged/mixed_effects_summary.json
runs/w18_main_matrix_full_local/merged/w18_primary_results.csv
docs/validation_report_w18_full_local_matrix.md
docs/figures/w18_full_local_matrix/
```

`tools/summarize_w18_results.py` reads the merged W18 `summary_with_ci.json`, extracts the six preregistered primary scalars, writes a CSV copy of the primary rows, generates per-metric line plots with bootstrap intervals, creates a combined panel figure, and writes the committed Markdown report. It also reads local `prepare_stats.json` files to make W18 data-staging boundaries explicit, including Maze being full-local `n=500` and BLINK being weak-oracle `n=1000`. T4 VSI is summarized separately through `tools/merge_vsi_accuracy.py` because it is an output-accuracy sanity task rather than a causal metric matrix row.

`tools/summarize_vstar_results.py` reads `runs/w20_spotlight_vstar_n191_final/merged/summary_with_ci.json`, extracts the four W20 primary spotlight scalars, writes `runs/w20_spotlight_vstar_n191_final/merged/w20_vstar_primary_results.csv`, generates the W20 V*Bench figures, and writes `docs/validation_report_w20_vstar_spotlight.md`.

W20 report artifacts:

```text
docs/validation_report_w20_vstar_spotlight.md
docs/figures/w20_vstar_spotlight/w20_vstar_primary_metric_lines.png
docs/figures/w20_vstar_spotlight/w20_vstar_vs_spd_availability.png
runs/w20_spotlight_vstar_n191_final/merged/w20_vstar_primary_results.csv
```

## 7. Sharding and GPU Use

`tools/run_and_hold.sh` holds the selected GPUs and launches the requested command. The required invocation style is:

```bash
bash tools/run_and_hold.sh 0,1,2,3 <experiment command>
```

`tools/launch_main_matrix.sh` is the reusable matrix launcher. It accepts comma-separated configs, models, metric groups, GPU IDs, and a run root. It round-robins jobs over visible GPUs. Each job internally runs:

```bash
CUDA_VISIBLE_DEVICES=<gpu> ./venv/bin/python run_all.py ... --device cuda:0 --no-analysis
```

This pattern is correct because each child process sees only one physical GPU as `cuda:0`. After all shards finish, `tools/merge_main_matrix.py` merges metric envelopes into `RUN_ROOT/merged`, copies a representative `config_snapshot.yaml`, reruns analysis, writes `mixed_effects_summary.json`, and reruns merged sanity with the source validation settings where available.

The W18 run used:

```bash
PYTHON_BIN=./venv/bin/python MPLCONFIGDIR=/tmp/matplotlib-lvr-eval HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.main_spd_n1000.yaml,config.main_maze_full500.yaml,config.main_blink_n1000.yaml \
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \
  --metrics all \
  --gpus 0,1,2,3 \
  --run-root runs/w18_main_matrix_full_local
```

The W20 V*Bench spotlight used the same local GPU-hold pattern:

```bash
PYTHON_BIN=./venv/bin/python MPLCONFIGDIR=/tmp/matplotlib-lvr-eval HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.main_vstar_n191.yaml \
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \
  --metrics all \
  --gpus 0,1,2,3 \
  --run-root runs/w20_spotlight_vstar_n191

PYTHON_BIN=./venv/bin/python MPLCONFIGDIR=/tmp/matplotlib-lvr-eval HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.main_vstar_n191.yaml \
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b \
  --metrics bf_conf_calibrated_progression,pf_b_patch_alignment \
  --gpus 0,1,2,3 \
  --run-root runs/w20_spotlight_vstar_n191_rerun_bfconf_pfb
```

The final merged W20 root is `runs/w20_spotlight_vstar_n191_final/merged`. It combines PF-A/CF-Stage from the first run with BF-Conf/PF-B from the rerun after the BF-Conf 512px low-memory fix.

## 8. Validators and Error Checks

The main validators are:

- `tools/validate_trace_latent_gate.py`: requires all expected trace-latent metrics, `trace_latent=True`, primary scalars, enough instrumented trace records, patch successes for BF metrics, control cells for BF-Swap, CI rows, sanity pass, trace-latent plots, transfer-rate CI rows, and acceptable parsed-fallback margin ratio for new runs. It can also require generation-score diagnostic summaries with `--require-score-diagnostics`.
- `tools/validate_trace_margin_quality.py`: checks patch metric `margin_source` provenance directly, reports generation-score failure reason counts, and fails when parsed-answer fallback exceeds the configured threshold.
- `tools/validate_main_paper_readiness.py`: checks manifest evidence-level discipline, `summary_with_ci.json`, `mixed_effects_summary.json`, and optionally trace margin quality. It has three strictness levels: `artifact_contract`, `full_main_matrix`, and `paper_ready`.
- `tools/validate_main_matrix.py`: checks task/model/metric coverage, sample thresholds, BF pair thresholds, and `summary_with_ci.json`.
- `tools/validate_monet_latent.py`: checks Monet metric ID/model ID, latent mode path, `vllm_scheduler_native=false`, patch counts, captured state counts, shape matches, prereg lock, sanity pass, and CI rows.
- `tools/validate_findings_gate.py`: checks the complete Findings evidence package, including true LVR latent gates, W7 SPD, and W9 Maze.

Sanity checks for v2 metrics live in `pipeline/sanity/v2.py`. They enforce finite reductions, minimum samples, mask-oracle policy, DINO policy, patch-cell success, error ratios, trace records, control cells, curve lengths, and output-accuracy prediction counts.

## 9. Current Completed Runs and Validation Status

Completed and validated evidence:

| Stage | Run | Validation status |
|---|---|---|
| W3 LVR last-step patch | `runs/w3_lvr_latent_patch_n50` | pass |
| W4 LVR step sweep | `runs/w4_lvr_latent_stepsweep_n50_s8` | pass |
| W5 LVR capacity sweep | `runs/w5_lvr_capacity_s*_n50` | pass; no monotonic claim |
| W6 LVR best-step replication | `runs/w6_lvr_beststep4_s8_n50` | pass |
| W7 SPD n=200 matrix | `runs/w7_spd_scale_m0_m1_m2_n200` | 18/18 sanity pass |
| W9 Maze n=200 matrix | `runs/w9_maze_findings_m0_m1_m2_n200_bbox` | Findings gate pass |
| W13 Monet latent gate | `runs/w13_monet_latent_patch_n50` | Monet validator pass |
| W14-W15 LVR trace-latent n=50 | `runs/w14_w15_lvr_trace_latent_n50_v2` | trace validator pass |
| W16 LVR trace-latent n=500 | `runs/w16_lvr_trace_latent_spd_n500_sharded/merged` | trace validator pass |
| W17 LVR trace-latent n=1000 light | `runs/w17_lvr_trace_latent_spd_n1000_light/merged` | trace validator + main-readiness validator pass |
| W16 T1/T2/T3 n=100 matrix | `runs/w16_main_matrix_t1_t2_t3_n100/merged` | main matrix validator pass |
| W18 T1/T2/T3 full local matrix | `runs/w18_main_matrix_full_local/merged` | main matrix + full-main readiness validators pass |
| W20 V*Bench/VStar spotlight n=191 | `runs/w20_spotlight_vstar_n191_final/merged` | main matrix validator pass; sanity pass 12/12 |

Recent code checks already passed in this repo state:

```text
py_compile: pass
smoke_test.py: pass
validate_trace_latent_gate.py on n=500: pass
validate_trace_latent_gate.py on n=1000 light: pass
validate_trace_margin_quality.py fixture with score diagnostics: pass
validate_main_paper_readiness.py on n=1000 light: pass
validate_main_matrix.py on T1/T2/T3 n=100: pass
validate_main_matrix.py on W18 full local matrix: pass, rows=42, ci_rows=180
validate_main_paper_readiness.py --mode full_main_matrix on W18: pass
validate_main_matrix.py on W20 V*Bench spotlight: pass, rows=12, ci_rows=51
DINO PF-B artifact contract on W21 pilot: pass, run_dirs=6, summary_ci_rows=54, sanity_files=7
validate_monet_latent.py on W13: pass
```

## 10. Correctness Boundaries

No syntax or validator errors are known on the completed experiment paths. The following are intentional boundaries rather than hidden implementation errors:

1. `pipeline/stats/mixed_effects.py` implements a fixed-effect fallback, not a full random-effect mixed model. Use it as a readiness artifact, not as final statistical polish.
2. Monet scheduler-native tracing is not implemented. W13 uses official Transformers `latent_mode` with `ce_patch_vec`, not modified-vLLM generation tracing.
3. PF-B DINO alignment is implemented as an optional external sensitivity check and completed in the W21 pilot. Current Main PF-B claims remain native-alignment claims; W21 DINO `dino_region_selectivity` is near zero with confidence intervals crossing zero on SPD-Faith n=100 and V*Bench n=191.
4. T4 VSI-Bench visual data is staged for the official `scannetpp` subset and the 32-frame three-model `output_accuracy_sanity` merge passes sanity. It remains output-accuracy sanity only, not causal evidence or native-video VSI-Bench performance.
5. BLINK now runs at `n=1000`, but all W18 BLINK PF rows are weak-oracle center fallback because reliable bboxes are absent locally.
6. Query-span matrices and trace-latent LVR gates answer different questions. W7/W9/W16/W18 main matrices should not be described as entirely real latent intervention evidence.
7. BF-Patch and BF-Swap margin shifts in the n=500 trace-latent run are near zero with confidence intervals crossing zero; answer-transfer rates are secondary but stable around 0.462. New patch/swap runs should use the continuous margin provenance checks and transfer-rate CI rows. The W17 `n=1000` light run intentionally disables BF-Patch/BF-Swap, so margin provenance is not expected in that artifact.
8. V*Bench support is implemented and validated as `source_type=vstar` with mandatory bbox metadata and high-resolution image preservation. W20 completed all 191 public samples from the full HF snapshot. It is still a small-n spotlight result, not a T1/T2/T3 Main matrix entry, and its headline interpretation is mixed: LVR is strongest on PF-B native alignment but not on PF-A bbox-corruption selectivity.

## 11. Implementation Conclusion

The current codebase correctly supports the completed audit path:

- paired and non-paired data loading;
- Qwen/LVR/Monet model loading;
- real LVR generation-time sparse trace capture;
- LVR hidden-feedback patching;
- Monet Transformers latent-state patching;
- six primary v2 metrics;
- output accuracy sanity;
- four-GPU sharded execution;
- merged bootstrap summaries and plots;
- strict validators for completed gates.

The remaining work is not to fix a known broken path, but to implement Monet modified-vLLM tracing as a separate spike, decide whether to scale the W21 DINO sensitivity check beyond the pilot, and replace the fixed-effect fallback with final sample-level mixed-effects statistics. V*Bench/VStar is now runnable and reported as W20 spotlight evidence; T4 VSI now has a completed 32-frame output-accuracy sanity artifact.
