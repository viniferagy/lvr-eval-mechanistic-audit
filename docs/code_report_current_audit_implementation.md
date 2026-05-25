# Code Report: Current Latent Visual Faithfulness Audit Implementation

Date: 2026-05-25

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
- LVR models are ordered last and `wait_for_lvr_model_if_needed` checks for incomplete model downloads before loading.
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
- `pipeline/data_blink.py`: loads BLINK manifest rows and appends answer choices to the prompt when present. If no bboxes exist, it records `weak_oracle=True`; PF validators can either fail or warn depending on config.
- `pipeline/data_vsi.py`: loads VSI-Bench-style static frame-grid image records for output accuracy sanity. The current public HF table available locally has question metadata but no image/frame-grid payloads, so real T4 runs require local visual files.

The loader layer is structurally sound for completed runs. The known data limitation is external: VSI-Bench images are absent locally, and full BLINK staging beyond Art_Style n=100 remains future work.

## 3. Model Wrappers and Adapters

`pipeline/model_utils.py` defines `VLMWrapper`, which stores the loaded model, processor, adapter, decoder layers, final norm, LM head, device, image token ID, and config. Metrics use the wrapper instead of depending on one model class.

Adapters live under `pipeline/adapters/`:

- `qwen_vl.py`: standard Qwen2.5-VL/Qwen3-VL adapter for baseline models.
- `lvr_qwen.py`: official LVR model adapter. It supports teacher-forced LVR span audits and a legacy approximate `generate_with_trace` path based on generated token IDs.
- `lvr_qwen_traced.py`: traced LVR adapter used by W14-W16 trace-latent experiments. This is the real generation-time path for current LVR trace-latent evidence.
- `monet_qwen.py`: Monet adapter for the official Transformers latent-mode path. It intentionally does not implement modified-vLLM scheduler-native generation tracing.

Important adapter distinction:

`LVRQwenAdapter.generate_with_trace` in `pipeline/adapters/lvr_qwen.py` returns `trace_quality="approx_from_generated_token_ids"`. This is not sufficient for W14-W16 trace-latent gates. The W14-W16 configs use `arch: lvr_qwen2_5_vl_traced`, which routes to `TracedLVRQwenAdapter` and `TraceRecorder`. Validators require `trace_quality="instrumented_sparse_v0"`, so accidental fallback to the approximate path is caught.

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

For trace-latent patching, the forward pre-hook can replace `last_position_hidden_state` with a source tensor. Shape policy is configurable as `strict` or `slice`; W16 uses strict shape matching. The recorder stores captured tensors as CPU float32 by default to avoid holding unnecessary GPU memory after generation.

The trace summary sets:

```text
trace_quality = instrumented_sparse_v0
```

This is the quality marker enforced by `tools/validate_trace_latent_gate.py`.

## 5. Metrics

Metrics are registered in `pipeline/metrics/registry.py`. The current primary v2 metric files are under `pipeline/metrics/v2/`.

### PF-A: `pf_a_corruption_selectivity`

This metric compares model-internal changes after masking relevant, irrelevant, and random visual regions. In standard/query-span mode it uses prompt-side internal readouts. In trace-latent mode it calls `trace_latent.generate_trace` on clean and masked images, then computes latent-state distances over captured generation states.

Primary scalar:

```text
selectivity = irrelevant_delta - relevant_delta
```

A negative value therefore means relevant-region corruption moved the latent state more than irrelevant-region corruption under the current sign convention.

### PF-B: `pf_b_patch_alignment`

This metric reports native visual alignment between hidden/latent representations and region-derived visual signals. In the current completed runs `use_dino=false`, so DINO alignment is not claimed. The code records DINO policy metadata and validators check that the requested policy is satisfied.

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

For generation-score margins, the code attempts token-level scoring from generation scores. When scores are unavailable or incomplete, trace-latent sanity can warn and use parsed constrained-answer fallback for transfer records. This is why W16 reports both margin shift and parsed answer-transfer rate.

Primary scalar:

```text
logprob_margin_shift
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
swap_margin_shift
```

Important secondary scalar:

```text
swap_answer_transfer_rate
```

### BF-Conf: `bf_conf_calibrated_progression`

This metric tracks answer confidence progression. In standard mode it uses layerwise internal curves and can include text-only controls. In trace-latent mode it reads answer-logit curves from captured LVR generation states.

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

## 6. Analysis, CI, and Visualization

`pipeline/analysis.py` writes generic metric plots and bootstrap summaries.

Key behavior:

- `plot_generic_metric_results` creates per-metric plots when payloads expose curves, layers, families, or trace-latent cells.
- `plot_trace_latent_metric_matrix` creates heatmaps for trace-latent scalar reductions.
- `build_summary_with_ci` extracts numeric per-sample reductions and computes bootstrap confidence intervals through `pipeline.stats.bootstrap.paired_bootstrap`.
- Results are written to `summary_with_ci.json`, `summary.json`, `metric_results_summary.json`, and `metric_plots/`.

The n=500 LVR trace-latent run produced:

```text
runs/w16_lvr_trace_latent_spd_n500_sharded/merged/summary_with_ci.json
runs/w16_lvr_trace_latent_spd_n500_sharded/merged/metric_plots/w16_trace_latent_n500_primary_heatmap.png
```

`pipeline/stats/mixed_effects.py` is still a placeholder and raises `NotImplementedError`. This is not a bug for completed gates because no completed validator invokes mixed-effects analysis. It is a Main-paper remaining task.

## 7. Sharding and GPU Use

`tools/run_and_hold.sh` holds the selected GPUs and launches the requested command. The required invocation style is:

```bash
bash tools/run_and_hold.sh 0,1,2,3 <experiment command>
```

`tools/launch_main_matrix.sh` is the reusable matrix launcher. It accepts comma-separated configs, models, metric groups, GPU IDs, and a run root. It round-robins jobs over visible GPUs. Each job internally runs:

```bash
CUDA_VISIBLE_DEVICES=<gpu> ./venv/bin/python run_all.py ... --device cuda:0 --no-analysis
```

This pattern is correct because each child process sees only one physical GPU as `cuda:0`. After all shards finish, `tools/merge_main_matrix.py` merges metric envelopes into `RUN_ROOT/merged` and reruns analysis.

One implementation caveat: the default merge sanity config is permissive because it must handle heterogeneous tasks. For the n=500 trace-latent result, merged sanity was rewritten with the real trace-latent config before final validation. The gate validator then passed, which is the stronger check.

## 8. Validators and Error Checks

The main validators are:

- `tools/validate_trace_latent_gate.py`: requires all expected trace-latent metrics, `trace_latent=True`, primary scalars, enough instrumented trace records, patch successes for BF metrics, control cells for BF-Swap, CI rows, sanity pass, and trace-latent plots.
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
| W16 T1/T2/T3 n=100 matrix | `runs/w16_main_matrix_t1_t2_t3_n100/merged` | main matrix validator pass |

Recent code checks already passed in this repo state:

```text
py_compile: pass
smoke_test.py: pass
validate_trace_latent_gate.py on n=500: pass
validate_main_matrix.py on T1/T2/T3 n=100: pass
validate_monet_latent.py on W13: pass
```

## 10. Correctness Boundaries

No syntax or validator errors are known on the completed experiment paths. The following are intentional boundaries rather than hidden implementation errors:

1. `pipeline/stats/mixed_effects.py` is not implemented. It is planned for the full Main matrix and should not be claimed as complete.
2. Monet scheduler-native tracing is not implemented. W13 uses official Transformers `latent_mode` with `ce_patch_vec`, not modified-vLLM generation tracing.
3. PF-B DINO alignment is not implemented in completed runs. Current PF-B claims are native-alignment claims.
4. T4 VSI-Bench cannot run until local images or frame-grid images are staged.
5. BLINK Art_Style n=100 has weak-oracle center fallback for PF metrics because reliable bboxes are absent locally.
6. Query-span matrices and trace-latent LVR gates answer different questions. W7/W9/W16 main matrix checkpoint should not be described as entirely real latent intervention evidence.
7. BF-Patch and BF-Swap margin shifts in the n=500 trace-latent run are near zero with confidence intervals crossing zero; answer-transfer rates are secondary but stable around 0.462.

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

The remaining work is not to fix a known broken path, but to extend the verified paths to larger samples, stronger cross-task coverage, Monet modified-vLLM tracing, DINO alignment, and mixed-effects statistics.

