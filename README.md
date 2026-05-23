# LVR-Eval Mechanistic Audit

`lvr-eval-mechanistic-audit` is a reproducible audit scaffold for VLM/LVR mechanistic experiments. It compares Qwen baselines and LVR-style models through a shared runner, explicit adapter spans, preregistered metric metadata, v2 causal metric entry points, sanity gates, and analysis artifacts.

## Current Status

This repository is currently at:

```text
Engineering stage: W2-final-gate / runnable-v0 validation passed
Scientific stage: range-level validation scaffold, not paper-grade causal evidence
```

The W2 gate demonstrates that the infrastructure can run end to end on real SPD-Faith paired data and real GPU models. It does **not** yet justify strong claims about inference-time LVR latent-state causality. In particular, the SPD range run treats `lvr_7b` as an LVR-weight model under query-span paired intervention, not as a true continuous latent-state intervention.

The next scientific milestone is W3/W4: combine trace-v2 hidden-feedback instrumentation with actual latent-state patching/replacement on LVR generation traces.

## Architecture

```text
lvr-eval-mechanistic-audit/
├── config.yaml
├── config.trace_v2.yaml
├── config.spd_faith.range.yaml
├── prereg/
│   └── manifest.yaml
├── run_all.py
├── merge_and_analyze.py
├── smoke_test.py
├── tools/
│   ├── prepare_spd_faith_hf.py
│   ├── validate_spd_range.py
│   ├── run_and_hold.sh
│   └── hold_gpu.py
├── docs/
│   ├── validation_report.md
│   └── validation_report_w2.md
└── pipeline/
    ├── adapters/
    │   ├── base.py
    │   ├── qwen_vl.py
    │   ├── lvr_qwen.py
    │   ├── lvr_qwen_traced.py
    │   ├── probe_catalog.py
    │   ├── spans.py
    │   └── registry.py
    ├── metrics/
    │   ├── legacy/
    │   ├── v2/
    │   │   ├── pf_a_corruption_selectivity.py
    │   │   ├── pf_b_patch_alignment.py
    │   │   ├── bf_patch_answer_transfer.py
    │   │   ├── bf_swap_latent_replacement.py
    │   │   ├── bf_conf_calibrated_progression.py
    │   │   └── cf_stage_decay.py
    │   ├── bf3_confidence_progression.py
    │   ├── pf3_attention_distance.py
    │   ├── bf1_latent_ablation.py
    │   ├── bf1_layer_ablation.py
    │   ├── cf2_pf_decay_curve.py
    │   ├── lvr_generation_trace.py
    │   └── registry.py
    ├── sanity/
    │   ├── report.py
    │   └── v2.py
    ├── stats/
    │   ├── bootstrap.py
    │   └── mixed_effects.py
    ├── data.py
    ├── data_spd_faith.py
    ├── data_maze.py
    ├── preregistration.py
    ├── corruptions.py
    ├── internal_metrics.py
    ├── ablation.py
    ├── degradation.py
    ├── results.py
    └── analysis.py
```

`run_all.py` reads the config, loads probe samples, resolves adapters and metrics, writes `config_snapshot.yaml` and `prereg.lock.json`, runs sanity checks, and produces analysis artifacts such as `summary.json` and `summary_with_ci.json`.

## Metric Layers

There are three metric layers.

1. **Legacy regression references**

   The `_legacy` metrics freeze the old proxy behavior for regression testing only. They are useful for continuity and sanity checks, but they do not support the main causal claim.

   Aliases include `bf3_legacy`, `pf3_legacy`, `bf1_legacy`, `bf1_layer_legacy`, `cf2_legacy`, and `lvr_trace_legacy`.

2. **Base proxy metrics**

   These are the original runnable metrics: BF-3 confidence progression, PF-3 attention distance, BF-1 ablation, CF-2/PF decay, and LVR generation trace.

3. **W2 v2 causal-metric scaffold**

   | metric id | primary scalar | W2 status |
   |---|---|---|
   | `pf_a_corruption_selectivity` | `selectivity` | runnable-v0 |
   | `pf_b_patch_alignment` | `native_alignment` | runnable native-attention proxy |
   | `bf_patch_answer_transfer` | `logprob_margin_shift` | runnable-v0 sequence logprob scoring |
   | `bf_swap_latent_replacement` | `swap_margin_shift` | runnable-v0 with self/reverse/random controls |
   | `bf_conf_calibrated_progression` | `gold_logit_slope` | runnable-v0 |
   | `cf_stage_decay` | `late_delta` | runnable-v0; `late_retention` is diagnostic only |

W2 v2 metrics are validated as runnable-v0 gates. They are not yet full paper-grade causal evidence.

## Data Sources

Supported `data.source_type` values include:

- `jsonl`
- `hf`
- `lvr_json`
- `spd_faith`
- `maze`

`spd_faith` is the paired counterfactual entry point used for the W2 range gate. It expects fields for clean/counterfactual images, clean/counterfactual answers, `paired_id`, and either bbox or region-mask oracle metadata.

Prepare the canonical local SPD-Faith manifest from the public Hugging Face dataset:

```bash
./venv/bin/python tools/prepare_spd_faith_hf.py \
  --out data/spd_faith_hf
```

For a tiny conversion smoke:

```bash
./venv/bin/python tools/prepare_spd_faith_hf.py \
  --max-per-split 2 \
  --out /tmp/spd_faith_prepare_smoke
```

This writes:

```text
data/spd_faith_hf/manifest.jsonl
data/spd_faith_hf/prepare_stats.json
data/spd_faith_hf/images/
```

## Trace v2

Trace v2 is exposed through the traced adapter arch:

```yaml
models:
  lvr_7b:
    arch: "lvr_qwen2_5_vl_traced"

trace_v2:
  required: true
  forbid_fallback: true
```

`config.trace_v2.yaml` runs the trace gate on LVR JSON samples. The trace metric now hard-fails when trace v2 is required but sparse instrumentation falls back, reports missing modules, or does not return `trace_quality=instrumented_sparse_v0`.

The SPD range config intentionally does not use traced latent-state intervention. It runs paired query-span interventions on SPD-Faith for Qwen/LVR-weight comparison.

## W2 Reproduction

CPU checks:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python -m py_compile \
  run_all.py smoke_test.py merge_and_analyze.py \
  pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py \
  pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py \
  pipeline/adapters/*.py pipeline/stats/*.py tools/*.py

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
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w2_final_spd_range_m0_m1_m2_n50
```

Post-run validation:

```bash
./venv/bin/python tools/validate_spd_range.py \
  runs/w2_final_spd_range_m0_m1_m2_n50 \
  --min-pairs 50
```

See [docs/validation_report_w2.md](/home/pengguangyue/workspace/proj/lvr-eval-mechanistic-audit/docs/validation_report_w2.md) for the recorded W2 gate.

## Analysis

`summary_with_ci.json` uses bootstrap confidence intervals. For v2 sample-level artifacts it groups derived rows by `paired_id`, falling back to `id`, before bootstrapping group-level values. This prevents multi-layer, multi-bucket, or multi-family derived rows from being counted as independent samples when group keys are present.

Legacy artifacts without group keys keep the old one-sample bootstrap fallback.

## Known Boundaries

- W2 is a reproducibility and runnable-v0 gate, not a final causal-result package.
- `pf_b_patch_alignment` currently reports native attention-proxy alignment; DINO patch correspondence remains optional and disabled for the W2 gate.
- `bf_swap_latent_replacement` now records self-swap, reverse-swap, and random-pair controls, but stronger W3/W4 controlled latent-block protocols are still needed.
- `cf_stage_decay` uses `late_delta` as the primary scalar; `late_retention` can explode when the clean late-stage baseline is near zero and is diagnostic only.
- True inference-time LVR latent-state patching remains a future milestone beyond the W2 SPD query-span range gate.
