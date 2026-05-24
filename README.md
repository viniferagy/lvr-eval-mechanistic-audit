# LVR-Eval Mechanistic Audit

`lvr-eval-mechanistic-audit` is a reproducible audit scaffold for VLM/LVR mechanistic experiments. It compares Qwen baselines and LVR-style models through a shared runner, explicit adapter spans, preregistered metric metadata, v2 causal metric entry points, sanity gates, and analysis artifacts.

## Current Status

This repository is currently at:

```text
Engineering stage: W8 evidence package passed
Scientific stage: true LVR hidden-feedback intervention + capacity/localization gates, not paper-grade causal evidence
```

The W2 gate demonstrates that the infrastructure can run end to end on real SPD-Faith paired data and real GPU models. The W3 gate adds true inference-time LVR hidden-feedback patching at `n=50`. W4 localizes which hidden-feedback steps drive that transfer at `n=50`, `lvr_steps=8`. W5 sweeps latent feedback budget, W6 replicates the best-step intervention, W7 scales SPD regression evidence to `n=200`, and W8 packages the evidence for review. These gates are stronger than the original proxy scaffold, but they are still not final paper-scale evidence.

The SPD range runs treat `lvr_7b` as an LVR-weight model under query-span paired intervention, not as a true continuous latent-state intervention. The true latent-state evidence comes from the W3-W6 traced LVR runs.

## Architecture

```text
lvr-eval-mechanistic-audit/
├── config.yaml
├── config.trace_v2.yaml
├── config.spd_faith.range.yaml
├── config.spd_faith.week7_scale.yaml
├── config.lvr_latent_patch.range.yaml
├── config.lvr_latent_patch.stepsweep.yaml
├── config.lvr_latent_patch.stepsweep_s2.yaml
├── config.lvr_latent_patch.stepsweep_s4.yaml
├── config.lvr_latent_patch.stepsweep_s16.yaml
├── config.lvr_latent_patch.beststep_s8.yaml
├── prereg/
│   └── manifest.yaml
├── run_all.py
├── merge_and_analyze.py
├── smoke_test.py
├── tools/
│   ├── prepare_spd_faith_hf.py
│   ├── validate_spd_range.py
│   ├── validate_capacity_sweep.py
│   ├── validate_w3_latent.py
│   ├── validate_w4_stepsweep.py
│   ├── build_evidence_pack.py
│   ├── check_lvr_env.py
│   ├── compare_capacity_sweep.py
│   ├── run_and_hold.sh
│   └── hold_gpu.py
├── docs/
│   ├── validation_report.md
│   ├── validation_report_w2.md
│   ├── validation_report_w3.md
│   ├── validation_report_w4.md
│   ├── validation_report_w5_w8.md
│   └── evidence_pack_w5_w8.md
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
    │   │   ├── cf_stage_decay.py
    │   │   └── lvr_latent_patch_answer_transfer.py
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

4. **W3-W6 true LVR hidden-feedback intervention**

   `lvr_latent_patch_answer_transfer` patches real `output_last_position_hidden_state` tensors captured from the LVR generation loop.

   | mode | config | primary scalar | status |
   |---|---|---|---|
   | W3 last-step gate | `config.lvr_latent_patch.range.yaml` | `latent_answer_transfer_rate` | passed at n=50 |
   | W4 step sweep | `config.lvr_latent_patch.stepsweep.yaml` | `best_step_transfer_rate`, `step_transfer_auc` | passed at n=50, lvr_steps=8 |
   | W5 capacity sweep | `config.lvr_latent_patch.stepsweep_s{2,4,16}.yaml` + W4 s8 config | `best_step_transfer_rate`, `step_transfer_auc` | passed at n=50 for s2/s4/s8/s16 |
   | W6 best-step replication | `config.lvr_latent_patch.beststep_s8.yaml` | `latent_answer_transfer_rate` | passed at n=50, step 4 |

5. **W7 SPD regression scale-up**

   `config.spd_faith.week7_scale.yaml` reruns the six W2 v2 metrics on 200 paired SPD-Faith samples across `qwen2_5_vl_3b`, `qwen2_5_vl_7b`, and `lvr_7b`. This is scale/regression evidence, not true latent-state intervention evidence.

## Data Sources

Supported `data.source_type` values include:

- `jsonl`
- `hf`
- `lvr_json`
- `spd_faith`
- `maze`

`spd_faith` is the paired counterfactual entry point used for the W2 range gate. It expects fields for clean/counterfactual images, clean/counterfactual answers, `paired_id`, and either bbox or region-mask oracle metadata.

Check the local LVR model and external `VincentLeebang/lvr` checkout without loading model weights:

```bash
./venv/bin/python tools/check_lvr_env.py \
  --config config.lvr_latent_patch.stepsweep.yaml
```

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

The W3-W6 latent patch configs also set `trace_v2.required=true`, `trace_v2.forbid_fallback=true`, and `lvr_latent_patch.patch_shape_policy="strict"`. Shape mismatches at the hidden-feedback injection site fail by default instead of silently slicing tensors.

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

## W3/W4 Reproduction

W3 latent patch gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.range.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w3_lvr_latent_patch_n50

./venv/bin/python tools/validate_w3_latent.py \
  runs/w3_lvr_latent_patch_n50 \
  --min-pairs 50
```

W4 latent step-localization gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w4_lvr_latent_stepsweep_n50_s8

./venv/bin/python tools/validate_w4_stepsweep.py \
  runs/w4_lvr_latent_stepsweep_n50_s8 \
  --min-pairs 50 \
  --min-steps 2
```

See [docs/validation_report_w3.md](/home/pengguangyue/workspace/proj/lvr-eval-mechanistic-audit/docs/validation_report_w3.md) and [docs/validation_report_w4.md](/home/pengguangyue/workspace/proj/lvr-eval-mechanistic-audit/docs/validation_report_w4.md).

## W5-W8 Reproduction

W5 latent capacity sweep:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep_s2.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w5_lvr_capacity_s2_n50

bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep_s4.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w5_lvr_capacity_s4_n50

bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w5_lvr_capacity_s8_n50

bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.stepsweep_s16.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w5_lvr_capacity_s16_n50

./venv/bin/python tools/validate_capacity_sweep.py \
  runs/w5_lvr_capacity_s2_n50 \
  runs/w5_lvr_capacity_s4_n50 \
  runs/w5_lvr_capacity_s8_n50 \
  runs/w5_lvr_capacity_s16_n50 \
  --min-pairs 50 \
  --min-steps 1 \
  --expected-steps 2 4 6 10

./venv/bin/python tools/compare_capacity_sweep.py \
  runs/w5_lvr_capacity_s2_n50 \
  runs/w5_lvr_capacity_s4_n50 \
  runs/w5_lvr_capacity_s8_n50 \
  runs/w5_lvr_capacity_s16_n50
```

W6 best-step replication:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_latent_patch.beststep_s8.yaml \
  --models lvr_7b \
  --only lvr_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w6_lvr_beststep4_s8_n50

./venv/bin/python tools/validate_w3_latent.py \
  runs/w6_lvr_beststep4_s8_n50 \
  --min-pairs 50
```

W7 SPD regression scale-up:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.spd_faith.week7_scale.yaml \
  --models qwen2_5_vl_3b qwen2_5_vl_7b lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w7_spd_scale_m0_m1_m2_n200

./venv/bin/python tools/validate_spd_range.py \
  runs/w7_spd_scale_m0_m1_m2_n200 \
  --min-pairs 200
```

W8 evidence pack:

```bash
./venv/bin/python tools/build_evidence_pack.py
```

The builder is fail-fast by default: missing metric JSON, missing or non-pass sanity summaries, missing W7 metric files, missing W7 primary CI rows, and invalid W5 capacity sweep artifacts abort generation. `--allow-missing` is available only for legacy/local drafting.

Recorded W5-W8 results:

| gate | result |
|---|---|
| W5 capacity sweep | all four budgets passed; best transfer rates s2/s4/s8/s16 = 0.44 / 0.52 / 0.44 / 0.48 |
| W6 best step | step 4 targeted patch passed with `latent_answer_transfer_rate=0.60` |
| W7 SPD scale-up | 18/18 sanity pass, 66 CI rows, paired BF metrics `200/0` success/error per model |
| W8 evidence pack | [docs/evidence_pack_w5_w8.md](/home/pengguangyue/workspace/proj/lvr-eval-mechanistic-audit/docs/evidence_pack_w5_w8.md) |

`prereg/manifest.yaml` is synchronized to `manifest_version: 0.8-w8-evidence-package`; it preserves the six W2 primary metrics and records W5, W6, W7, and W8 as completed gates.

See [docs/validation_report_w5_w8.md](/home/pengguangyue/workspace/proj/lvr-eval-mechanistic-audit/docs/validation_report_w5_w8.md) for exact commands, reductions, and validators.

## Analysis

`summary_with_ci.json` uses bootstrap confidence intervals. For v2 sample-level artifacts it groups derived rows by `paired_id`, falling back to `id`, before bootstrapping group-level values. This prevents multi-layer, multi-bucket, or multi-family derived rows from being counted as independent samples when group keys are present.

Legacy artifacts without group keys keep the old one-sample bootstrap fallback.

## Known Boundaries

- W2-W8 are reproducibility, localization, capacity, and scale-up gates, not a final causal-result package.
- W8 is an engineering/review evidence package, not a claim-finalizing result.
- W5 capacity sweep passed validation, but it does not establish monotonic capacity scaling.
- W6 best-step replication is an `n=50` gate-level replication, not a full replication study.
- `pf_b_patch_alignment` currently reports native attention-proxy alignment; DINO patch correspondence remains optional and disabled for the W2 gate.
- `bf_swap_latent_replacement` now records self-swap, reverse-swap, and random-pair controls, but stronger controlled latent-block protocols are still needed.
- `cf_stage_decay` uses `late_delta` as the primary scalar; `late_retention` can explode when the clean late-stage baseline is near zero and is diagnostic only.
- W3-W6 latent patching is true inference-time LVR hidden-feedback intervention on SPD-Faith constrained `original` / `modified` answers.
- W7 remains query-span regression evidence across Qwen/LVR weights, not true latent-state intervention evidence.
- Broader tasks, larger sample sizes, external LVR paradigms, and layer-level localization remain future work.
