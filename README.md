# LVR-Eval Mechanistic Audit

`lvr-eval-mechanistic-audit` is a reproducible audit scaffold for VLM/LVR mechanistic experiments. It compares Qwen baselines and LVR-style models through a shared runner, explicit adapter spans, preregistered metric metadata, v2 causal metric entry points, sanity gates, and analysis artifacts.

## Current Status

This repository is currently at:

```text
Engineering stage: W17 LVR trace-latent n=1000 light gate + W16 T1/T2/T3 n=100 matrix passed
Scientific stage: Findings gate passed; true LVR hidden-feedback intervention, SPD/Maze evidence, Monet Transformers latent-mode range evidence, LVR real-trace versions of all six v2 primary metrics at n=500, LVR real-trace light metrics at n=1000, and hundred-scale T1/T2/T3 matrix evidence are present
Main-track next step: expand T1/T2/T3 beyond hundred-scale where local data permits; keep Monet modified-vLLM scheduler-native tracing as a separate high-risk spike
```

The W2 gate demonstrates that the infrastructure can run end to end on real SPD-Faith paired data and real GPU models. The W3 gate adds true inference-time LVR hidden-feedback patching at `n=50`. W4 localizes which hidden-feedback steps drive that transfer at `n=50`, `lvr_steps=8`. W5 sweeps latent feedback budget, W6 replicates the best-step intervention, W7 scales SPD regression evidence to `n=200`, and W8 packages the evidence for review. These gates are stronger than the original proxy scaffold, but they are still not final paper-scale evidence.

The SPD range runs treat `lvr_7b` as an LVR-weight model under query-span paired intervention, not as a true continuous latent-state intervention. The true latent-state evidence comes from the W3-W6 traced LVR runs.

The Findings gate is explicit rather than implicit: `tools/validate_findings_gate.py` requires W3/W4/W6 true latent artifacts, W7 SPD artifacts, and a real Maze Findings run. The current local repository has all required artifacts, including `runs/w9_maze_findings_m0_m1_m2_n200_bbox`.

The Main-track expansion now includes Monet and an LVR trace-latent metric upgrade. W12 established a standard-forward preflight for the public `NOVAglow646/Monet-7B` checkpoint and `NOVAglow646/Monet-SFT-125K` data. W13 adds a causal range gate on real SPD-Faith paired data using Monet's official Transformers latent-mode path (`ce_patch_pos` / `ce_patch_vec`): source counterfactual latent tensors are injected into target latent positions and scored over constrained `original` / `modified` answers. Local W13 run: `runs/w13_monet_latent_patch_n50`, `n=50`, sanity pass, `latent_answer_transfer_rate=0.06` with CI `[0.00, 0.14]`. W14-W15 then upgrades all six LVR v2 primary metrics to an opt-in real generation-trace latent path. W16 scales all six LVR trace-latent metrics to n=500, and W17 scales the lighter PF-A/PF-B/BF-Conf/CF-Stage subset to n=1000. Monet remains causal through Transformers latent mode only; it is still not the modified-vLLM scheduler-native generation loop.

## Why Monet Next

Monet is the W12 second-paradigm choice because it is the lowest-risk path from Findings toward Main:

| Criterion | Why Monet fits |
|---|---|
| Public artifacts | Official `NOVAglow646/Monet-7B` weights, `NOVAglow646/Monet-SFT-125K` data, and source code are available. |
| Backbone control | It uses Qwen2.5-VL-7B, keeping the comparison close to the existing Qwen/LVR stack instead of mixing in a new backbone confound. |
| Distinct latent mechanism | Monet enters latent visual reasoning through `<abs_vis_token>` and the official modified vLLM runner, making it a real paradigm contrast to LVR hidden-feedback. |
| Hook path clarity | The release exposes a customized Transformers model for standard-forward probes and a modified vLLM runner for the future true-latent trace adapter. |
| Dataset utility | Monet-SFT provides a public source for adapter preflight samples before we attempt causal latent intervention. |

Latent Sketchpad remains important as the MazePlanning data source and a possible later model candidate. CrystaL remains a good M3 candidate, but it overlaps more directly with PF-style corruption/alignment ideas and is less immediately clean as the next adapter target. Monet therefore gives the best near-term balance of public availability, same-backbone comparability, and a genuine latent-runtime mechanism.

## Architecture

```text
lvr-eval-mechanistic-audit/
├── config.yaml
├── config.trace_v2.yaml
├── config.spd_faith.range.yaml
├── config.spd_faith.week7_scale.yaml
├── config.findings_spd_scale.yaml
├── config.maze.findings.yaml
├── config.monet.preflight.yaml
├── config.monet_latent_patch.range.yaml
├── config.lvr_trace_latent.w14_w15.yaml
├── config.lvr_trace_latent.spd_n500.yaml
├── config.lvr_trace_latent.spd_n1000_light.yaml
├── config.main_spd_n1000.yaml
├── config.main_maze_n1000.yaml
├── config.main_blink_n1000.yaml
├── config.main_vsi_accuracy.yaml
├── config.main_vstar_n191.yaml
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
│   ├── prepare_maze_planning_hf.py
│   ├── prepare_blink_hf.py
│   ├── prepare_vsi_bench_hf.py
│   ├── prepare_vstar_hf.py
│   ├── prepare_monet_sft_hf.py
│   ├── validate_spd_range.py
│   ├── validate_capacity_sweep.py
│   ├── validate_w3_latent.py
│   ├── validate_w4_stepsweep.py
│   ├── validate_monet_latent.py
│   ├── validate_trace_latent_gate.py
│   ├── validate_trace_margin_quality.py
│   ├── validate_main_matrix.py
│   ├── validate_main_paper_readiness.py
│   ├── launch_main_matrix.sh
│   ├── merge_main_matrix.py
│   ├── build_evidence_pack.py
│   ├── build_findings_pack.py
│   ├── check_monet_env.py
│   ├── check_lvr_env.py
│   ├── compare_capacity_sweep.py
│   ├── validate_findings_gate.py
│   ├── run_and_hold.sh
│   └── hold_gpu.py
├── docs/
│   ├── validation_report.md
│   ├── validation_report_w2.md
│   ├── validation_report_w3.md
│   ├── validation_report_w4.md
│   ├── validation_report_w5_w8.md
│   ├── validation_report_w14_w15_trace_latent.md
│   ├── validation_report_w16_main_matrix.md
│   ├── findings_validation_report.md
│   ├── evidence_pack_w5_w8.md
│   └── findings_evidence_pack.md
└── pipeline/
    ├── adapters/
    │   ├── base.py
    │   ├── qwen_vl.py
    │   ├── lvr_qwen.py
    │   ├── lvr_qwen_traced.py
    │   ├── monet_qwen.py
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
    │   │   ├── lvr_latent_patch_answer_transfer.py
    │   │   ├── monet_latent_patch_answer_transfer.py
    │   │   ├── output_accuracy_sanity.py
    │   │   └── trace_latent.py
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
   | `bf_patch_answer_transfer` | `continuous_margin_shift` | Main-facing continuous margin; legacy `logprob_margin_shift` retained |
   | `bf_swap_latent_replacement` | `continuous_margin_shift` | Main-facing continuous margin; legacy `swap_margin_shift` retained |
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

6. **W12 Monet second-paradigm preflight**

   `config.monet.preflight.yaml` introduces Monet-7B as a real public latent-paradigm candidate. `pipeline/adapters/monet_qwen.py` loads Monet's customized Transformers model for architecture and standard-forward probes. The local preflight loaded `models/Monet-7B`, verified official source commit `08939998d3d643a73a316e349faa34f420429153`, prepared 16 Monet-SFT samples, and ran PF-A/PF-B/BF-Conf at `runs/w12_monet_preflight_n16`. This is intentionally not the final Monet causal metric path: Monet's true latent reasoning happens through the official modified vLLM runner, so paper-grade Monet latent evidence requires a dedicated vLLM trace adapter.

7. **W13 Monet Transformers latent-mode causal range gate**

   `monet_latent_patch_answer_transfer` uses Monet's official Transformers latent-mode machinery. It captures `ce_patch_vec` from source/counterfactual SPD-Faith samples, injects those tensors at the target/clean sample's `ce_patch_pos`, and scores the constrained answer candidates. The local run `runs/w13_monet_latent_patch_n50` passed with `n_paired=50`, `n_success=50`, `n_error=0`, and `latent_answer_transfer_rate=0.06` (95% bootstrap CI `[0.00, 0.14]`). This moves Monet beyond preflight into causal hidden-state evidence, but the boundary remains explicit: it is not yet the modified-vLLM scheduler-native generation loop.

8. **W14-W15 LVR trace-latent primary-metric gate**

   `trace_latent.enabled=true` switches the six v2 primary metrics onto real LVR generation-time hidden-feedback states captured by `lvr_qwen2_5_vl_traced`. This path preserves the original metric IDs but changes the audit object from query spans to captured `output_last_position_hidden_state` tensors. Local run: `runs/w14_w15_lvr_trace_latent_n50_v2`, `n=50`, sanity pass, 20 CI rows, 32 trace-latent plots, and `TRACE LATENT GATE VALIDATION PASSED`.

9. **W16 scale-up and T1-T4 matrix tooling**

   W16 adds configs and tooling for LVR trace-latent scale-up (`n=500` all six metrics, `n=1000` light metrics), T1/T2/T3 main matrices over Maze/SPD/BLINK, and T4 VSI-Bench output accuracy sanity. The new `output_accuracy_sanity` metric is intentionally not causal evidence; it is an external task-performance check.

   The LVR trace-latent SPD `n=500` six-metric scale gate completed on 2026-05-25 and passed `tools/validate_trace_latent_gate.py` at `runs/w16_lvr_trace_latent_spd_n500_sharded/merged`. The W17 light gate completed on 2026-05-26 at `runs/w17_lvr_trace_latent_spd_n1000_light/merged`, with PF-A, PF-B, BF-Conf, and CF-Stage at `n=1000`; BF-Patch/BF-Swap are intentionally disabled in this light config. A hundred-scale real matrix was also completed to verify the full path before launching the larger cross-task matrix: LVR trace-latent SPD `n=100` passed at `runs/w16_lvr_trace_latent_spd_n100_sharded/merged`, and the T1/T2/T3 matrix passed `tools/validate_main_matrix.py` with 42 metric rows and 168 CI rows at `runs/w16_main_matrix_t1_t2_t3_n100/merged`. T4 VSI-Bench did not run because the public HF table has scene metadata but no local visual frames/frame grids.

   BF-Patch/BF-Swap trace-latent margins now diagnose generation-score reliability explicitly. New runs record `clean_score_diagnostic` and `patched_score_diagnostic` separately, use generation scores only when the generated candidate token aligns with the score index, and otherwise fall back to `latent_logit_lens` before the final parsed-answer fallback. The Main-facing scalar is `continuous_margin_shift`, which excludes parsed-answer fallback; `effective_margin_shift` and legacy `logprob_margin_shift` / `swap_margin_shift` remain compatibility fields. The validators surface failure-reason counts so a Main run can distinguish missing scores, tokenization issues, and decision-index mismatch instead of hiding them behind one fallback label.

## Data Sources

Supported `data.source_type` values include:

- `jsonl`
- `hf`
- `lvr_json`
- `spd_faith`
- `maze`
- `blink`
- `vsi`
- `vstar`

`spd_faith` is the paired counterfactual entry point used for the W2 range gate. It expects fields for clean/counterfactual images, clean/counterfactual answers, `paired_id`, and either bbox or region-mask oracle metadata.

`blink` is the W16 fine-grained perception task. If no region metadata is present, PF-A/PF-B use center fallback and the run must be interpreted as weak-oracle diagnostic evidence. `vsi` is reserved for output accuracy sanity, usually with static frame-grid images. `vstar` is the V*Bench high-resolution bbox-localization spotlight task; it requires bbox metadata and should not use center fallback.

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

Prepare the canonical local MazePlanning manifest from the public Latent Sketchpad test set:

```bash
./venv/bin/python tools/prepare_maze_planning_hf.py \
  --out data/maze_planning
```

Prepare BLINK and VSI-Bench manifests:

```bash
./venv/bin/python tools/prepare_blink_hf.py \
  --configs all \
  --split val \
  --out data/blink

./venv/bin/python tools/prepare_vsi_bench_hf.py \
  --split test \
  --image-root /path/to/local/vsi/frame_grids \
  --out data/vsi_bench

./venv/bin/python tools/prepare_vstar_hf.py \
  --split test \
  --out data/vstar
```

BLINK is a multi-config Hugging Face dataset; `--configs all` concatenates the official configs. VSI-Bench's public HF table provides question and scene metadata, so the converter needs a local image/frame-grid root to write runnable visual samples. V*Bench is small (`n=191`) and bbox-centric, so it is tracked as a spotlight gate rather than as a Main T1/T2/T3 matrix task.

For a tiny conversion smoke:

```bash
./venv/bin/python tools/prepare_maze_planning_hf.py \
  --max-samples 2 \
  --out /tmp/maze_planning_prepare_smoke
```

This writes:

```text
data/maze_planning/manifest.jsonl
data/maze_planning/prepare_stats.json
data/maze_planning/images/
```

Prepare a small Monet-SFT preflight manifest from the public Monet dataset:

```bash
./venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
for name in ["CogCoM/images.zip"]:
    print(hf_hub_download("NOVAglow646/Monet-SFT-125K", name, repo_type="dataset", local_dir="data/monet_sft_raw"))
PY

./venv/bin/python tools/prepare_monet_sft_hf.py \
  --max-samples 16 \
  --image-archive-root data/monet_sft_raw \
  --out data/monet_sft
```

Check local Monet-7B and the official `NOVAglow646/Monet` source checkout without loading model weights:

```bash
./venv/bin/python tools/check_monet_env.py \
  --config config.monet.preflight.yaml \
  --check-data
```

The first Monet probe is standard-forward only:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.monet.preflight.yaml \
  --models monet_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment bf_conf_calibrated_progression \
  --device cuda:0 \
  --run-name w12_monet_preflight_n16
```

The W13 Monet causal range gate on SPD-Faith:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.monet_latent_patch.range.yaml \
  --models monet_7b \
  --only monet_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w13_monet_latent_patch_n50

./venv/bin/python tools/validate_monet_latent.py \
  runs/w13_monet_latent_patch_n50 \
  --min-pairs 50
```

The W14-W15 LVR trace-latent primary-metric gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.lvr_trace_latent.w14_w15.yaml \
  --models lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w14_w15_lvr_trace_latent_n50_v2

./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w14_w15_lvr_trace_latent_n50_v2 \
  --min-pairs 50 \
  --min-samples 50
```

Recorded W14-W15 primary results:

| metric | primary scalar | mean | 95% bootstrap CI |
|---|---|---:|---:|
| PF-A | `selectivity` | -0.0382235 | [-0.0721962, -0.00607026] |
| PF-B | `native_alignment` | 0.845704 | [0.822772, 0.866863] |
| BF-Patch | `logprob_margin_shift` | -0.120000 | [-0.280000, 0.040000] |
| BF-Swap | `swap_margin_shift` | -0.120000 | [-0.280000, 0.000000] |
| BF-Conf | `gold_logit_slope` | 0.110406 | [0.0545209, 0.167702] |
| CF-Stage | `late_delta` | -4.046318 | [-5.252559, -2.870815] |

The W14-W15 validator checks that all six metric payloads are in `trace_latent` mode, that LVR traces have `trace_quality=instrumented_sparse_v0`, that BF patch/swap records applied hidden-feedback patches, that primary CI rows exist, and that trace-latent visualizations were written under `metric_plots/`. For post-diagnostic BF artifacts, add `--require-score-diagnostics` to require machine-readable generation-score failure summaries.

The W16 LVR trace-latent scale gates:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.lvr_trace_latent.spd_n500.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_patch_answer_transfer|bf_swap_latent_replacement|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w16_lvr_trace_latent_spd_n500_sharded

./venv/bin/python tools/merge_main_matrix.py runs/w16_lvr_trace_latent_spd_n500_sharded
./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w16_lvr_trace_latent_spd_n500_sharded/merged \
  --min-pairs 500 \
  --min-samples 500

bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.lvr_trace_latent.spd_n1000_light.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w17_lvr_trace_latent_spd_n1000_light

./venv/bin/python tools/merge_main_matrix.py runs/w17_lvr_trace_latent_spd_n1000_light
./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w17_lvr_trace_latent_spd_n1000_light/merged \
  --min-pairs 0 \
  --min-samples 800 \
  --allow-disabled-metrics
```

Recorded W16 LVR trace-latent `n=500` scale result:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.lvr_trace_latent.spd_n500.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_patch_answer_transfer|bf_swap_latent_replacement|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w16_lvr_trace_latent_spd_n500_sharded

./venv/bin/python tools/merge_main_matrix.py runs/w16_lvr_trace_latent_spd_n500_sharded
./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w16_lvr_trace_latent_spd_n500_sharded/merged \
  --min-pairs 500 \
  --min-samples 500 \
  --compat-allow-legacy-margin
```

| metric | scalar | n | value | 95% bootstrap CI |
|---|---|---:|---:|---:|
| PF-A | `selectivity` | 500 | -0.044502 | [-0.054109, -0.035044] |
| PF-B | `native_alignment` | 500 | 0.842298 | [0.835494, 0.848849] |
| BF-Patch | legacy `logprob_margin_shift` | 500 | -0.004000 | [-0.040000, 0.032000] |
| BF-Swap | legacy `swap_margin_shift` | 500 | -0.012000 | [-0.048000, 0.024000] |
| BF-Conf | `gold_logit_slope` | 500 | 0.150825 | [0.134188, 0.167482] |
| CF-Stage | `late_delta` | 500 | -3.372306 | [-3.852121, -2.902451] |

Secondary transfer rates in the same run: BF-Patch `answer_transfer_rate=0.462`, BF-Swap `swap_answer_transfer_rate=0.462`.

The W16 n=500 artifact predates the generation-score diagnostic fields, so it should be revalidated with `--compat-allow-legacy-margin`. New BF-Patch/BF-Swap runs should omit that compatibility flag, may add `--require-score-diagnostics`, and should treat `continuous_margin_shift` as the Main-facing primary scalar. Legacy `logprob_margin_shift` / `swap_margin_shift` remain only for historical artifacts and compatibility tooling.

Recorded W17 LVR trace-latent `n=1000` light result:

| metric | scalar | n | value | 95% bootstrap CI |
|---|---|---:|---:|---:|
| PF-A | `selectivity` | 1000 | -0.055257 | [-0.062864, -0.047651] |
| PF-B | `native_alignment` | 1000 | 0.835482 | [0.830548, 0.840310] |
| BF-Conf | `gold_logit_slope` | 1000 | 0.139953 | [0.128714, 0.151739] |
| CF-Stage | `late_delta` | 1000 | -3.844755 | [-4.168295, -3.525368] |

Validation for the W17 light gate:

```text
TRACE LATENT GATE VALIDATION PASSED
MAIN PAPER READINESS VALIDATION PASSED
merged sanity: 4/4 pass
summary_with_ci rows: 16
```

This is a light trace-latent scale gate, not the all-six BF intervention gate: `config.lvr_trace_latent.spd_n1000_light.yaml` disables BF-Patch and BF-Swap by design.

Recorded W16 hundred-scale trace-latent result:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs /tmp/lvr_trace_latent_spd_n100.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_patch_answer_transfer|bf_swap_latent_replacement|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w16_lvr_trace_latent_spd_n100_sharded

./venv/bin/python tools/merge_main_matrix.py runs/w16_lvr_trace_latent_spd_n100_sharded
./venv/bin/python tools/validate_trace_latent_gate.py \
  runs/w16_lvr_trace_latent_spd_n100_sharded/merged \
  --min-pairs 100 \
  --min-samples 100
```

| metric | scalar | n | value | 95% bootstrap CI |
|---|---|---:|---:|---:|
| PF-A | `selectivity` | 100 | -0.045730 | [-0.067747, -0.024061] |
| PF-B | `native_alignment` | 100 | 0.845153 | [0.830550, 0.859188] |
| BF-Patch | `logprob_margin_shift` | 100 | 0.060000 | [-0.020000, 0.160000] |
| BF-Swap | `swap_margin_shift` | 100 | 0.000000 | [-0.100000, 0.100000] |
| BF-Conf | `gold_logit_slope` | 100 | 0.149496 | [0.112923, 0.188046] |
| CF-Stage | `late_delta` | 100 | -3.885112 | [-4.934226, -2.796776] |

The W16 T1/T2/T3 matrix and T4 accuracy sanity:

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

bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.main_vsi_accuracy.yaml \
  --models qwen2_5_vl_3b,qwen2_5_vl_7b,lvr_7b,monet_7b \
  --metrics output_accuracy_sanity \
  --gpus 0,1,2,3 \
  --run-root runs/w16_vsi_accuracy_sanity
```

Recorded W16 hundred-scale T1/T2/T3 matrix:

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

./venv/bin/python tools/merge_main_matrix.py runs/w16_main_matrix_t1_t2_t3_n100
./venv/bin/python tools/validate_main_matrix.py \
  runs/w16_main_matrix_t1_t2_t3_n100 \
  --tasks maze,spd_faith,blink \
  --min-samples 100 \
  --bf-min-pairs 100
```

T1 Maze `n=100`:

| Metric | Qwen3B | Qwen7B | LVR-7B |
|---|---:|---:|---:|
| PF-A | -0.660587 | -0.898330 | -0.644481 |
| PF-B | 0.475198 | 0.390295 | 0.458739 |
| BF-Conf | 0.190406 | 0.278682 | 0.030451 |
| CF-Stage | 1.465018 | 1.974893 | 1.516935 |

T2 SPD-Faith `n=100`:

| Metric | Qwen3B | Qwen7B | LVR-7B |
|---|---:|---:|---:|
| PF-A | 0.176829 | 0.192988 | 0.206082 |
| PF-B | 0.684626 | 0.672777 | 0.769803 |
| BF-Patch | 0.008750 | 0.020000 | -0.017578 |
| BF-Swap | 0.006250 | 0.026250 | -0.031289 |
| BF-Conf | 0.192592 | 0.349234 | 0.070154 |
| CF-Stage | 1.845391 | 2.195018 | 1.387603 |

T3 BLINK Art_Style `n=100`:

| Metric | Qwen3B | Qwen7B | LVR-7B |
|---|---:|---:|---:|
| PF-A | 0.304636 | 0.243137 | 0.278320 |
| PF-B | 0.543969 | 0.529590 | 0.634386 |
| BF-Conf | -0.103736 | 0.069303 | 0.048056 |
| CF-Stage | 1.725590 | 1.948871 | 1.565728 |

The BLINK Art_Style run has no bbox/region metadata, so PF-A/PF-B are weak-oracle center-fallback diagnostics.

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

`prereg/manifest.yaml` is synchronized to `manifest_version: 0.9-findings-gate`; it preserves the six W2 primary metrics, records W5-W8 as completed gates, adds W9 Findings SPD/Maze/evidence-pack gates, and locks the W13 Monet Transformers latent range gate.

See [docs/validation_report_w5_w8.md](/home/pengguangyue/workspace/proj/lvr-eval-mechanistic-audit/docs/validation_report_w5_w8.md) for exact commands, reductions, and validators.

## Findings Gate

Findings requires the existing true latent intervention artifacts plus a second task family. SPD-Faith is represented by W7, and MazePlanning is represented by the W9 bbox-oracle subset run.

SPD n=500 scale-up config:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.findings_spd_scale.yaml \
  --models qwen2_5_vl_3b qwen2_5_vl_7b lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_patch_answer_transfer bf_swap_latent_replacement \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w9_spd_findings_m0_m1_m2_n500
```

Maze n=200 configured gate:

```bash
./venv/bin/python tools/prepare_maze_planning_hf.py \
  --out data/maze_planning

bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.maze.findings.yaml \
  --models qwen2_5_vl_3b qwen2_5_vl_7b lvr_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment \
         bf_conf_calibrated_progression cf_stage_decay \
  --device cuda:0 \
  --run-name w9_maze_findings_m0_m1_m2_n200_bbox
```

Findings validator and pack:

```bash
./venv/bin/python tools/validate_findings_gate.py
./venv/bin/python tools/build_findings_pack.py
```

## Analysis

`summary_with_ci.json` uses bootstrap confidence intervals. For v2 sample-level artifacts it groups derived rows by `paired_id`, falling back to `id`, before bootstrapping group-level values. This prevents multi-layer, multi-bucket, or multi-family derived rows from being counted as independent samples when group keys are present.

Legacy artifacts without group keys keep the old one-sample bootstrap fallback.

## Known Boundaries

- W2-W8 are reproducibility, localization, capacity, and scale-up gates, not a final causal-result package.
- Findings-ready status is now enforced by `tools/validate_findings_gate.py`; it requires true latent W3/W4/W6 gates, W7 SPD, and W9 Maze.
- W8 is an engineering/review evidence package, not a claim-finalizing result.
- W5 capacity sweep passed validation, but it does not establish monotonic capacity scaling.
- W6 best-step replication is an `n=50` gate-level replication, not a full replication study.
- `pf_b_patch_alignment` currently reports native attention-proxy alignment; DINO patch correspondence remains optional and disabled for the W2 gate.
- `bf_swap_latent_replacement` now records self-swap, reverse-swap, and random-pair controls, but stronger controlled latent-block protocols are still needed.
- `cf_stage_decay` uses `late_delta` as the primary scalar; `late_retention` can explode when the clean late-stage baseline is near zero and is diagnostic only.
- W3-W6 latent patching is true inference-time LVR hidden-feedback intervention on SPD-Faith constrained `original` / `modified` answers.
- W7 remains query-span regression evidence across Qwen/LVR weights, not true latent-state intervention evidence.
- W13 Monet is true hidden-state patching through Monet's official Transformers `latent_mode` / `ce_patch_vec` path on SPD-Faith constrained answers. It is a second-paradigm causal range gate, but not the official modified-vLLM scheduler-native generation trace.
- W14-W15 closes the LVR-side six-primary-metric trace path at gate scale: PF-A/PF-B/BF-Patch/BF-Swap/BF-Conf/CF-Stage can all run on captured LVR generation-time hidden-feedback states when `trace_latent.enabled=true`.
- W14-W15 is still `n=50` LVR/SPD-Faith gate evidence, not a Main-scale cross-task/cross-model matrix.
- In W14-W15 BF-Patch/BF-Swap, LVR generation scores can be unavailable for constrained answer-token margins; those rows record a parsed-answer margin fallback while still measuring generated answer transfer and hidden-feedback patch application from the traced generation loop.
- W16 configs and matrix tooling are implemented for `n=500`/`n=1000` scale-up, but full GPU results are separate artifacts and should be validated before being claimed as completed evidence.
- BLINK PF-A/PF-B can be weak-oracle diagnostics when BLINK records do not include bboxes or masks.
- VSI-Bench is output accuracy sanity only in W16, not a causal audit task.
- PF-B remains native attention-proxy alignment in the current delivered runs; DINOv3 alignment is still future work.
- Broader tasks, larger sample sizes, modified-vLLM Monet tracing, DINO alignment, and layer-level localization remain future work.
