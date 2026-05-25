# W17 Main-Shortest Path Validation Report

Date: 2026-05-26

## Scope

This report records the Main-shortest path update after W16. The goal was to fix the trace-latent BF margin path, add stricter evidence validators, implement a minimal regression artifact, and complete the LVR trace-latent `n=1000` light gate without blocking on Monet modified-vLLM scheduler-native tracing.

## Code Changes

The trace-latent BF patch path now uses a three-level margin priority:

```text
generation_scores_aligned_first_token -> latent_logit_lens -> parsed_answer_fallback
```

The generation-score path no longer reads `scores[-1]` blindly. `score_margin_with_diagnostics()` derives generated token ids from the LVR trace, checks candidate tokenization, finds the first generated source/target candidate token, and uses the matching score index only when that alignment exists. If the score path cannot be used, the record stores clean and patched diagnostics separately with reasons such as `missing_scores`, `empty_scores`, `multi_token_candidate`, `decision_index_mismatch`, `score_generated_length_mismatch`, `bad_score_shape`, and `nonfinite_score`.

The `latent_logit_lens` path takes the final captured `output_last_position_hidden_state`, applies `final_norm` when present, projects through `lm_head`, and computes a continuous source-answer minus target-answer token logit margin. BF-Patch and BF-Swap records now expose `clean_score_diagnostic`, `patched_score_diagnostic`, `score_failure_reasons`, `margin_source`, `effective_margin_shift`, and `latent_logit_margin_shift`, while retaining legacy `logprob_margin_shift`.

Transfer-rate reductions are now CI-ready through per-sample reductions:

```text
answer_transfer_rate
swap_answer_transfer_rate
```

New validation tooling:

```text
tools/validate_trace_margin_quality.py
tools/validate_main_paper_readiness.py
```

`tools/validate_trace_latent_gate.py` now rejects excessive parsed-answer fallback for new patch runs, can require generation-score diagnostic summaries, and keeps `--compat-allow-legacy-margin` for pre-diagnostic artifacts. `tools/validate_trace_margin_quality.py` reports score failure reason counts in addition to margin-source provenance. `tools/merge_main_matrix.py` now copies a representative config snapshot into merged artifacts, preserves trace-light sanity settings, and writes `mixed_effects_summary.json`.

## Evidence Discipline

`prereg/manifest.yaml` now explicitly separates evidence levels:

| Model family | Evidence level |
|---|---|
| Qwen2.5-VL baselines | query-span / hidden-state controls |
| LVR-7B | generation-time hidden-feedback trace-latent evidence when `trace_latent.enabled=true` |
| Monet-7B | Transformers latent-mode real-latent range gate, not modified-vLLM scheduler-native trace |

This prevents W7/W9/W16 query-span matrices from being described as real latent interventions.

## Completed Scale Run

LVR trace-latent SPD `n=1000` light gate was run through the required four-GPU launcher:

```bash
bash tools/run_and_hold.sh 0,1,2,3 bash tools/launch_main_matrix.sh \
  --configs config.lvr_trace_latent.spd_n1000_light.yaml \
  --models lvr_7b \
  --metrics 'pf_a_corruption_selectivity|pf_b_patch_alignment|bf_conf_calibrated_progression|cf_stage_decay' \
  --gpus 0,1,2,3 \
  --run-root runs/w17_lvr_trace_latent_spd_n1000_light

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python \
  tools/merge_main_matrix.py runs/w17_lvr_trace_latent_spd_n1000_light

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python \
  tools/validate_trace_latent_gate.py \
  runs/w17_lvr_trace_latent_spd_n1000_light/merged \
  --allow-disabled-metrics \
  --min-samples 800 \
  --min-pairs 0

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python \
  tools/validate_main_paper_readiness.py \
  runs/w17_lvr_trace_latent_spd_n1000_light/merged
```

Validation result:

```text
TRACE LATENT GATE VALIDATION PASSED
MAIN PAPER READINESS VALIDATION PASSED
merged sanity: 4/4 pass
summary_with_ci rows: 16
```

## Results

Primary LVR trace-latent SPD `n=1000` light results:

| Metric | n | value | 95% bootstrap CI |
|---|---:|---:|---:|
| PF-A selectivity | 1000 | -0.055257 | [-0.062864, -0.047651] |
| PF-B native_alignment | 1000 | 0.835482 | [0.830548, 0.840310] |
| BF-Conf gold_logit_slope | 1000 | 0.139953 | [0.128714, 0.151739] |
| CF-Stage late_delta | 1000 | -3.844755 | [-4.168295, -3.525368] |

The run is a light gate by design. BF-Patch and BF-Swap are disabled in `config.lvr_trace_latent.spd_n1000_light.yaml`, so no `answer_transfer_rate`, `swap_answer_transfer_rate`, or margin-provenance rows are expected for this specific `n=1000` artifact. The all-six trace-latent BF intervention scale gate remains `runs/w16_lvr_trace_latent_spd_n500_sharded/merged`.

## Data-Limited Matrix Status

The local staged data state remains:

| Task | Local staged state | Consequence |
|---|---|---|
| T1 Maze | 500 records | can run full local set, still below `n>=800` Main target |
| T2 SPD-Faith | 2996 manifest rows, 2831 usable after bbox filter | enough for `n=1000` |
| T3 BLINK | Art_Style `n=100` staged | PF rows are weak-oracle center fallback; full BLINK staging still needed |
| T4 VSI-Bench | no local visual frames/frame grids | accuracy sanity blocked on images |

The completed T1/T2/T3 matrix remains the W16 hundred-scale checkpoint at `runs/w16_main_matrix_t1_t2_t3_n100/merged`.

## Checks

Completed in this update:

```text
py_compile all repo Python files: pass
smoke_test.py: pass
git diff --check: pass
validate_trace_latent_gate.py on W16 n=500 legacy margin compatibility: pass
validate_trace_latent_gate.py on W17 n=1000 light: pass
validate_main_paper_readiness.py on W17 n=1000 light: pass
smoke fixture requiring generation-score diagnostics: pass
```

## Boundary

This update does not implement full multi-token candidate sequence scoring from generation traces. Multi-token candidates are diagnosed and handled by the latent logit-lens fallback rather than mixed with a non-equivalent teacher-forced scoring path. It also does not implement Monet modified-vLLM scheduler-native generation tracing, PF-B DINO alignment, or a full random-effect mixed model. The mixed-effects artifact is a fixed-effect fallback (`value ~ model + task`) intended to make the Main-readiness artifact contract executable; full mixed-effects statistics remain a paper-polish task for the full cross-task matrix.
