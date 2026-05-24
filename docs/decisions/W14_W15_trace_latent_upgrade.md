# W14-W15 LVR Trace-Latent Metric Upgrade Decision

Date: 2026-05-25

## Summary

W14-W15 upgrades the six v2 primary metrics from query-span audits to an opt-in real LVR generation-trace latent path. The upgrade keeps W7/W9 regression behavior intact by default, and only activates when config sets `trace_latent.enabled=true`.

The implemented path uses `lvr_qwen2_5_vl_traced` and captures real `output_last_position_hidden_state` tensors from the official LVR generation loop. Patch metrics inject those tensors back through `forward_pre.last_position_hidden_state`; PF/CF metrics compare latent-state deltas under controlled image corruption; BF-Conf reads captured latent states through the final norm and LM head.

## Decision

Add a shared trace-latent utility layer:

```text
pipeline/metrics/v2/trace_latent.py
```

Then route the six v2 metrics through it when requested:

```text
pf_a_corruption_selectivity
pf_b_patch_alignment
bf_patch_answer_transfer
bf_swap_latent_replacement
bf_conf_calibrated_progression
cf_stage_decay
```

The old query-span path remains the default for baseline/Qwen and regression matrices. This preserves Findings/W7 comparability while making Main-track LVR audits use true inference-time hidden feedback.

## Local Result

Run dir: `runs/w14_w15_lvr_trace_latent_n50_v2`

Config: `config.lvr_trace_latent.w14_w15.yaml`

Validator: `tools/validate_trace_latent_gate.py`

```text
sanity overall_status: pass
summary_with_ci rows: 20
trace-latent plots: 32
validator: TRACE LATENT GATE VALIDATION PASSED
```

Primary reductions:

| metric | scalar | mean |
|---|---|---:|
| PF-A | selectivity | -0.0382235 |
| PF-B | native_alignment | 0.845704 |
| BF-Patch | logprob_margin_shift | -0.120000 |
| BF-Swap | swap_margin_shift | -0.120000 |
| BF-Conf | gold_logit_slope | 0.110406 |
| CF-Stage | late_delta | -4.046318 |

## Boundary

This is true LVR generation-time hidden-feedback evidence for LVR-7B on SPD-Faith at `n=50`. It does not turn W7/W9 query-span Qwen/LVR matrices into real latent interventions, and it does not implement Monet modified-vLLM scheduler-native tracing.

For BF-Patch/BF-Swap, generation scores are not always reliable on the constrained answer tokens in the LVR generation API. The trace-latent path therefore records finite parsed-answer fallback margins when generation-score margins are unavailable. Answer-transfer rates and hidden-feedback patch application remain directly measured from generated outputs and trace hooks.

## Next Gate

Scale the trace-latent LVR path beyond `n=50`, then port the same trace contract to Monet's modified-vLLM runner.
