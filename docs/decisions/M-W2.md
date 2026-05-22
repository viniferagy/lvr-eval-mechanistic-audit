# Week 2 Model Go/No-Go

## Summary

Week 2 keeps the main experimental pool limited to adapters that already have local weights, span semantics, decoder layer hooks, final norm, and lm head verified in GPU smoke. Monet, Latent Sketchpad, and CrystaL are tracked as candidate models but do not enter primary Week 2 runs.

## Decisions

| Model | Public weights | Hookability | Main pool | Rationale | Next action |
| --- | --- | --- | --- | --- | --- |
| Monet | unverified | unknown adapter surface | no-go Week 2 | Do not enter main pool until local public weights and decoder-layer hooks are verified. | Record exact checkpoint path and run a no-forward architecture probe before GPU audit. |
| Latent Sketchpad | unverified | likely custom loop | no-go Week 2 | Latent interface may not expose Qwen-style placeholder spans or standard decoder layers. | Inspect model source for latent-token lifecycle and map span semantics before integration. |
| CrystaL | unverified | unknown adapter surface | no-go Week 2 | Needs public-weight availability and hook path confirmation before causal metrics are meaningful. | Add candidate config only after weights, processor, final norm, lm_head, and layers are discoverable. |

## Static Probe

The machine-readable version lives in `pipeline/adapters/probe_catalog.py`. It is intentionally static and does not import model libraries or touch the network. The smoke test validates that each candidate has public-weight status, hookability, go/no-go status, rationale, and next action before future adapter work starts.

## Week 2 Scope

The primary GPU-verified pool remains:

- `qwen2_5_vl_3b`
- `qwen2_5_vl_7b`
- `lvr_7b`

Candidate adapters can move into the pool only after a later decision records local checkpoint availability and a passing hook-path probe.
