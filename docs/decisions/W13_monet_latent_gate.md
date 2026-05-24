# W13 Monet Transformers Latent Gate Decision

Date: 2026-05-25

## Summary

W13 moves Monet beyond W12 preflight into a causal hidden-state range gate. The implemented path uses Monet's official Transformers latent-mode API:

- source/counterfactual SPD-Faith sample runs with `latent_mode=True`;
- the adapter captures `ce_patch_pos` and `ce_patch_vec`;
- source `ce_patch_vec` tensors are injected into the target/clean sample's latent positions;
- constrained answer candidates `original` / `modified` are scored by sequence logprob.

This is real hidden-state intervention evidence for Monet's Transformers latent-mode path. It is not yet the official modified-vLLM scheduler-native generation loop.

## Local Result

Run dir: `runs/w13_monet_latent_patch_n50`

Config: `config.monet_latent_patch.range.yaml`

Validator: `tools/validate_monet_latent.py`

```text
n_paired=50
n_success=50
n_error=0
n_patch_applied=50
n_with_captured_state=50
latent_answer_transfer_rate=0.06
95% CI=[0.00, 0.14]
latent_margin_shift=-0.02749999612569809
95% CI=[-0.05999999076128006, 0.005000012088567018]
sanity=pass
```

## Boundary

W13 answers the reviewer concern that Monet was only standard-forward plumbing: it now has a causal latent-state patch metric on real SPD-Faith paired data at `n=50`.

It does not answer the stronger modified-vLLM question. Monet paper inference switches latent mode inside `inference/vllm/monet_gpu_model_runner.py` after `<abs_vis_token>` is sampled. That scheduler-native path still needs a dedicated vLLM trace adapter before claiming Monet runtime-generation causal evidence.

PF-B also remains native alignment / attention-proxy in the delivered Monet work; DINOv3 alignment is still future work.

## Commands

Environment check:

```bash
./venv/bin/python tools/check_monet_env.py \
  --config config.monet_latent_patch.range.yaml \
  --check-data
```

Range gate:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.monet_latent_patch.range.yaml \
  --models monet_7b \
  --only monet_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w13_monet_latent_patch_n50
```

Validation:

```bash
./venv/bin/python tools/validate_monet_latent.py \
  runs/w13_monet_latent_patch_n50 \
  --min-pairs 50
```

CPU:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python -m py_compile \
  run_all.py smoke_test.py merge_and_analyze.py \
  pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py \
  pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py \
  pipeline/adapters/*.py pipeline/stats/*.py tools/*.py

MPLCONFIGDIR=/tmp/matplotlib-lvr-eval ./venv/bin/python smoke_test.py
```

## Next Gate

Implement Monet modified-vLLM scheduler-native trace capture around `monet_gpu_model_runner.py`, then rerun the same SPD-Faith patch protocol with actual sampled `<abs_vis_token>` latent episodes.
