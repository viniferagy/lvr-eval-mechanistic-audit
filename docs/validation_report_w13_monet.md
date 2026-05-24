# W13 Monet Latent Validation Report

Date: 2026-05-25

## Scope

W13 validates a Monet second-paradigm causal range gate on real SPD-Faith paired data. The metric is `monet_latent_patch_answer_transfer`, and the model is `monet_7b`.

This report is intentionally narrow:

- It is a causal hidden-state patch through Monet's official Transformers `latent_mode` / `ce_patch_vec` path.
- It is not Monet's modified-vLLM scheduler-native generation trace.
- It does not upgrade PF-B to DINOv3 alignment.

## Artifacts

```text
config: config.monet_latent_patch.range.yaml
run_dir: runs/w13_monet_latent_patch_n50
metric: runs/w13_monet_latent_patch_n50/metrics/monet_latent_patch_answer_transfer_monet_7b.json
sanity: runs/w13_monet_latent_patch_n50/sanity/summary_sanity.json
ci: runs/w13_monet_latent_patch_n50/summary_with_ci.json
validator: tools/validate_monet_latent.py
```

Monet source:

```text
path: ../Monet
commit: 08939998d3d643a73a316e349faa34f420429153
dirty: false
```

## Result

```text
n_paired: 50
n_success: 50
n_error: 0
n_patch_applied: 50
n_with_captured_state: 50
sanity overall_status: pass
```

Primary scalar:

| metric | model | scalar | n | mean | ci_low | ci_high |
|---|---|---|---:|---:|---:|---:|
| `monet_latent_patch_answer_transfer` | `monet_7b` | `latent_answer_transfer_rate` | 50 | 0.06 | 0.00 | 0.14 |

Secondary scalar:

| metric | model | scalar | n | mean | ci_low | ci_high |
|---|---|---|---:|---:|---:|---:|
| `monet_latent_patch_answer_transfer` | `monet_7b` | `latent_margin_shift` | 50 | -0.0274999961 | -0.0599999908 | 0.0050000121 |

## Interpretation

The W13 gate proves that the repo can perform a second-paradigm causal hidden-state patch on real data, beyond the W12 standard-forward Monet preflight. The low transfer rate is itself a valid range result: under the current `latent_size=10`, all-latent-block replacement protocol, source Monet latent tensors rarely move the target SPD-Faith answer toward the counterfactual answer.

This is not paper-grade Monet runtime evidence. For that, the next gate must trace and patch the official modified-vLLM generation path where sampled `<abs_vis_token>` activates scheduler-level latent mode.

## Commands

```bash
./venv/bin/python tools/check_monet_env.py \
  --config config.monet_latent_patch.range.yaml \
  --check-data
```

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.monet_latent_patch.range.yaml \
  --models monet_7b \
  --only monet_latent_patch_answer_transfer \
  --device cuda:0 \
  --run-name w13_monet_latent_patch_n50
```

```bash
./venv/bin/python tools/validate_monet_latent.py \
  runs/w13_monet_latent_patch_n50 \
  --min-pairs 50
```

Validator output:

```text
monet_latent_patch_answer_transfer: n_paired=50 reduction={'latent_answer_transfer_rate': 0.06, 'latent_margin_shift': -0.02749999612569809, 'n_paired': 50, 'n_success': 50, 'n_error': 0, 'n_patch_applied': 50, 'n_with_captured_state': 50}
summary_with_ci rows=2
MONET LATENT VALIDATION PASSED
```
