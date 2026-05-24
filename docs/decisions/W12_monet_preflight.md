# W12 Monet Preflight Decision

Date: 2026-05-24

## Summary

The next Main-track step is to add a second real latent paradigm. Monet is the preferred W12 candidate because the official release exposes:

- public `NOVAglow646/Monet-7B` weights;
- public `NOVAglow646/Monet-SFT-125K` data;
- a customized Qwen2.5-VL Transformers model for standard forward probes;
- a customized vLLM runner where `<abs_vis_token>` switches decoding into latent visual mode.

This moves Monet from Week-2 no-go into a W12 preflight candidate, but not yet into the paper evidence matrix.

## Local Result

The local preflight has passed:

```text
Monet source: ../Monet
Monet source commit: 08939998d3d643a73a316e349faa34f420429153
Monet source dirty: false
Model path: models/Monet-7B
Weights: 4 safetensors shards present
Data path: data/monet_sft/manifest.jsonl
Data samples: 16/16 written from NOVAglow646/Monet-SFT-125K
Run dir: runs/w12_monet_preflight_n16
Sanity: 3/3 reports passed, 0 failed; overall status warn due to center-fallback oracle warnings
```

Primary standard-forward reductions:

| metric | scalar | n | value |
|---|---|---:|---:|
| `pf_a_corruption_selectivity` | `selectivity` | 16 | 0.900743 |
| `pf_b_patch_alignment` | `native_alignment` | 16 | 0.602261 |
| `bf_conf_calibrated_progression` | `gold_logit_slope` | 16 | 0.385785 |

## Boundary

The adapter added in this repository loads Monet's published Transformers model for architecture and standard-forward probes. It does not yet audit Monet's true runtime latent feedback loop. Monet true-latent causal evidence requires a vLLM trace adapter around `inference/vllm/monet_gpu_model_runner.py`, with `LATENT_START_ID=151666`, `LATENT_END_ID=151667`, and a recorded `LATENT_SIZE`.

## Commands

Clone/download outside generated-artifact paths:

```bash
git clone https://github.com/NOVAglow646/Monet.git ../Monet
huggingface-cli download NOVAglow646/Monet-7B --local-dir models/Monet-7B
```

Prepare a small public-data preflight manifest:

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

Check local model/source/data paths without loading the model:

```bash
./venv/bin/python tools/check_monet_env.py \
  --config config.monet.preflight.yaml \
  --check-data
```

Run the first standard-forward probe after the environment check passes:

```bash
bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py \
  --config config.monet.preflight.yaml \
  --models monet_7b \
  --only pf_a_corruption_selectivity pf_b_patch_alignment bf_conf_calibrated_progression \
  --device cuda:0 \
  --run-name w12_monet_preflight_n16
```

## Go/No-Go Criteria

Monet can move from preflight into the Main model pool only after:

- `tools/check_monet_env.py` passes and records the source git commit; done locally.
- the standard-forward preflight run loads the checkpoint and locates decoder layers, final norm, lm head, and image spans; done locally.
- a dedicated Monet vLLM trace adapter captures at least one true latent-mode tensor during generation;
- the trace adapter fails hard on fallback, missing runner patching, or missing latent states.

Until then, Monet is a real public candidate and data source, not a completed second-paradigm causal result.
