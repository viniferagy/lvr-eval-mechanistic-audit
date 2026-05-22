# Week 2 Task Allocation

## Summary

Week 2 converts the Week 1 foundation into primary metric code. The work is split into small commit-sized batches so each batch can be verified on CPU fixtures and at least one real GPU smoke.

## Batch 1: PF-A Corruption Selectivity

- Status: complete in commit batch 1.
- Owner: metric/statistics line.
- Goal: replace the PF-A skeleton with a runnable relevant-vs-irrelevant-random corruption selectivity metric.
- Implementation target: `pipeline/metrics/v2/pf_a_corruption_selectivity.py`, `pipeline/corruptions.py`, `pipeline/internal_metrics.py`, and `pipeline/analysis.py`.
- Implementation result: PF-A now applies relevant, irrelevant, and random binary masks to the clean image and computes clean-vs-region-masked query-to-image attention KL through an explicit helper instead of reusing random PF-3 corruption with severity 0.
- Acceptance evidence:
  - `./venv/bin/python -m py_compile run_all.py smoke_test.py merge_and_analyze.py pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py pipeline/adapters/*.py pipeline/stats/*.py`
  - `./venv/bin/python smoke_test.py`
  - `bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py --config /tmp/lvr_gpu_qwen3b_smoke_pass.yaml --models qwen2_5_vl_3b --only pf_a_corruption_selectivity --device cuda:0 --run-name gpu_week2_pfa_qwen3b_smoke2`
- GPU artifact: `runs/gpu_week2_pfa_qwen3b_smoke2/metrics/pf_a_corruption_selectivity_qwen2_5_vl_3b.json` reports `selectivity=0.2931104886035124`, `relevant_kl=0.14045639687942135`, `irrelevant_kl=0.43356688548293376`, `random_kl=0.4952622064285808`, `n=1`; `summary_with_ci.json` reports all four scalars with `n=1`; sanity summary status is `pass`.

## Batch 2: BF-Patch Answer Transfer

- Status: complete in commit batch 2.
- Owner: causal patching line.
- Goal: implement the first native PyTorch hook loop for 5 layers x 3 position buckets on paired samples.
- Implementation target: `pipeline/metrics/v2/bf_patch_answer_transfer.py`.
- Implementation result: BF-Patch now builds paired clean/counterfactual samples, captures source hidden states at the selected decoder layer and position bucket, patches the target forward pass, and reports source-vs-target answer logit margin shift plus answer-transfer rate. The public schema keeps the default 5 x 3 grid while config may restrict layers/buckets for smoke runs.
- Acceptance evidence:
  - `./venv/bin/python -m py_compile run_all.py smoke_test.py merge_and_analyze.py pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py pipeline/adapters/*.py pipeline/stats/*.py`
  - `./venv/bin/python smoke_test.py`
  - `bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py --config /tmp/lvr_gpu_bf_patch_qwen3b.yaml --models qwen2_5_vl_3b --only bf_patch_answer_transfer --device cuda:0 --run-name gpu_week2_bf_patch_qwen3b_smoke3`
- GPU artifact: `runs/gpu_week2_bf_patch_qwen3b_smoke3/metrics/bf_patch_answer_transfer_qwen2_5_vl_3b.json` reports `n_paired=1`, `n_cells=3`, `logit_margin_shift=0.1875`, `answer_transfer_rate=0.0`, with all three smoke cells at `n_success=1` and `n_error=0`; `summary_with_ci.json` reports `answer_transfer` and `logit_margin_shift`; sanity summary status is `pass`.

## Batch 3: BF-Swap And BF-Conf

- Status: complete in commit batch 3.
- Owner: behavior-faithfulness line.
- Goal: add controlled latent replacement and calibrated confidence progression modules.
- Implementation target: new v2 modules under `pipeline/metrics/v2/`.
- Implementation result: `bf_swap_latent_replacement` performs paired hidden-state replacement from counterfactual source to clean target and reports swap margin shift / answer-transfer rate. `bf_conf_calibrated_progression` wraps BF-3 confidence progression with answer-token logit slope and a recorded text-only control, avoiding OOD zero-ablation as the primary scalar.
- Acceptance evidence:
  - `./venv/bin/python -m py_compile run_all.py smoke_test.py merge_and_analyze.py pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py pipeline/metrics/legacy/*.py pipeline/metrics/v2/*.py pipeline/adapters/*.py pipeline/stats/*.py`
  - `./venv/bin/python smoke_test.py`
  - `bash tools/run_and_hold.sh 0,1,2,3 ./venv/bin/python run_all.py --config /tmp/lvr_gpu_bf_swap_conf_qwen3b.yaml --models qwen2_5_vl_3b --only bf_swap_latent_replacement bf_conf_calibrated_progression --device cuda:0 --run-name gpu_week2_bf_swap_conf_qwen3b_smoke`
- GPU artifact: `runs/gpu_week2_bf_swap_conf_qwen3b_smoke/metrics/bf_swap_latent_replacement_qwen2_5_vl_3b.json` reports `n_paired=1`, `n_cells=1`, `swap_margin_shift=0.0`, `swap_answer_transfer_rate=0.0`, `n_success=1`, `n_error=0`. `bf_conf_calibrated_progression_qwen2_5_vl_3b.json` reports `early_to_late_drop=1.5879336893558502`, `gold_logit_slope=0.11714124839124847`, and `text_only_control_rate=1.0`; `summary_with_ci.json` includes both metrics; sanity summary status is `pass`.

## Batch 4: CF-Stage And PF-B

- Owner: retention/alignment line.
- Goal: add stagewise early/mid/late aggregation and native/DINO alignment entrypoints.
- Implementation target: `cf_stage_decay.py`, `pf_b_patch_alignment.py`, and optional DINO dependency gating.
- Acceptance: CPU schema smoke passes; GPU smoke can run native-only PF-B without DINO if DINO weights are absent.

## Batch 5: Model Go/No-Go Scaffolding

- Owner: adapter line.
- Goal: document and scaffold Monet / Latent Sketchpad / CrystaL adapter probes without committing to unstable weights.
- Implementation target: `docs/decisions/M-W2.md` plus lightweight adapter sanity stubs.
- Acceptance: decision document records public-weight availability, hookability, and whether each model enters the main pool.
