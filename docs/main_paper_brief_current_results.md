# Latent Visual Faithfulness Is Not Latent Visual Use

Date: 2026-05-25

This note is a self-contained, Main-paper-oriented brief of the evidence currently in the repository. It is written as a compact manuscript skeleton rather than as an internal run log. The current evidence supports a reproducible causal audit toolkit and several strong gates, but it should not yet be described as final Main-track evidence: the full cross-task matrix still needs larger sample sizes, Monet still needs scheduler-native modified-vLLM tracing, and PF-B still uses native alignment rather than DINO alignment. A minimal fixed-effect regression summary is now implemented as a Main-readiness fallback; a full mixed-effects model remains future work.

## Introduction

Multimodal large language models increasingly solve visual tasks by moving some reasoning into hidden or continuous representations rather than exposing every intermediate step as text. In latent visual reasoning (LVR), a model may create visual latent tokens or hidden states that are meant to store task-relevant visual evidence, such as where an object is, which image region changed, or how a path through a maze should proceed. This design is attractive because it can reduce token cost and may let the model reason in a more visual space.

However, performance alone does not tell us whether these latent states are faithful. A model can answer correctly while ignoring the latent states, or it can encode visual evidence in latent states but fail to use that evidence when choosing the final answer. The central question of this project is therefore not "does the model get the answer right?", but:

> When visual reasoning is moved into hidden latent states, can we separately measure whether the state encodes visual evidence, whether it causally drives the answer, and whether its effect survives through the reasoning chain?

We organize this into three axes. The first is availability: can task-relevant visual evidence be detected in the latent representation? The second is usage: does changing or replacing the latent state change the answer in the expected direction? The third is retention: does the latent contribution persist across early, middle, and late stages of reasoning, or does it decay before the final answer?

The current repository implements a six-metric primary audit suite around these axes, supports Qwen2.5-VL baselines, LVR-7B, and a Monet-7B causal range gate, and has run evidence across SPD-Faith, MazePlanning, and BLINK. The strongest current LVR result is the W16/W17 trace-latent scale package: all six primary metrics run on real LVR generation-time hidden-feedback states at n=500 on SPD-Faith, and the lighter PF-A/PF-B/BF-Conf/CF-Stage subset now runs at n=1000. This moves beyond prompt-side or teacher-forced proxies for LVR. The broader task matrix is currently verified at n=100, and the Findings-level SPD/Maze gates are complete at n=200.

## Related Work

Latent visual reasoning methods. Recent MLLM work explores different ways to represent intermediate visual reasoning in latent space. Mirage introduced helper-image-derived latent visual tokens. LVR generates latent representations focused on bounding-box regions. Latent Sketchpad adds a vision-head mechanism and a sketch decoder, and its MazePlanning data is useful for step-level reasoning audits. Monet emits visual latent tokens through an official modified-vLLM runtime and also exposes a Transformers latent-mode path. CoVT, CrystaL, LaCoT, ILVR, LIVR, and related methods broaden the design space with interleaved visual tokens, self-supervision, variational latent reasoning, or selective perceptual modeling. Our project treats these systems as audit targets rather than proposing another latent-reasoning model.

Faithfulness and diagnostic work. Several recent studies already show that latent visual reasoning can fail to use its own latents. CapImagine reports input-to-latent and latent-to-answer disconnects. "What is Holding Back LVR?" shows that replacing latent tokens with uninformative tokens can leave accuracy largely unchanged. "Visual Latents Know More Than They Say" frames a similar suppression problem: latents may contain useful visual information that is not expressed in the answer. These papers make the bare claim "latents may not be used" less novel. Our intended differentiation is a unified causal audit protocol that separates availability, usage, and retention across models, tasks, and intervention sites.

Benchmarks and external anchors. SPD-Faith is a paired spot-the-difference benchmark designed to diagnose multimodal chain-of-thought faithfulness; it is especially useful for counterfactual answer-transfer tests because each sample can have a clean and modified visual fact. MazePlanning provides visual planning samples with natural stage structure, making it useful for retention and stage-decay tests. BLINK provides fine-grained visual perception cases, used here as the T3 mainline perception task. VSI-Bench is reserved for output accuracy sanity only; it is not used as causal audit evidence in the current matrix because the public table available locally lacks image/frame-grid payloads.

## Methodology

### Models

The current model pool has three main audit models plus one second-paradigm range gate:

| Model | Role | Current status |
|---|---|---|
| Qwen2.5-VL-3B | no-latent small baseline | used in SPD, Maze, BLINK matrices |
| Qwen2.5-VL-7B | no-latent baseline | used in SPD, Maze, BLINK matrices |
| LVR-7B | explicit latent visual reasoning model | true hidden-feedback patch gates and trace-latent metrics |
| Monet-7B | second latent paradigm | Transformers latent-mode causal gate only; modified-vLLM trace deferred |

For LVR, the key distinction is query-span audit versus trace-latent audit. Query-span metrics inspect prompt-side or teacher-forced spans and are useful for regression matrices, but they are not the same as intervening on the model's real hidden-feedback generation loop. The W14-W16 trace-latent path instruments LVR generation and captures `output_last_position_hidden_state` tensors during LVR mode. These tensors can then be compared or patched during generation. Qwen rows are therefore query-span / hidden-state controls by design, LVR rows can be real generation-trace latent evidence when `trace_latent.enabled=true`, and Monet is currently a Transformers latent-mode real-latent range gate rather than a scheduler-native generation trace.

### Tasks

| Task | Purpose | Current evidence |
|---|---|---|
| T1 MazePlanning | step-like visual planning and retention | n=200 Findings matrix; n=100 W16 main-matrix checkpoint |
| T2 SPD-Faith | paired visual counterfactuals and answer transfer | n=200 Findings matrix; n=500 LVR trace-latent scale gate |
| T3 BLINK | fine-grained perception generalization | n=100 W16 checkpoint on BLINK Art_Style |
| T4 VSI-Bench | output accuracy sanity only | blocked until local images/frame grids are available |

### Metrics

The six primary metrics are designed to avoid treating one proxy as the whole story.

| Metric | Axis | Plain-language definition |
|---|---|---|
| PF-A corruption selectivity | availability | Compare latent change after masking relevant, irrelevant, and random visual regions. |
| PF-B patch alignment | availability | Measure alignment between latent/hidden representations and native visual-region signals. Current runs use native alignment, not DINO. |
| BF-Patch answer transfer | usage | Patch latent state from a counterfactual source into a target run and test whether the answer moves toward the source answer. |
| BF-Swap latent replacement | usage | Replace latent state with controlled alternatives and measure answer or margin shifts. |
| BF-Conf calibrated progression | usage | Track whether the gold-answer logit margin changes across layers or trace states, with text-only controls when applicable. |
| CF-Stage decay | retention | Compare early, middle, and late stage readouts under perturbation to estimate whether visual influence is retained. |

For uncertainty, the repository writes bootstrap confidence intervals in `summary_with_ci.json`. Validators enforce minimum sample/pair counts, sanity checks, trace quality, patch application, CI rows, and expected metric coverage.

## Experiments

### True LVR Hidden-Feedback Gates

The earliest causal LVR gates operated on constrained SPD-Faith answers and patched LVR hidden-feedback states during generation.

| Gate | n | Main result |
|---|---:|---|
| W3 last-step patch | 50 | latent answer transfer rate 0.52 |
| W4 step sweep | 50 | best step 4, best transfer 0.44, step AUC 0.424 |
| W5 capacity sweep | 50 each | best transfer roughly 0.44 to 0.52 across tested capacities; no monotonic scaling claim |
| W6 best-step replication | 50 | step 4 transfer 0.60 |

These are real LVR hidden-feedback intervention gates, but they are still small-n and mostly SPD-Faith constrained-answer experiments.

### Findings-Level SPD and Maze Matrices

The Findings gate requires true LVR latent artifacts plus SPD and Maze evidence. It is complete locally.

SPD-Faith n=200 query-span/regression matrix:

| Metric | Qwen3B | Qwen7B | LVR-7B |
|---|---:|---:|---:|
| PF-A selectivity | 0.211966 | 0.192499 | 0.179774 |
| PF-B native_alignment | 0.715810 | 0.688120 | 0.785818 |
| BF-Patch logprob_margin_shift | -0.002500 | -0.003750 | -0.006797 |
| BF-Swap swap_margin_shift | 0.003125 | -0.012500 | -0.018828 |
| BF-Conf gold_logit_slope | 0.194958 | 0.343618 | 0.069962 |
| CF-Stage late_delta | 1.804084 | 2.215903 | 1.318340 |

MazePlanning n=200 matrix:

| Metric | Qwen3B | Qwen7B | LVR-7B |
|---|---:|---:|---:|
| PF-A selectivity | -0.654818 | -0.884300 | -0.626778 |
| PF-B native_alignment | 0.476241 | 0.391128 | 0.462267 |
| BF-Conf gold_logit_slope | 0.189739 | 0.277957 | 0.030699 |
| CF-Stage late_delta | 1.486417 | 2.000213 | 1.517284 |

The important boundary is that these W7/W9 matrices are not all real latent interventions. They provide reproducible cross-model regression evidence, while W3/W4/W6 and W14-W16 provide the true LVR hidden-feedback intervention evidence.

### Monet Latent Gate

W13 adds a second-paradigm causal range gate for Monet-7B. It uses Monet's official Transformers `latent_mode` path: capture source `ce_patch_vec` tensors and inject them at target `ce_patch_pos` positions, then score constrained `original` versus `modified` answers.

| Metric | n | Mean | 95% CI |
|---|---:|---:|---:|
| latent_answer_transfer_rate | 50 | 0.06 | [0.00, 0.14] |
| latent_margin_shift | 50 | -0.027500 | [-0.060000, 0.005000] |

This proves that the repository can run a second latent-paradigm hidden-state patch. It does not yet audit Monet's official modified-vLLM scheduler-native generation trace.

### LVR Trace-Latent Scale Gate

W14-W16 upgrade the six primary LVR metrics to real generation-time hidden-feedback states. The all-six scale result is the n=500 SPD-Faith trace-latent gate:

| Metric | n | Value | 95% bootstrap CI |
|---|---:|---:|---:|
| PF-A selectivity | 500 | -0.044502 | [-0.054109, -0.035044] |
| PF-B native_alignment | 500 | 0.842298 | [0.835494, 0.848849] |
| BF-Patch logprob_margin_shift | 500 | -0.004000 | [-0.040000, 0.032000] |
| BF-Patch answer_transfer_rate | 500 | 0.462000 | secondary summary |
| BF-Swap swap_margin_shift | 500 | -0.012000 | [-0.048000, 0.024000] |
| BF-Swap swap_answer_transfer_rate | 500 | 0.462000 | secondary summary |
| BF-Conf gold_logit_slope | 500 | 0.150825 | [0.134188, 0.167482] |
| CF-Stage late_delta | 500 | -3.372306 | [-3.852121, -2.902451] |

The n=500 gate passed trace-latent validation with 6/6 sanity pass and 20 CI rows. The interpretation is mixed but informative: PF-B, BF-Conf, and CF-Stage are stable; BF-Patch and BF-Swap show stable answer-transfer rates around 0.462, but their margin-shift confidence intervals cross zero. A follow-up code path now prefers continuous latent logit-lens margins when generation scores are unavailable, so new runs should report margin provenance and answer-transfer confidence intervals rather than silently falling back to parsed-answer three-value margins.

W17 completed the planned light scale gate for the four non-patch trace-latent metrics at n=1000:

| Metric | n | Value | 95% bootstrap CI |
|---|---:|---:|---:|
| PF-A selectivity | 1000 | -0.055257 | [-0.062864, -0.047651] |
| PF-B native_alignment | 1000 | 0.835482 | [0.830548, 0.840310] |
| BF-Conf gold_logit_slope | 1000 | 0.139953 | [0.128714, 0.151739] |
| CF-Stage late_delta | 1000 | -3.844755 | [-4.168295, -3.525368] |

This W17 artifact passed trace-latent validation, merged sanity, and the new Main-readiness artifact validator. It is deliberately a light gate: BF-Patch and BF-Swap remain represented by the n=500 all-six run.

### W16 T1/T2/T3 Hundred-Scale Matrix

A real n=100 checkpoint verified the full main-matrix path over Maze, SPD-Faith, and BLINK with three models. The validator passed with 42 metric rows and 168 CI rows. BLINK carries a warning because the local Art_Style subset lacks reliable region boxes, so PF metrics use a weak-oracle center fallback.

| Task | Main observation |
|---|---|
| Maze n=100 | Qwen7B has stronger BF-Conf slope than LVR-7B; PF-A is negative across models under this task definition. |
| SPD n=100 | PF-B is highest for LVR-7B, but BF-Patch/BF-Swap margins remain near zero. |
| BLINK Art_Style n=100 | PF and CF run end to end, but PF evidence is weak-oracle because of missing bbox metadata. |

## Current Claim Boundary and Next Steps

The current evidence is strong enough to support a Findings-level claim: the repository provides a reproducible causal audit toolkit, true LVR hidden-feedback intervention gates, SPD/Maze evidence, a Monet Transformers latent-mode causal range gate, and an n=500 LVR trace-latent scale gate.

It is not yet a final Main-paper causal matrix. The next steps are:

1. Expand T1/T2/T3 from n=100 checkpoint scale to n=800-1000 where data permits.
2. Add local VSI-Bench images/frame grids for T4 accuracy sanity.
3. Keep Monet as a Transformers latent-mode range gate for the Main-shortest path; implement modified-vLLM scheduler-native tracing as a separate high-risk spike.
4. Replace or supplement PF-B native alignment with DINO-style alignment.
5. Replace the current fixed-effect regression fallback with full mixed-effects analysis over the full matrix.
