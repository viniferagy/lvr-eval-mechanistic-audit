# W16 Main Matrix Scale Decision

Date: 2026-05-25

## Decision

W16 turns the W14-W15 LVR trace-latent gate into a Main-track scale-up path and adds task coverage beyond SPD/Maze gates.

The task split is:

| Task | Dataset | Role | Metrics |
|---|---|---|---|
| T1 | MazePlanning | chain/step visual reasoning | PF-A, PF-B, BF-Conf, CF-Stage |
| T2 | SPD-Faith | paired counterfactual bridge | six primary metrics |
| T3 | BLINK | fine-grained perception generalization | PF/BF/CF subset; weak-oracle fallback if no bbox |
| T4 | VSI-Bench | external accuracy sanity | output accuracy only |

The LVR trace-latent scale-up uses two tiers:

1. `config.lvr_trace_latent.spd_n500.yaml`: six trace-latent primary metrics at `n=500`.
2. `config.lvr_trace_latent.spd_n1000_light.yaml`: PF-A, PF-B, BF-Conf, and CF-Stage at `n=1000`.

## Implementation Notes

- `source_type=blink` and `source_type=vsi` are first-class loaders.
- `output_accuracy_sanity` is a runnable metric for T4 and optional external sanity checks.
- `tools/launch_main_matrix.sh` is the 4-GPU data-parallel launcher. It must be run through `tools/run_and_hold.sh`.
- `tools/merge_main_matrix.py` merges shard outputs and reruns analysis.
- `tools/validate_main_matrix.py` checks expected task/model/metric coverage.

## Boundary

BLINK can enter the matrix even when no reliable bbox exists, but PF-A/PF-B must then be interpreted as weak-oracle center-fallback diagnostics. T4 VSI-Bench is not causal evidence in this round; it is accuracy sanity only.
