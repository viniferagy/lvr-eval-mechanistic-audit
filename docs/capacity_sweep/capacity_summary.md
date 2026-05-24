# W5 Capacity Sweep Comparison

This report describes paired differences between adjacent latent feedback budgets. It does not require monotonicity.

## Runs

| run | n | steps | best | last | auc |
|---|---:|---:|---:|---:|---:|
| `runs/w5_lvr_capacity_s2_n50` | 50 | 2 | 0.44 | 0.44 | 0.42 |
| `runs/w5_lvr_capacity_s4_n50` | 50 | 4 | 0.54 | 0.42 | 0.465 |
| `runs/w5_lvr_capacity_s8_n50` | 50 | 6 | 0.46 | 0.42 | 0.4233333333333333 |
| `runs/w5_lvr_capacity_s16_n50` | 50 | 10 | 0.52 | 0.46 | 0.44400000000000006 |

## Adjacent Paired Differences

| right-left | scalar | n | mean diff | 95% CI |
|---|---|---:|---:|---|
| `w5_lvr_capacity_s4_n50` - `w5_lvr_capacity_s2_n50` | best_step_transfer | 50 | 0.1 | [0.0, 0.2] |
| `w5_lvr_capacity_s4_n50` - `w5_lvr_capacity_s2_n50` | last_step_transfer | 50 | -0.02 | [-0.1, 0.04] |
| `w5_lvr_capacity_s4_n50` - `w5_lvr_capacity_s2_n50` | step_auc | 50 | 0.045 | [-0.01, 0.11] |
| `w5_lvr_capacity_s8_n50` - `w5_lvr_capacity_s4_n50` | best_step_transfer | 0 | None | [None, None] |
| `w5_lvr_capacity_s8_n50` - `w5_lvr_capacity_s4_n50` | last_step_transfer | 0 | None | [None, None] |
| `w5_lvr_capacity_s8_n50` - `w5_lvr_capacity_s4_n50` | step_auc | 0 | None | [None, None] |
| `w5_lvr_capacity_s16_n50` - `w5_lvr_capacity_s8_n50` | best_step_transfer | 0 | None | [None, None] |
| `w5_lvr_capacity_s16_n50` - `w5_lvr_capacity_s8_n50` | last_step_transfer | 0 | None | [None, None] |
| `w5_lvr_capacity_s16_n50` - `w5_lvr_capacity_s8_n50` | step_auc | 0 | None | [None, None] |
