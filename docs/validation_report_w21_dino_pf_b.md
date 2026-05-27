# W21 PF-B DINO Sensitivity Pilot

Date: 2026-05-26

This pilot adds an optional DINOv2 image-backbone sensitivity readout to PF-B. Native PF-B remains the model-facing primary scalar. DINO is an external visual-backbone check over the same clean/relevant/irrelevant/random masked images, so its DINO rows are expected to be identical across model wrappers for a fixed dataset.

## Native PF-B

| task | model | n | mean | 95% CI |
|---|---:|---:|---:|---:|
| SPD-Faith n=100 | Qwen2.5-VL-3B | 100 | 0.684626 | [0.650218, 0.719669] |
| SPD-Faith n=100 | Qwen2.5-VL-7B | 100 | 0.672777 | [0.639889, 0.706553] |
| SPD-Faith n=100 | LVR-7B | 100 | 0.769803 | [0.741720, 0.797583] |
| V*Bench n=191 | Qwen2.5-VL-3B | 191 | 0.916909 | [0.906963, 0.926299] |
| V*Bench n=191 | Qwen2.5-VL-7B | 191 | 0.863911 | [0.852201, 0.875477] |
| V*Bench n=191 | LVR-7B | 191 | 0.958472 | [0.952635, 0.963841] |

Interpretation: native PF-B preserves the earlier pattern. LVR-7B is highest on both SPD-Faith and V*Bench. This is the model-dependent alignment claim.

## DINO Region Selectivity

| task | model | n | mean | 95% CI |
|---|---:|---:|---:|---:|
| SPD-Faith n=100 | Qwen2.5-VL-3B | 100 | -0.007127 | [-0.026751, 0.012914] |
| SPD-Faith n=100 | Qwen2.5-VL-7B | 100 | -0.007127 | [-0.026751, 0.012914] |
| SPD-Faith n=100 | LVR-7B | 100 | -0.007127 | [-0.026751, 0.012914] |
| V*Bench n=191 | Qwen2.5-VL-3B | 191 | 0.000467 | [-0.000705, 0.001892] |
| V*Bench n=191 | Qwen2.5-VL-7B | 191 | 0.000467 | [-0.000705, 0.001892] |
| V*Bench n=191 | LVR-7B | 191 | 0.000467 | [-0.000705, 0.001892] |

Interpretation: DINO region selectivity is near zero and its confidence intervals cross zero on both datasets. The external image backbone does not show a strong relevant-region perturbation advantage under the current mask construction. This should be reported as a sensitivity/null check, not as support for a stronger bbox-localization claim.

## Artifacts

- CSV: `runs/w21_dino_pf_b_pilot/merged/w21_dino_pf_b_results.csv`
- Figure: `docs/figures/w21_dino_pf_b/w21_native_alignment.png`
- Figure: `docs/figures/w21_dino_pf_b/w21_dino_region_selectivity.png`

## Next Decision

Priority 1/2 pilot succeeded technically: DINO is implemented, cached, validated by sanity, and produces CI rows. The signal is mixed scientifically: native PF-B supports LVR > Qwen, while DINO selectivity is effectively null. Before expanding to SPD n=1000, prewarm the DINO cache or run DINO on GPU in a separate low-memory phase; otherwise the CPU DINO backend becomes the runtime bottleneck.
