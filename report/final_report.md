# Mask-Aware Temporal Keypoint Imputation for Robust Low-Dimensional Diffusion Policy Push-T

## Abstract

This project reproduces the official low-dimensional Push-T checkpoint from Diffusion Policy and studies a deployment failure mode in the low-dimensional keypoint interface. The official Push-T keypoint runner exposes keypoint coordinates and a visibility mask, but the low-dimensional policy consumes only the coordinate vector. Under detector-style keypoint dropout, zero-filled missing keypoints create out-of-distribution observations for a checkpoint trained on fully visible keypoints.

We implement a training-free, mask-aware observation wrapper that imputes masked keypoints before normalization and policy inference. On 20 matched official Push-T seeds, per-keypoint temporal imputation substantially improves robustness under synthetic keypoint dropout. At 50% visibility, zero-fill obtains `0.177` score and `0.00` success, while carry-forward reaches `0.901 / 0.75` and linear extrapolation reaches `0.875 / 0.85`. At 25% visibility, linear improves from `0.134 / 0.00` to `0.668 / 0.35`.

## Claim

Defensible final claim:

> We identify a mask-handling failure mode in the official low-dimensional Push-T Diffusion Policy deployment path and show that a training-free, mask-aware temporal observation wrapper substantially improves robustness under detector-style keypoint dropout.

This does not claim a new general imputation algorithm, clean-setting improvement, or visual Diffusion Policy occlusion robustness.

## Source Audit

The official low-dimensional policy normalizes and consumes only `obs_dict["obs"]`. The Push-T keypoint runner constructs both:

```python
"obs": obs[..., :self.n_obs_steps, :Do]
"obs_mask": obs[..., :self.n_obs_steps, Do:] > 0.5
```

The policy does not read `obs_mask`.

The training dataset reads `keypoint`, `state`, and `action` from replay data, then concatenates keypoints and agent position into a 20-dimensional observation. It has no mask channel and no keypoint dropout augmentation. The training configuration uses `keypoint_visible_rate: 1.0`. Therefore, the official checkpoint is trained on full keypoint observations.

## Method

The implementation wraps the loaded official policy at inference time. It changes only the observation passed into `predict_action`; checkpoint weights, normalizer, sampler, action horizon, and environment dynamics are unchanged. Imputation happens in the original Push-T coordinate space before normalization.

Compared strategies:

| Method | Description |
|---|---|
| zero-fill | pass zeroed masked coordinates |
| mean prior | fill with checkpoint normalizer training mean |
| frame hold | use the most recent fully visible keypoint frame |
| carry-forward | independently use the most recent visible value per coordinate |
| linear | independently extrapolate from the two most recent visible values per coordinate |
| oracle | retain true coordinates despite the mask; upper bound only |

## Results

Official low-dimensional Push-T checkpoint, DDIM `(k,h)=(100,8)`, 20 matched seeds:

| visible rate | method | score | success >=0.95 | delta vs zero | 95% CI |
|---:|---|---:|---:|---:|---:|
| 50% | zero-fill | 0.177 | 0.00 | 0.000 | - |
| 50% | mean prior | 0.405 | 0.25 | +0.228 | [+0.058, +0.405] |
| 50% | frame hold | 0.403 | 0.20 | +0.226 | [+0.060, +0.404] |
| 50% | carry-forward | 0.901 | 0.75 | +0.725 | [+0.586, +0.846] |
| 50% | linear | 0.875 | 0.85 | +0.698 | [+0.544, +0.835] |
| 50% | oracle | 0.900 | 0.90 | +0.723 | [+0.571, +0.854] |
| 25% | zero-fill | 0.134 | 0.00 | 0.000 | - |
| 25% | mean prior | 0.257 | 0.00 | +0.123 | [+0.041, +0.214] |
| 25% | frame hold | 0.257 | 0.00 | +0.123 | [+0.040, +0.216] |
| 25% | carry-forward | 0.575 | 0.20 | +0.440 | [+0.290, +0.587] |
| 25% | linear | 0.668 | 0.35 | +0.533 | [+0.378, +0.685] |
| 25% | oracle | 0.900 | 0.90 | +0.766 | [+0.627, +0.881] |

Linear has higher mean and success than carry-forward at 25% visibility, but paired confidence intervals cross zero: score delta `+0.093`, 95% CI `[-0.125, +0.305]`; success delta `+0.15`, 95% CI `[-0.10, +0.40]`. This is a trend, not a strict dominance claim.

## Limitations

The current occlusion model is iid per-keypoint dropout from the official environment mask. Real detector failures are often temporally correlated, so continuous occlusion is an important next experiment. The result is specific to low-dimensional Push-T keypoints and does not directly transfer to RGB policies. At 25% visibility, oracle remains substantially stronger, so severe occlusion is only partially recovered.

## Key Artifacts

- CoRL-style English report: `report/corl_final_report.tex`
- CoRL-style English PDF: `report/corl_final_report.pdf`
- Chinese report: `report/final_report_zh.md`
- Runner: `scripts/run_official_kh_grid.py`
- Analysis: `scripts/analyze_occluded_imputation.py`
- Summary CSV: `results/official_occluded_imputation_all_summary.csv`
- Figure: `figures/official_occluded_imputation_all.png`
- Audit log: `artifacts/logs/innovation_search_audit_zh.md`
