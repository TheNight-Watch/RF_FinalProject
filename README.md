# Mask-Aware Temporal Keypoint Imputation for Diffusion Policy Push-T

This final project reproduces the official low-dimensional Push-T checkpoint from **Diffusion Policy** and implements a training-free deployment improvement: a mask-aware temporal observation wrapper for keypoint dropout.

## Final Claim

The supported claim is narrow:

> We identify a mask-handling failure mode in the official low-dimensional Push-T Diffusion Policy deployment path and show that a training-free, mask-aware temporal observation wrapper substantially improves robustness under detector-style keypoint dropout.

This project does **not** claim a new general imputation algorithm, clean full-observation improvement, or visual Diffusion Policy occlusion robustness.

## Why This Is the Main Contribution

The official Push-T keypoint runner constructs both coordinates and masks:

```python
"obs": obs[..., :self.n_obs_steps, :Do]
"obs_mask": obs[..., :self.n_obs_steps, Do:] > 0.5
```

The low-dimensional policy consumes only `obs_dict["obs"]`; it does not use `obs_mask`. The training dataset also has no mask channel or keypoint dropout augmentation, and the low-dimensional training configuration uses `keypoint_visible_rate: 1.0`.

Therefore, detector-style missing keypoints that are zero-filled at deployment become out-of-distribution observations. The project evaluates whether simple mask-aware imputation can repair that deployment failure without retraining.

## Main Results

Reproduction sanity check:

| Evaluation | Mean score | Notes |
|---|---:|---|
| Local 50-seed DDPM reproduction | 0.919 | compatibility environment |
| Checkpoint filename reference | 0.969 | reported by released checkpoint |

The gap is reported as an environment-version reproduction gap because this project runs the official code in a compatibility stack rather than the original Python 3.9 / PyTorch 1.12 / Gym 0.21 setup.

Main robustness experiment: official low-dimensional Push-T checkpoint, DDIM `(k,h)=(100,8)`, 20 matched seeds.

DDIM is used only as a deterministic, stable evaluation sampler. It is not the final contribution. The mask-handling failure happens before diffusion sampling because the deployment path discards `obs_mask` before policy inference; the empirical table below is reported under DDIM.

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

Interpretation: per-keypoint temporal imputation, especially carry-forward and linear extrapolation, substantially improves robustness over zero-fill, mean-prior fill, and full-frame hold. The oracle row is an upper bound and is not deployable.

## Key Files

- English CoRL-style report: `report/corl_final_report.pdf`
- English report source: `report/corl_final_report.tex`
- Chinese report: `report/final_report_zh.md`
- English Markdown report: `report/final_report.md`
- Evaluation runner: `scripts/run_official_kh_grid.py`
- Occlusion analysis: `scripts/analyze_occluded_imputation.py`
- Main summary CSV: `results/official_occluded_imputation_all_summary.csv`
- Main figure: `figures/official_occluded_imputation_all.png`
- Innovation and claim audit: `artifacts/logs/innovation_search_audit_zh.md`

## Notes for Claude to Build the PPT

Read these files first:

1. `report/corl_final_report.pdf`  
   Final English CoRL-style report. Use this as the authoritative narrative.
2. `report/final_report_zh.md`  
   Chinese explanation with defense wording and likely Q&A.
3. `results/official_occluded_imputation_all_summary.csv`  
   Main result table with all imputation baselines and paired bootstrap intervals.
4. `figures/official_occluded_imputation_all.png`  
   Main figure for slides.
5. `artifacts/logs/innovation_search_audit_zh.md`  
   Audit trail explaining why scheduler/DDIM were downgraded and why this claim is safer.
6. `scripts/run_official_kh_grid.py` and `scripts/analyze_occluded_imputation.py`  
   Implementation and analysis details, only if code-level verification is needed.

Suggested PPT storyline:

1. Reproduce: official low-dimensional Push-T Diffusion Policy checkpoint is loaded and evaluated.
2. Source audit: runner provides `obs_mask`, but low-dimensional policy inference uses only `obs_dict["obs"]`.
3. Failure mode: detector-style keypoint dropout with zero-filled missing keypoints causes severe OOD observations.
4. Improvement: a training-free mask-aware temporal observation wrapper imputes missing keypoints before normalization.
5. Baselines: zero-fill, mean prior, frame hold, carry-forward, linear extrapolation, oracle upper bound.
6. Results: temporal per-keypoint imputation substantially improves robustness on 20 matched official seeds.
7. Limitations: iid keypoint dropout only; continuous occlusion and RGB policy robustness are future work.

Key numbers for slides:

Reproduction:

| Evaluation | Mean score |
|---|---:|
| Local 50-seed DDPM reproduction | 0.919 |
| Checkpoint filename reference | 0.969 |

Improvement:

| visible rate | zero-fill | mean prior | frame hold | carry-forward | linear | oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 0.177 / 0.00 | 0.405 / 0.25 | 0.403 / 0.20 | 0.901 / 0.75 | 0.875 / 0.85 | 0.900 / 0.90 |
| 25% | 0.134 / 0.00 | 0.257 / 0.00 | 0.257 / 0.00 | 0.575 / 0.20 | 0.668 / 0.35 | 0.900 / 0.90 |

Each cell is `mean score / success rate`.

Safe one-sentence claim:

> A no-retraining, mask-aware temporal observation wrapper substantially improves official low-dimensional Push-T Diffusion Policy robustness under detector-style keypoint dropout.

Do not claim:

- We invented temporal imputation.
- This solves general RGB/visual Diffusion Policy occlusion.
- This improves clean full-observation performance.
- Linear extrapolation is statistically significantly better than carry-forward.
- The earlier surrogate scheduler proves official checkpoint improvement.

Defense notes:

- Training data has no mask channel and no keypoint dropout augmentation; the checkpoint was trained on full keypoint observations.
- DDIM `(100,8)` is used for deterministic evaluation, not claimed as the contribution.
- The mask-handling failure is before the sampler: `obs_mask` is discarded before policy inference.
- Imputation is performed in raw 0-512 keypoint coordinate space before the policy normalizer.
- First invisible observation falls back to the training mean prior, not zero.
- Carry-forward and linear are per-keypoint/per-coordinate updates, not full-frame hold.
- Linear is higher than carry-forward at 25% visibility, but the paired CI crosses zero, so present it as a trend.

## Reproduce the Main Occlusion Experiment

The formal logs are already included under:

```text
official_reproduction/pusht_official_occluded_impute_n20/
```

To rerun one visibility rate:

```bash
cd /root/FinalProject
for impute in none mean_prior frame_hold carry_forward linear oracle; do
  CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache \
    /root/FinalProject/.venv_official/bin/python scripts/run_official_kh_grid.py \
    --sampler ddim \
    --obs-impute "$impute" \
    --keypoint-visible-rate 0.5 \
    --occlude-masked-keypoints \
    --output-root "official_reproduction/pusht_official_occluded_impute_n20/vr_0.5/${impute}" \
    --csv-out "results/official_occluded_impute_n20_vr_0.5_${impute}.csv" \
    --n-test 20 --n-envs 5 --pairs 100,8
done

/root/FinalProject/.venv_official/bin/python scripts/analyze_occluded_imputation.py
```

Use `--keypoint-visible-rate 0.25` and output paths with `vr_0.25` for the severe dropout setting.

## Limitations

The current occlusion model uses the official environment's iid per-keypoint visibility mask. Real keypoint detectors often fail with temporally correlated occlusion, so continuous dropout is the most important next experiment. The result is specific to low-dimensional Push-T keypoints and should not be presented as a general RGB-policy occlusion method.

## Deprecated Exploratory Directions

The repository also contains older DDIM/scheduler/postprocessing probes. They are retained for auditability but are not the final contribution:

- surrogate scheduler gains are not official-checkpoint evidence;
- DDIM is an established sampler and not a novel improvement;
- action clipping, rate limiting, and geometric fallback did not provide a reliable final result.
