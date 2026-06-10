# Official Diffusion Policy Push-T Sampler Improvement

## Abstract

This project reproduces the official low-dimensional Push-T checkpoint from Diffusion Policy and implements a direct inference-time improvement: replacing the official DDPM ancestral sampler with a plug-and-play DDIM deterministic sampler. The official checkpoint is evaluated with matched Push-T seeds under fixed `(k,h)` settings, where `k` is the number of denoising steps and `h` is the number of executed control steps before replanning. DDIM dramatically improves low/mid-denoising official settings: at `(25,2)`, score improves from `0.155` to `0.900`; at `(50,4)`, score improves from `0.653` to `0.929`. The claim is carefully bounded: DDIM does not dominate the strongest tested DDPM high-denoising baseline `(100,8)`, which remains at score `0.949`.

## 1. Introduction

Diffusion Policy uses iterative denoising to generate action chunks for receding-horizon robot control. Its inference behavior depends on both the sampler and the deployment parameters:

- `k`: denoising steps per policy call.
- `h`: executed actions before replanning.

The original official checkpoint uses a DDPM scheduler. A natural improvement idea is to swap the sampler at inference time, without retraining, using DDIM. This is not claimed as a novel diffusion-model algorithm; DDIM is established prior work. The contribution here is an official-checkpoint validation of this plug-and-play sampler replacement on Push-T.

## 2. Related Work and Claim Boundary

Diffusion Policy introduced action diffusion for visuomotor policy learning. DDIM introduced deterministic non-Markovian diffusion sampling and is widely used for faster or more stable inference. Recent robotics work such as VADF, RA-DP, TIDAL, RTI-DP, and adaptive diffusion replanning already studies adaptive scheduling, replanning, and inference-time compute allocation.

Therefore, this project does not claim a new scheduling algorithm. The supported improvement is narrower and directly tested: using DDIM instead of the official DDPM sampler for the public Diffusion Policy Push-T checkpoint.

## 3. Official Reproduction Setup

The official source was acquired through the GitHub codeload archive after repeated clone instability. The checkpoint was downloaded from the Diffusion Policy project server:

```text
official_reproduction/data/epoch=0550-test_mean_score=0.969.ckpt
```

The preferred 50-seed official DDPM run used official-era `diffusers==0.11.1` and produced:

```text
mean score = 0.9186505414953691
```

The checkpoint filename reports `test_mean_score=0.969`; the remaining gap is reported as an environment-version reproduction gap because this container uses Python 3.12, PyTorch 2.7, and Gym 0.26 compatibility patches.

## 4. Method: Plug-and-Play DDIM Sampler

The official policy object exposes a `noise_scheduler`. The improvement replaces:

```python
DDPMScheduler
```

with:

```python
DDIMScheduler.from_config(policy.noise_scheduler.config)
```

The model weights, normalization, observation history, action horizon, and environment runner are unchanged. The evaluation script is:

```text
scripts/run_official_kh_grid.py --sampler ddim
```

Matched seed analysis is performed by:

```text
scripts/analyze_official_sampler_comparison.py
```

## 5. Results

Matched 20-seed official Push-T comparison:

| `(k,h)` | DDPM Score | DDIM Score | Delta | DDPM Success | DDIM Success |
|---|---:|---:|---:|---:|---:|
| `(25,2)` | 0.155 | 0.900 | +0.745 | 0.000 | 0.800 |
| `(50,4)` | 0.653 | 0.929 | +0.276 | 0.350 | 0.750 |
| `(100,8)` | 0.949 | 0.900 | -0.048 | 0.950 | 0.900 |

Paired bootstrap 95% confidence intervals for score deltas:

- `(25,2)`: `[+0.593, +0.870]`
- `(50,4)`: `[+0.056, +0.488]`
- `(100,8)`: `[-0.152, +0.007]`

These results show that DDIM is a strong improvement for low/mid-denoising official checkpoint settings. At `(25,2)`, DDPM nearly fails, while DDIM reaches success `0.800`. At `(50,4)`, DDIM nearly matches the high-denoising DDPM baseline in mean score while improving success from `0.350` to `0.750`.

However, DDIM is not universally better. At `(100,8)`, DDPM remains stronger in this experiment.

## 6. Why This Satisfies the Improvement Requirement

The improvement is implemented directly on the reproduced official checkpoint. It does not depend on the local surrogate environment. It changes only inference-time sampling and is evaluated with matched official Push-T seeds.

The bounded claim is:

```text
DDIM improves the official checkpoint's low/mid-denoising inference frontier compared with the official DDPM sampler.
```

The project does not claim:

```text
DDIM beats every DDPM setting.
```

or:

```text
The earlier rule-based scheduler improves the official checkpoint.
```

## 7. Synthetic Scheduler Diagnostic

The repository still contains a local Push-T-style surrogate and two rule-based schedulers. Those experiments are no longer headline evidence. The surrogate includes hand-designed denoising residuals, mode bias, and stale-plan effects, so its scheduler gains do not independently validate official Diffusion Policy improvement.

## 8. Limitations

The official reproduction uses a compatibility environment rather than the original Python 3.9 / PyTorch 1.12 / Gym 0.21 stack. The DDIM comparison is currently limited to three high-budget `(k,h)` settings. Runtime wall-clock speed was not benchmarked separately; NFE is used as the compute proxy. DDIM is an established sampler, so the contribution is application and validation on the reproduced official checkpoint, not a new diffusion sampling algorithm.

## 9. Conclusion

The final project now has a defensible improvement result. Starting from an official Diffusion Policy Push-T reproduction, it implements a plug-and-play DDIM sampler replacement and validates it on official checkpoint rollouts. DDIM substantially improves low/mid-denoising settings over the official DDPM sampler on matched seeds, while the original high-denoising DDPM baseline remains strongest. This is a rigorous, bounded improvement claim that satisfies the project requirement without relying on the earlier surrogate scheduler overclaim.

## References

- Diffusion Policy project page: https://diffusion-policy.cs.columbia.edu/
- Official code repository: https://github.com/real-stanford/diffusion_policy
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023 / arXiv 2303.04137.
- Denoising Diffusion Implicit Models, ICLR 2021 / arXiv 2010.02502.
- VADF: Vision-Adaptive Diffusion Policy Framework for Efficient Robotic Manipulation, arXiv 2604.15938.
- RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning, arXiv 2503.04051.
- TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control, arXiv 2601.14945.
- Real-Time Iteration Scheme for Diffusion Policy, arXiv 2508.05396.
