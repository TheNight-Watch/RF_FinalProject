# Official Diffusion Policy Push-T Sampler Improvement

This final project reproduces the official low-dimensional Push-T checkpoint from **Diffusion Policy** and implements a direct inference-time improvement on top of it: replacing the official DDPM ancestral sampler with a plug-and-play DDIM deterministic sampler at evaluation time.

## What This Project Supports

- The official low-dimensional Push-T Diffusion Policy checkpoint was loaded and evaluated with the upstream workspace and runner.
- Fixed `(k,h)` and sampler overrides were tested directly on the official checkpoint, where:
  - `k` is the number of denoising steps per policy call.
  - `h` is the number of executed control steps before replanning.
  - `B ~= k/h` is the nominal amortized inference budget.
- DDIM substantially improves low/mid-denoising official checkpoint performance over DDPM at matched seeds and matched `(k,h)`.
- DDPM `(100,8)` remains the strongest tested high-denoising baseline, so the claim is not that DDIM universally dominates every setting.

## What This Project Does Not Claim

This project no longer claims that its rule-based dynamic scheduler is a novel Diffusion Policy improvement. Related work such as VADF, RA-DP, TIDAL, RTI-DP, and adaptive diffusion replanning already studies closely related adaptive inference, replanning, and scheduling ideas.

The local surrogate scheduler result is retained only as a synthetic diagnostic. It is not independent evidence that the official checkpoint improves under dynamic scheduling.

See [claim_risk_audit_zh.md](/root/FinalProject/artifacts/logs/claim_risk_audit_zh.md) for the stop-loss review.

## Official Push-T Reproduction

- Source tree: `official_reproduction/diffusion_policy_source`
- Checkpoint: `official_reproduction/data/epoch=0550-test_mean_score=0.969.ckpt`
- Checkpoint size: `1044185793` bytes
- 50-seed official runner score: `0.9186505414953691`
- 50-seed log: `official_reproduction/pusht_eval_official_n50_diffusers011/eval_log.json`

The checkpoint filename reports `test_mean_score=0.969`. The local score gap is recorded as an environment-version reproduction gap because this container uses Python 3.12, PyTorch 2.7, and Gym 0.26 compatibility patches rather than the original Python 3.9 / PyTorch 1.12 stack.

## Main Improvement Result

Matched 20-seed official checkpoint comparison:

| `(k,h)` | DDPM Score | DDIM Score | Score Delta | DDPM Success | DDIM Success |
|---|---:|---:|---:|---:|---:|
| `(25,2)` | 0.155 | 0.900 | +0.745 | 0.000 | 0.800 |
| `(50,4)` | 0.653 | 0.929 | +0.276 | 0.350 | 0.750 |
| `(100,8)` | 0.949 | 0.900 | -0.048 | 0.950 | 0.900 |

Paired bootstrap 95% confidence intervals for score deltas:

- `(25,2)`: `[+0.593, +0.870]`
- `(50,4)`: `[+0.056, +0.488]`
- `(100,8)`: `[-0.152, +0.007]`

Interpretation: DDIM is a strong plug-and-play improvement for low/mid-denoising official checkpoint settings, but the original DDPM sampler remains best at `(100,8)` in this test.

## Official DDPM Fixed Frontier

Official checkpoint fixed `(k,h)` frontier, 20 seeds:

| Official `(k,h)` | Nominal `B=k/h` | Score | Success | Policy Calls |
|---|---:|---:|---:|---:|
| `(12,1)` | 12.0 | 0.107 | 0.000 | 300.0 |
| `(25,2)` | 12.5 | 0.155 | 0.000 | 150.0 |
| `(50,4)` | 12.5 | 0.653 | 0.350 | 75.0 |
| `(100,8)` | 12.5 | 0.949 | 0.950 | 38.0 |

Interpretation before sampler replacement: equal or similar nominal compute budgets are not interchangeable for the official checkpoint. Under DDPM, allocating compute toward denoising depth is much more important than frequent replanning in this tested high-budget family.

## Synthetic Diagnostic

The project also contains a local Push-T-style surrogate with rule-based schedulers:

- `joint_scheduler`
- `joint_scheduler_safe`

Those results are not used as official Diffusion Policy evidence. The surrogate hand-designs denoising residuals, mode bias, and stale-plan effects, so it cannot independently validate the scheduler claim.

## Key Files

- Official eval wrapper: `scripts/run_official_pusht_eval.py`
- Official fixed frontier runner: `scripts/run_official_kh_grid.py`
- Official frontier CSV: `results/official_kh_grid_results.csv`
- Official frontier figure: `figures/official_kh_frontier.png`
- Official sampler comparison summary: `results/official_sampler_comparison_summary.csv`
- Official sampler comparison figure: `figures/official_sampler_comparison.png`
- Claim-risk audit: `artifacts/logs/claim_risk_audit_zh.md`
- Completion audit: `artifacts/logs/completion_audit.md`
- Chinese report: `report/final_report_zh.md`
- English report: `report/final_report.md`

## Reproduce Official Frontier

```bash
cd /root/FinalProject
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache \
  /root/FinalProject/.venv_official/bin/python scripts/run_official_kh_grid.py \
  --output-root official_reproduction/pusht_official_kh_grid_highB_n20 \
  --n-test 20 --n-envs 5 --pairs 12,1 25,2 50,4 100,8
```

## Reproduce DDIM Improvement

```bash
cd /root/FinalProject
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache \
  /root/FinalProject/.venv_official/bin/python scripts/run_official_kh_grid.py \
  --sampler ddim \
  --output-root official_reproduction/pusht_official_sampler_ddim_n20 \
  --csv-out results/official_sampler_ddim_n20.csv \
  --n-test 20 --n-envs 5 --pairs 25,2 50,4 100,8

python scripts/analyze_official_sampler_comparison.py
```

## Conservative Final Claim

The defensible final claim is:

> This project reproduces the official Diffusion Policy Push-T checkpoint and implements a plug-and-play DDIM sampler replacement. On matched official Push-T seeds, DDIM dramatically improves low/mid-denoising settings over the official DDPM sampler, while the original high-denoising DDPM baseline remains strongest at `(100,8)`.
