# Joint Compute-Control Budgeting for Receding-Horizon Diffusion Policy

This final project studies the inference-time tradeoff in Diffusion Policy-style receding-horizon control:

- `k`: denoising steps per policy call.
- `h`: executed control steps before replanning.
- `B ~= k / h`: amortized inference budget per control step.

The reproduction target is **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (RSS 2023). The original Stanford/Columbia source and checkpoint are included under `official_reproduction/`, and the official low-dimensional Push-T checkpoint evaluation was executed in this container. The main improvement experiment uses a deterministic Push-T-style closed-loop surrogate benchmark because the proposed contribution changes inference-time allocation variables `(k,h)` that are easier to isolate and sweep reproducibly in a controlled runner.

An official reproduction wrapper is included at `scripts/run_official_pusht_eval.py`. It loads the official checkpoint, instantiates the upstream workspace and PushT keypoints runner, and allows short or 50-seed evaluation overrides. The preferred 50-seed official run used `diffusers==0.11.1` and produced mean score `0.919`. The checkpoint filename reports `test_mean_score=0.969`; the observed gap is recorded as an environment-version reproduction gap because the current container uses Python 3.12, PyTorch 2.7, and Gym 0.26 compatibility patches rather than the original Python 3.9 / PyTorch 1.12 stack.

## Official Push-T Reproduction

- Source tree: `official_reproduction/diffusion_policy_source`
- Checkpoint: `official_reproduction/data/epoch=0550-test_mean_score=0.969.ckpt`
- Checkpoint size: `1044185793` bytes.
- 4-seed smoke score: `0.997`.
- 50-seed score: `0.919`.
- 50-seed log: `/root/FinalProject/official_reproduction/pusht_eval_official_n50_diffusers011/eval_log.json`.
- Compatibility patches: `shared_memory=False`, Gym reset wrapper, Gym vector concatenate argument order.

## Official `(k,h)` Frontier Check

The optimized project also runs fixed `(k,h)` overrides directly through the official checkpoint with `scripts/run_official_kh_grid.py`. On a 20-seed high-budget frontier, the best tested official point is `(k=100, h=8)`, with mean score `0.949`, success `0.950`, and `38.0` policy calls per episode. Lower-denoising points on the same high-budget family fail sharply, so the official checkpoint evidence is that denoising depth remains critical for this pretrained model.

## Near-Official LeRobot Reproduction

- Model: `lerobot/diffusion_pusht`.
- HF model-card 500-episode evaluation: average max reward `0.955`, success `65.4%`.
- Model-card comparison to the original Diffusion Policy repository: average max reward `0.957`, success `64.2%`.
- Local checkpoint status: `1002.2` MiB safetensors file downloaded through `hf-mirror.com`, safetensors structure verified, LeRobot `DiffusionPolicy` loaded with 262.7M parameters, real `gym-pusht` environment smoke tested.
- Local rollout smoke test: 1 episode, 300 steps, `100` denoising steps, max reward `0.627`, success rate `0.000`.

Important caveat: the mirror-assembled safetensors file loads and runs, but its SHA did not match the HF LFS metadata. Therefore the HF model-card metrics are treated as the near-official benchmark reference, and the local rollout is reported as an executable smoke test.

## Main Result

On the `B ~= 2` frontier with 20 matched seeds:

- Best fixed score baseline `(k=2, h=1)`: score `0.871`, success `0.900`, policy calls `120.0`, smoothness cost `0.420`.
- Score-oriented joint scheduler: score `0.893`, success `0.850`, policy calls `89.2`, smoothness cost `0.257`.
- Safe joint scheduler: score `0.863`, success `0.950`, policy calls `105.3`, smoothness cost `0.340`.

The two scheduler variants expose a Pareto tradeoff. The score-oriented scheduler improves mean score while reducing calls and action roughness; the safe scheduler improves success rate while still reducing calls and smoothness cost relative to the best fixed score baseline.

## Reproduce

```bash
cd /root/FinalProject
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache python scripts/eval_kh_grid.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --episode-steps 120
python scripts/plot_iso_compute.py
python scripts/generate_deliverables.py
```

Official checkpoint evaluation path, for a network-stable run:

```bash
cd /root/FinalProject
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache   /root/FinalProject/.venv_official/bin/python scripts/run_official_pusht_eval.py   --output-dir official_reproduction/pusht_eval_official_n50_diffusers011   --n-test 50 --n-envs 5 --n-test-vis 0 --max-steps 300
```

Official fixed `(k,h)` frontier:

```bash
cd /root/FinalProject
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache   /root/FinalProject/.venv_official/bin/python scripts/run_official_kh_grid.py   --output-root official_reproduction/pusht_official_kh_grid_highB_n20   --n-test 20 --n-envs 5 --pairs 12,1 25,2 50,4 100,8
```

## Outputs

- Raw and summarized results: `results/`
- Figures: `figures/`
- Main comparison bar plot: `figures/main_comparison_bars.png`
- Rollout animation: `artifacts/videos/joint_scheduler_rollout.gif`
- Environment log: `artifacts/logs/environment_log.md`
- Completion audit: `artifacts/logs/completion_audit.md`
- Official reproduction status: `results/official_reproduction_status.json`
- Official `(k,h)` frontier: `results/official_kh_grid_results.csv` and `figures/official_kh_frontier.png`
- Official fixed `(k,h)` runner: `scripts/run_official_kh_grid.py`
- Official checkpoint eval wrapper: `scripts/run_official_pusht_eval.py`
- Official reference notes: `official_reproduction/REFERENCE_NOTES.md`
- LeRobot PushT smoke results: `results/lerobot_pusht_smoke_summary.json` and `results/lerobot_pusht_smoke_results.csv`
- LeRobot PushT smoke video: `artifacts/videos/lerobot_pusht_episode_00.gif`
- Final report: `report/final_report.pdf` and `report/final_report.md`
- Slides: `slides/final_presentation.pptx` and `slides/final_presentation.pdf`

## Important Limitation

The scheduler-improvement numbers are from the local Push-T-style surrogate, not official Diffusion Policy checkpoint rollouts. The report states this explicitly and frames the result as a reproducible inference-time analysis layer built on top of an official Push-T checkpoint reproduction.
