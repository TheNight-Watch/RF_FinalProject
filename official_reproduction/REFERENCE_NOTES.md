# Official Diffusion Policy Reproduction Notes

Source repository: https://github.com/real-stanford/diffusion_policy

Project page: https://diffusion-policy.cs.columbia.edu/

Paper: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023.

## Official Evaluation Path

The upstream README provides a pre-trained low-dimensional Push-T checkpoint:

```text
https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt
```

It also gives the evaluation command pattern:

```bash
python eval.py --checkpoint data/0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output --device cuda:0
```

The local helper script [../scripts/run_official_diffusion_policy_eval.sh](../scripts/run_official_diffusion_policy_eval.sh) implements this path with:

- a vGPU cache workaround for this container,
- shallow clone of the official repository,
- resumable checkpoint download,
- evaluation output under `official_reproduction/pusht_eval_output`,
- logs under `artifacts/logs/official_diffusion_policy_eval.log`.

## Official Environment Mismatch

The official conda environment is named `robodiff` and targets:

- Python 3.9,
- PyTorch 1.12.1,
- CUDA toolkit 11.6,
- Gym 0.21,
- pymunk 6.2.1,
- Hydra 1.2.0,
- wandb 0.13.3.

The available project container is Python 3.12 with PyTorch 2.7.0 and CUDA/vGPU runtime for an RTX 5090. This is useful for local experiments but not ABI-compatible with the original environment without creating a separate conda/mamba environment.

## Current Reproduction Status

The original `git clone` path was unreliable, but the official source was later acquired through the GitHub
`codeload` archive and extracted to:

- `official_reproduction/diffusion_policy_source`

The official checkpoint was downloaded successfully to:

- `official_reproduction/data/epoch=0550-test_mean_score=0.969.ckpt`
- size: `1044185793` bytes

The official PushT keypoints runner was executed with the official checkpoint and EMA policy. Because the
available container is Python 3.12 with modern Gym/PyTorch, the local source copy includes compatibility patches:

- use `shared_memory=False` in the PushT keypoints async vector runner,
- provide an old-API `reset()` wrapper for Gym 0.26,
- update `gym.vector.utils.concatenate` call order for Gym 0.26.

Recorded official-run outputs:

- `official_reproduction/pusht_eval_official_n4/eval_log.json`: 4 test seeds, mean score `0.9964484097146112`.
- `official_reproduction/pusht_eval_official_n50/eval_log.json`: 50 test seeds, mean score `0.917716613377729`.
- `official_reproduction/pusht_eval_official_n4_diffusers011/eval_log.json`: 4 test seeds with `diffusers==0.11.1`, mean score `0.9968081815172856`.
- `official_reproduction/pusht_eval_official_n50_diffusers011/eval_log.json`: 50 test seeds with `diffusers==0.11.1`, mean score `0.9186505414953691`.

The 50-seed scores are below the checkpoint filename score `0.969`. This should be reported as a measured
environment-version reproduction gap rather than a download or execution failure. The controlled improvement
experiment remains separated in `results/summary_results.csv` and uses the local Push-T-style surrogate.

## Official Fixed `(k,h)` Frontier

The optimized project also evaluates fixed `(k,h)` overrides with the official checkpoint through:

- `scripts/run_official_kh_grid.py`
- `results/official_kh_grid_results.csv`
- `figures/official_kh_frontier.png`

The 20-seed high-budget frontier run used:

```bash
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache \
  /root/FinalProject/.venv_official/bin/python scripts/run_official_kh_grid.py \
  --output-root official_reproduction/pusht_official_kh_grid_highB_n20 \
  --n-test 20 --n-envs 5 --pairs 12,1 25,2 50,4 100,8
```

Recorded official fixed-frontier outputs:

- `(k=12, h=1)`: mean score `0.10652993045961456`, success `0.0`.
- `(k=25, h=2)`: mean score `0.15480444475860594`, success `0.0`.
- `(k=50, h=4)`: mean score `0.6531389287361444`, success `0.35`.
- `(k=100, h=8)`: mean score `0.9485482200015596`, success `0.95`.

This official checkpoint result is intentionally reported as a supplementary frontier rather than the main
scheduler-improvement result. It shows that, for the pretrained official checkpoint, low denoising depth cannot
be compensated by frequent replanning at the tested high-budget frontier.
