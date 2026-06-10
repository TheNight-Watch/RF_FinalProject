# Joint Compute-Control Budgeting for Receding-Horizon Diffusion Policy

## Abstract

Diffusion Policy produces action chunks through iterative denoising and deploys them in a receding-horizon loop. This project studies a practical inference question: under a fixed per-step compute budget, should a robot spend computation on more denoising steps or on more frequent replanning? We first reproduce the official low-dimensional Push-T checkpoint using the upstream workspace and runner, obtaining a 50-seed mean score of `0.919` in the current Python 3.12 / PyTorch 2.7 container. We then evaluate the proposed inference-time idea in a lightweight Push-T-style closed-loop surrogate and sweep denoising steps `k` and execution horizon `h`. A training-free joint scheduler chooses `(k,h)` on the same `B ~= k/h` frontier. On the `B ~= 2` frontier with 20 matched seeds, the score-oriented scheduler obtains score `0.893` versus `0.871` for the best fixed score baseline `(2,1)`, while reducing policy calls from `120.0` to `89.2` and smoothness cost from `0.420` to `0.257`. A conservative safe scheduler reaches success `0.950` versus `0.900` for `(2,1)`. A supplementary official-checkpoint frontier shows that the pretrained official policy remains strongly denoising-limited at low `k`: the best tested official point is `(k=100, h=8)`, score `0.949`, success `0.950`.

## 1. Introduction

Diffusion Policy is a strong visuomotor imitation learning method because iterative action denoising can represent multimodal action distributions and produce coherent action chunks. Deployment introduces a resource allocation problem. More denoising steps can improve action quality, but each policy call becomes more expensive. More frequent replanning improves reactivity, especially near contact, but increases the number of policy calls. This project formulates that tradeoff as joint compute-control budgeting.

## 2. Related Work

The reproduced base paper is **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (RSS 2023). The method generates action sequences with a conditional diffusion model and executes them in a receding-horizon control loop. ACT shows why action chunks can stabilize imitation learning, while newer accelerated-policy work such as Consistency Policy focuses on reducing diffusion-policy inference latency through distillation. DDIM and DPM-Solver establish the broader diffusion-model theme that sampling step count can often be traded against quality. Recent component analyses of Diffusion Policy also isolate receding horizon and action-sequence execution as major design choices. The novelty boundary here is intentionally narrower: the project evaluates denoising and replanning jointly under an iso-compute constraint and compares a joint scheduler against fixed and single-axis baselines.

## 3. Problem Formulation

Let `k` be the number of denoising steps per policy call and `h` be the number of controls executed before the next replan. The amortized inference budget is approximated as `B ~= k / h`. The static grid uses `k in {2,4,8,16}` and `h in {1,2,4,8}`, with main frontiers `B=1`, `B=2`, and `B=4`. The main comparison fixes the average budget near `B=2`.

## 4. Reproduction Setup

The original Diffusion Policy repository was checked by `git ls-remote`. Full and shallow `git clone` attempts were intermittent, so the official source was acquired through the GitHub codeload archive and extracted to `official_reproduction/diffusion_policy_source`. The official low-dimensional Push-T checkpoint was downloaded from the project server to `official_reproduction/data/epoch=0550-test_mean_score=0.969.ckpt`; the file size is `1044185793` bytes. PyTorch 2.7.0 with the RTX 5090 vGPU works when `CUDA_DEVICE_MEMORY_SHARED_CACHE` is redirected to `/tmp`; the default cache path triggered a vGPU runtime failure during initial testing.

The upstream checkpoint was evaluated with `scripts/run_official_pusht_eval.py`, which instantiates the original workspace, EMA policy, and PushT keypoints runner. Because this container is newer than the original conda environment, the local source copy includes compatibility patches for Gym 0.26: disabling shared-memory batching for the custom PushT observation space, restoring the old reset return contract, and updating vector concatenate argument order. The preferred 50-seed run used official-era `diffusers==0.11.1` and obtained mean score `0.919`. The checkpoint filename reports `test_mean_score=0.969`, so the remaining gap is treated as an environment-version reproduction gap rather than an execution failure.

As a supplementary path, a Hugging Face / LeRobot checkpoint was also tested. The `lerobot/diffusion_pusht` model card reports a 500-episode PushT evaluation with average max reward `0.955` and success `65.4%`; the same card lists the original Diffusion Policy repository comparison at average max reward `0.957` and success `64.2%`. Official HF transfer of the 1.0 GB safetensors file was slow, so `hf-mirror.com` and chunked HTTP range requests were used. The assembled file has the expected size, can be opened as safetensors, and loads into LeRobot `DiffusionPolicy` with 262.7M parameters. Its SHA did not match the HF LFS metadata, so local LeRobot rollouts are reported as executable smoke tests rather than cryptographically verified benchmark runs.

The local LeRobot smoke test ran the real `gym-pusht` environment for `1` episode, `300` steps, and `100` denoising steps. It reached max reward `0.627` and success rate `0.000`. The improvement experiment below uses a deterministic Push-T-style surrogate environment with closed-loop contact dynamics, action chunks, denoising-quality noise, and stale-plan effects.

The surrogate is not a substitute for official benchmark numbers. It is a controlled reproduction of the deployment mechanism needed for the proposed improvement: action diffusion quality versus receding-horizon replanning frequency.

For a reproduced official run in this project folder:

```bash
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache   /root/FinalProject/.venv_official/bin/python scripts/run_official_pusht_eval.py   --output-dir official_reproduction/pusht_eval_official_n50_diffusers011   --n-test 50 --n-envs 5 --n-test-vis 0 --max-steps 300
```

The official reproduction reference note in `official_reproduction/REFERENCE_NOTES.md` records the upstream checkpoint path, command pattern, and original conda environment mismatch.

The optimized project additionally evaluates fixed `(k,h)` overrides inside the official checkpoint runner with `scripts/run_official_kh_grid.py`. On a 20-seed high-budget frontier, the tested points have mean scores `0.107` for `(12,1)`, `0.155` for `(25,2)`, `0.653` for `(50,4)`, and `0.949` for `(100,8)`. This result is important because it prevents overclaiming: the surrogate benefits from low-`k` frequent replanning, but the official pretrained checkpoint needs sufficient denoising depth before replanning frequency matters.

## 5. Method

The fixed runner evaluates every `(k,h)` pair. Larger `k` reduces residual action noise and mode bias in the diffusion-like planner. Larger `h` reduces policy calls but increases open-loop staleness in contact. The proposed training-free scheduler is restricted to the `B=2` candidate set and uses state features available in the simulator:

- pusher-object distance,
- object-goal distance,
- contact proxy,
- recent progress.

The score-oriented rule selects `(4,2)` in free-space approach and final alignment, and `(2,1)` during close-contact pushing or unstable progress. A conservative safe variant uses the same candidate set but switches to `(2,1)` earlier when contact or negative progress is detected. Both keep average `B` near 2 while shifting control frequency toward high-risk phases.

## 6. Results

| Method | B | Score | Success | Calls | NFE | Smoothness |
| --- | --- | --- | --- | --- | --- | --- |
| Default DP-style (8,4) | 2.00 | 0.704 | 0.550 | 30.0 | 240.0 | 0.035 |
| Best fixed B=2 (2,1) | 2.00 | 0.871 | 0.900 | 120.0 | 240.0 | 0.420 |
| Denoising-only point (4,2) | 2.00 | 0.756 | 0.700 | 60.0 | 240.0 | 0.105 |
| Long denoise/chunk (16,8) | 2.00 | 0.523 | 0.200 | 15.0 | 240.0 | 0.009 |
| AAC/DVAC-style h-only | 2.60 | 0.728 | 0.550 | 39.0 | 312.4 | 0.046 |
| Joint scheduler (score) | 2.00 | 0.893 | 0.850 | 89.2 | 240.6 | 0.257 |
| Joint scheduler (safe) | 2.00 | 0.863 | 0.950 | 105.3 | 240.3 | 0.340 |

The static grid shows that compute-equivalent choices are not interchangeable. On the `B=2` frontier, `(2,1)` gives the highest fixed score because the surrogate task is contact sensitive and benefits from frequent replanning. However, `(2,1)` also creates the largest action roughness. The score-oriented scheduler preserves high-frequency updates near contact while using `(4,2)` in easier phases. This raises mean score to `0.893` and reduces smoothness cost by about `38.7%` relative to `(2,1)`. The safe scheduler trades some mean score for reliability, reaching success `0.950` while still reducing policy calls from `120.0` to `105.3`.

## 7. Phase-Wise Analysis

The phase-wise plot separates free-space approach, contact pushing, and near-goal alignment. Fixed long-horizon settings are efficient but lose score when contact starts. Fixed short-horizon replanning is robust but noisy. The scheduler's benefit comes from recognizing that free-space and alignment tolerate a longer action chunk, while contact needs immediate feedback.

## 8. Limitations

The largest limitation is that the official checkpoint evaluation runs in a compatibility environment, not the original Python 3.9 / PyTorch 1.12 conda stack, and its 50-seed score is below the checkpoint filename score. A LeRobot checkpoint was also downloaded through an alternate mirror, loaded, and executed locally, but the mirror-assembled file did not pass HF LFS SHA verification. The scheduler-improvement environment is a surrogate, so those numeric values should not be reported as official Diffusion Policy benchmark performance. The official `(k,h)` frontier is therefore included as a sanity check and shows model-dependent behavior: low-denoising frequent replanning is not enough for the official checkpoint. The scheduler is hand-tuned on the same task family, the compute model uses `k/h` rather than measured neural network latency, and there is no real-robot validation.

## 9. Conclusion

The experiment supports the project hypothesis: under the same amortized inference budget, the best allocation can depend on task phase. Jointly scheduling denoising and replanning can outperform or match a tuned fixed budget point while improving execution smoothness. For deployment, the practical lesson is to treat action diffusion quality and receding-horizon frequency as one coupled budget rather than two independent hyperparameters.

## References

- Diffusion Policy project page: https://diffusion-policy.cs.columbia.edu/
- Official code repository: https://github.com/real-stanford/diffusion_policy
- LeRobot model card used for near-official PushT reference: https://huggingface.co/lerobot/diffusion_pusht
- Paper: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023 / arXiv 2303.04137.
- Official checkpoint example from the repository README: `low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt`.
- Action Chunking with Transformers: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, RSS 2023 / arXiv 2304.13705.
- Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation, RSS 2024 / arXiv 2405.07503.
- Denoising Diffusion Implicit Models, ICLR 2021 / arXiv 2010.02502.
- DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps, NeurIPS 2022 / arXiv 2206.00927.
- Unpacking the Individual Components of Diffusion Policy, arXiv 2412.00084.
