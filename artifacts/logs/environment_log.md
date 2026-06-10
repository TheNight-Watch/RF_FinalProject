# Environment Log

Python: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0]
Platform: Linux-5.15.0-94-generic-x86_64-with-glibc2.39

## PyTorch / CUDA
The default container vGPU cache path can make torch import fail. The working invocation is:

```bash
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache python -c "import torch; ..."
```

Observed torch output:
```text
2.7.0a0+7c8ec84dab.nv25.03
True
NVIDIA GeForce RTX 5090
```

## Core Python Packages
```text
numpy 1.26.4
pandas 2.2.3
matplotlib 3.10.1
python-pptx ok
reportlab ok
```

## Official Diffusion Policy Reproduction
The official repository was reachable for metadata:
```text
5ba07ac6661db573af695b419a7947ecb704690f	HEAD
```
Full and shallow git clone attempts were unstable, so the official source was acquired through the GitHub codeload archive and extracted locally.

Official source tree: `/root/FinalProject/official_reproduction/diffusion_policy_source`
Official checkpoint: `/root/FinalProject/official_reproduction/data/epoch=0550-test_mean_score=0.969.ckpt`
Checkpoint size: `1044185793` bytes
50-seed official PushT mean score: `0.919`
Official eval log: `/root/FinalProject/official_reproduction/pusht_eval_official_n50_diffusers011/eval_log.json`

Compatibility patches were required for the current Python 3.12 / Gym 0.26 environment: disable shared-memory batching for the custom PushT observation space, restore the old reset return contract, and update Gym vector concatenate argument order.

## LeRobot / Hugging Face Near-Official Path
After official GitHub clone and large-file transfer attempts remained unstable, a second reproduction path was executed through the LeRobot `lerobot/diffusion_pusht` checkpoint. Official HF transfer was slow, so the large `model.safetensors` file was downloaded through `hf-mirror.com` with chunked range requests. The assembled safetensors file has the expected size and can be opened and loaded by LeRobot, but its SHA did not match the HF LFS metadata, so local checkpoint rollouts are reported as smoke tests rather than cryptographically verified benchmark reproduction.

LeRobot smoke command:
```bash
CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/finalproject-vgpu-cache.cache /root/FinalProject/.venv_lerobot/bin/python scripts/eval_lerobot_pusht_smoke.py --episodes 1 --max-steps 300 --num-inference-steps 100 --save-video
```
