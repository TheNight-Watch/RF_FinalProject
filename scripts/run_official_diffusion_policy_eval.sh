#!/usr/bin/env bash
set -euo pipefail

# Official Diffusion Policy Push-T checkpoint evaluation helper.
#
# This script is intentionally separate from the lightweight surrogate runner.
# It follows the official repository README path:
#   repo: https://github.com/real-stanford/diffusion_policy
#   checkpoint:
#     https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt
#
# The current container needs CUDA_DEVICE_MEMORY_SHARED_CACHE redirected away
# from /usr/local/vgpu; otherwise torch import can fail in the vGPU runtime.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_DIR="${ROOT}/official_reproduction"
REPO_DIR="${OFFICIAL_DIR}/diffusion_policy"
DATA_DIR="${OFFICIAL_DIR}/data"
OUTPUT_DIR="${OFFICIAL_DIR}/pusht_eval_output"
LOG_DIR="${ROOT}/artifacts/logs"
CKPT_URL="https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt"
CKPT_PATH="${DATA_DIR}/epoch=0550-test_mean_score=0.969.ckpt"
CLONE_TIMEOUT_SEC="${CLONE_TIMEOUT_SEC:-900}"
DOWNLOAD_TIMEOUT_SEC="${DOWNLOAD_TIMEOUT_SEC:-7200}"

mkdir -p "${OFFICIAL_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

export CUDA_DEVICE_MEMORY_SHARED_CACHE="${CUDA_DEVICE_MEMORY_SHARED_CACHE:-/tmp/finalproject-vgpu-cache.cache}"
export PYTHONUNBUFFERED=1

{
  echo "# Official Diffusion Policy Evaluation Attempt"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=${ROOT}"
  echo "cuda_cache=${CUDA_DEVICE_MEMORY_SHARED_CACHE}"
  echo "clone_timeout_sec=${CLONE_TIMEOUT_SEC}"
  echo "download_timeout_sec=${DOWNLOAD_TIMEOUT_SEC}"
  echo
  echo "## Python / Torch"
  python - <<'PY'
import sys
print(sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
except Exception as exc:
    print("torch_probe_failed", repr(exc))
PY
  echo
  echo "## Official repository"
  if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "Cloning official repository..."
    timeout "${CLONE_TIMEOUT_SEC}" git -c http.proxy= -c https.proxy= clone --depth 1 https://github.com/real-stanford/diffusion_policy.git "${REPO_DIR}"
  else
    echo "Repository already exists; fetching latest shallow HEAD..."
    timeout "${CLONE_TIMEOUT_SEC}" git -C "${REPO_DIR}" -c http.proxy= -c https.proxy= fetch --depth 1 origin
  fi
  git -C "${REPO_DIR}" rev-parse HEAD
  echo
  echo "## Checkpoint"
  if [ ! -s "${CKPT_PATH}" ]; then
    echo "Downloading checkpoint to ${CKPT_PATH}"
    curl -L --continue-at - --retry 5 --retry-delay 10 --connect-timeout 30 --max-time "${DOWNLOAD_TIMEOUT_SEC}" \
      -o "${CKPT_PATH}" "${CKPT_URL}"
  else
    echo "Checkpoint already exists: ${CKPT_PATH}"
  fi
  ls -lh "${CKPT_PATH}"
  echo
  echo "## Evaluation command"
  echo "python eval.py --checkpoint ${CKPT_PATH} --output_dir ${OUTPUT_DIR} --device cuda:0"
  cd "${REPO_DIR}"
  python eval.py --checkpoint "${CKPT_PATH}" --output_dir "${OUTPUT_DIR}" --device cuda:0
} 2>&1 | tee "${LOG_DIR}/official_diffusion_policy_eval.log"
