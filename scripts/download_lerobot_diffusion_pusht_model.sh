#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/official_reproduction/hf_lerobot_diffusion_pusht"
TMP_DIR="${OUT_DIR}/model_chunks"
URL="${HF_MIRROR_URL:-https://hf-mirror.com/lerobot/diffusion_pusht/resolve/main/model.safetensors}"
OUT="${OUT_DIR}/model.safetensors"
SIZE=1050862408
SHA256="995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8"
CHUNK_SIZE="${CHUNK_SIZE:-33554432}"
JOBS="${JOBS:-8}"

mkdir -p "${TMP_DIR}"

if [ -s "${OUT}" ]; then
  echo "Existing model found: ${OUT}"
  sha256sum "${OUT}"
  exit 0
fi

echo "Downloading ${SIZE} bytes from ${URL}"
echo "chunk_size=${CHUNK_SIZE} jobs=${JOBS}"

manifest="${TMP_DIR}/manifest.txt"
: > "${manifest}"
i=0
start=0
while [ "${start}" -lt "${SIZE}" ]; do
  end=$((start + CHUNK_SIZE - 1))
  if [ "${end}" -ge "$((SIZE - 1))" ]; then
    end=$((SIZE - 1))
  fi
  printf "%06d %d %d %s/part_%06d\n" "${i}" "${start}" "${end}" "${TMP_DIR}" "${i}" >> "${manifest}"
  i=$((i + 1))
  start=$((end + 1))
done

download_one() {
  idx="$1"
  start="$2"
  end="$3"
  path="$4"
  expected=$((end - start + 1))
  if [ -s "${path}" ] && [ "$(stat -c%s "${path}")" -eq "${expected}" ]; then
    echo "chunk ${idx} exists"
    return 0
  fi
  partial=0
  if [ -s "${path}.tmp" ]; then
    partial="$(stat -c%s "${path}.tmp")"
    if [ "${partial}" -ge "${expected}" ]; then
      actual="${partial}"
    else
      actual=0
    fi
  else
    actual=0
  fi
  if [ "${actual}" -eq "${expected}" ]; then
    mv "${path}.tmp" "${path}"
    echo "chunk ${idx} completed from tmp"
    return 0
  fi
  adjusted_start=$((start + partial))
  if [ "${adjusted_start}" -gt "${end}" ]; then
    echo "chunk ${idx} partial file is larger than expected" >&2
    return 1
  fi
  echo "chunk ${idx}: bytes ${adjusted_start}-${end} (partial=${partial})"
  curl -L --fail --no-progress-meter --retry 12 --retry-delay 5 --connect-timeout 30 \
    --speed-limit 2048 --speed-time 120 \
    -r "${adjusted_start}-${end}" -o "${path}.more" "${URL}"
  cat "${path}.more" >> "${path}.tmp"
  rm -f "${path}.more"
  actual="$(stat -c%s "${path}.tmp")"
  if [ "${actual}" -ne "${expected}" ]; then
    echo "chunk ${idx} size mismatch: expected ${expected}, got ${actual}" >&2
    return 1
  fi
  mv "${path}.tmp" "${path}"
}
export -f download_one
export URL

xargs -n 4 -P "${JOBS}" bash -c 'download_one "$@"' _ < "${manifest}"

tmp_out="${OUT}.tmp"
: > "${tmp_out}"
while read -r idx start end path; do
  cat "${path}" >> "${tmp_out}"
done < "${manifest}"

actual_size="$(stat -c%s "${tmp_out}")"
if [ "${actual_size}" -ne "${SIZE}" ]; then
  echo "assembled size mismatch: expected ${SIZE}, got ${actual_size}" >&2
  exit 1
fi

actual_sha="$(sha256sum "${tmp_out}" | awk '{print $1}')"
if [ "${actual_sha}" != "${SHA256}" ]; then
  echo "sha256 mismatch: expected ${SHA256}, got ${actual_sha}" >&2
  exit 1
fi

mv "${tmp_out}" "${OUT}"
echo "Downloaded and verified ${OUT}"
