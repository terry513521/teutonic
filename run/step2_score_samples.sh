#!/usr/bin/env bash
set -euo pipefail

# Defaults to the built-in dataset plan:
# 30,000 samples per dataset source. Non-Quasar sources select up to 10
# random shards and sample 3,000 rows per shard; Quasar keeps dataset-level
# sampling.

USER_SEED="${SEED:-}"
USER_DEVICE="${DEVICE:-}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

N_SCORE="${N_SCORE:-10000}"
SHARD_START="${SHARD_START:-0}"
RANDOM_SHARDS="${RANDOM_SHARDS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
CACHE_ONLY="${CACHE_ONLY:-0}"
DATASET_CACHE="${DATASET_CACHE:-/workspace/teutonic-mining/cache/datasets}"
STEP2_PER_DEVICE_BATCH_SIZE="${STEP2_PER_DEVICE_BATCH_SIZE:-8}"
# Comma-separated CUDA devices are supported, e.g. DEVICE=0,1.
# Default `auto` scores on every visible CUDA GPU.
DEVICE="${USER_DEVICE:-auto}"

cmd=(
  "${PYTHON_BIN}" scripts/mining/step2_score_samples.py
  --work "${WORK_DIR}"
  --king-dir "${KING_DIR}"
  --shard-start "${SHARD_START}"
  --n-score "${N_SCORE}"
  --device "${DEVICE}"
  --download-workers "${DOWNLOAD_WORKERS}"
  --min-free-gb "${MIN_FREE_GB}"
  --dataset-cache "${DATASET_CACHE}"
  --per-device-batch-size "${STEP2_PER_DEVICE_BATCH_SIZE}"
)

append_step2_shard_selection_arg cmd
if [[ "${CACHE_ONLY}" == "1" || "${CACHE_ONLY}" == "true" ]]; then
  cmd+=(--cache-only)
fi
if [[ -n "${USER_SEED}" ]]; then
  cmd+=(--seed "${USER_SEED}")
fi
append_optional_model_args cmd
if [[ -n "${DATASETS_CONFIG:-}" ]]; then
  cmd+=(--datasets-config "${DATASETS_CONFIG}")
fi

run_cmd "${cmd[@]}"
