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
MIN_FREE_GB="${MIN_FREE_GB:-5}"
DATASET_CACHE="${DATASET_CACHE:-/workspace/teutonic-mining/cache/datasets}"
DATASETS_CONFIG="${DATASETS_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/datasets.json}"
STEP2_PER_DEVICE_BATCH_SIZE="${STEP2_PER_DEVICE_BATCH_SIZE:-8}"
STEP2_LM_HEAD_CHUNK="${STEP2_LM_HEAD_CHUNK:-512}"
STEP2_EMPTY_CACHE_EVERY="${STEP2_EMPTY_CACHE_EVERY:-0}"
STEP2_ATTN_IMPLEMENTATION="${STEP2_ATTN_IMPLEMENTATION:-auto}"
# Comma-separated CUDA devices are supported, e.g. DEVICE=0,1.
# Default `auto` scores on every visible CUDA GPU.
DEVICE="${USER_DEVICE:-auto}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  "${PYTHON_BIN}" scripts/mining/step2_score_samples.py
  --work "${WORK_DIR}"
  --king-dir "${KING_DIR}"
  --n-score "${N_SCORE}"
  --device "${DEVICE}"
  --download-workers "${DOWNLOAD_WORKERS}"
  --min-free-gb "${MIN_FREE_GB}"
  --dataset-cache "${DATASET_CACHE}"
  --per-device-batch-size "${STEP2_PER_DEVICE_BATCH_SIZE}"
  --lm-head-chunk "${STEP2_LM_HEAD_CHUNK}"
  --empty-cache-every "${STEP2_EMPTY_CACHE_EVERY}"
  --attn-implementation "${STEP2_ATTN_IMPLEMENTATION}"
)

if [[ -n "${USER_SEED}" ]]; then
  cmd+=(--seed "${USER_SEED}")
fi
append_optional_model_args cmd
cmd+=(--datasets-config "${DATASETS_CONFIG}")

run_cmd "${cmd[@]}"
