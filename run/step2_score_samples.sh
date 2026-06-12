#!/usr/bin/env bash
set -euo pipefail

# Defaults to the built-in dataset plan:
# automathtext-v2 10 shards x 740 samples
# quasar-sn3       1 shard  x 6400 samples
# ultradata-math   4 shards x 600 samples
# finewebedu       6 shards x 500 samples

USER_SEED="${SEED:-}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

N_SCORE="${N_SCORE:-19200}"
SHARD_START="${SHARD_START:-0}"
RANDOM_SHARDS="${RANDOM_SHARDS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
CACHE_ONLY="${CACHE_ONLY:-1}"

cmd=(
  "${PYTHON_BIN}" scripts/mining/step2_score_samples.py
  --work "${WORK_DIR}"
  --king-dir "${KING_DIR}"
  --shard-start "${SHARD_START}"
  --n-score "${N_SCORE}"
  --device "${DEVICE}"
  --download-workers "${DOWNLOAD_WORKERS}"
  --min-free-gb "${MIN_FREE_GB}"
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
