#!/usr/bin/env bash
set -euo pipefail

# Compare a merged fine-tuned model against a freshly downloaded current king.
# Override CHALLENGER_DIR if your model lives somewhere else.

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHALLENGER_DIR="${CHALLENGER_DIR:-${MERGED_DIR}}"
COMPARE_KING_DIR="${COMPARE_KING_DIR:-${WORK_DIR}/current_king}"
COMPARE_KING_METADATA="${COMPARE_KING_METADATA:-${WORK_DIR}/current_king.json}"
COMPARISON_OUT="${COMPARISON_OUT:-${WORK_DIR}/comparison.json}"
N_EVAL="${N_EVAL:-2000}"
EVAL_N_SHARDS_PER_DATASET="${EVAL_N_SHARDS_PER_DATASET:-1}"
EVAL_SHARD_START="${EVAL_SHARD_START:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
TRAIN_SUMMARY="${TRAIN_SUMMARY:-${WORK_DIR}/score_summary.json}"

cmd=(
  "${PYTHON_BIN}" scripts/mining/compare_with_king.py
  --work "${WORK_DIR}"
  --king-dir "${COMPARE_KING_DIR}"
  --king-metadata-out "${COMPARE_KING_METADATA}"
  --download-workers "${DOWNLOAD_WORKERS}"
  --refresh-current-king
  --challenger-dir "${CHALLENGER_DIR}"
  --n-shards-per-dataset "${EVAL_N_SHARDS_PER_DATASET}"
  --shard-start "${EVAL_SHARD_START}"
  --n-eval "${N_EVAL}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --train-summary "${TRAIN_SUMMARY}"
  --comparison-out "${COMPARISON_OUT}"
)

append_random_shards_arg cmd
if [[ -n "${DATASETS_CONFIG:-}" ]]; then
  cmd+=(--datasets-config "${DATASETS_CONFIG}")
fi
if [[ "${INCLUDE_TRAIN_SHARDS:-0}" == "1" || "${INCLUDE_TRAIN_SHARDS:-0}" == "true" ]]; then
  cmd+=(--include-train-shards)
fi

run_cmd "${cmd[@]}"
