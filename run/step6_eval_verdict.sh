#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

N_EVAL="${N_EVAL:-2000}"
EVAL_N_SHARDS_PER_DATASET="${EVAL_N_SHARDS_PER_DATASET:-1}"
EVAL_SHARD_START="${EVAL_SHARD_START:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
TRAIN_SUMMARY="${TRAIN_SUMMARY:-${WORK_DIR}/score_summary.json}"

cmd=(
  "${PYTHON_BIN}" scripts/mining/step6_eval_verdict.py
  --work "${WORK_DIR}"
  --king-dir "${KING_DIR}"
  --merged-dir "${MERGED_DIR}"
  --n-shards-per-dataset "${EVAL_N_SHARDS_PER_DATASET}"
  --shard-start "${EVAL_SHARD_START}"
  --n-eval "${N_EVAL}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --train-summary "${TRAIN_SUMMARY}"
)

append_random_shards_arg cmd
if [[ -n "${DATASETS_CONFIG:-}" ]]; then
  cmd+=(--datasets-config "${DATASETS_CONFIG}")
fi

run_cmd "${cmd[@]}"
