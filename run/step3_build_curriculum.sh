#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRAIN_PER_ITER="${TRAIN_PER_ITER:-110000}"
VAL_SIZE="${VAL_SIZE:-2000}"
SEED="${SEED:-2026}"

cmd=(
  "${PYTHON_BIN}" scripts/mining/step3_build_curriculum.py
  --work "${WORK_DIR}"
  --train-per-iter "${TRAIN_PER_ITER}"
  --val-size "${VAL_SIZE}"
  --seed "${SEED}"
)

run_cmd "${cmd[@]}"
