#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-4.3GB}"

run_cmd "${PYTHON_BIN}" scripts/mining/step5_merge_lora.py \
  --work "${WORK_DIR}" \
  --king-dir "${KING_DIR}" \
  --merged-dir "${MERGED_DIR}" \
  --max-shard-size "${MAX_SHARD_SIZE}"
