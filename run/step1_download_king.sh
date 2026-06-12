#!/usr/bin/env bash
set -euo pipefail

# KING_SOURCE=dashboard|hippius|hf
# dashboard = live current king. hippius/hf require KING_REPO or KING_MODEL_URL.
KING_SOURCE="${KING_SOURCE:-dashboard}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

case "${KING_SOURCE}" in
  dashboard)
    cmd=(
      "${PYTHON_BIN}" scripts/mining/step1_download_king.py
      --work "${WORK_DIR}"
      --king-dir "${KING_DIR}"
      --download-workers "${DOWNLOAD_WORKERS}"
    )
    ;;
  hippius)
    cmd=(
      "${PYTHON_BIN}" scripts/mining/step1_1_download_king.py
      --work "${WORK_DIR}"
      --king-dir "${KING_DIR}"
      --download-workers "${DOWNLOAD_WORKERS}"
    )
    append_optional_model_args cmd
    ;;
  hf)
    cmd=(
      "${PYTHON_BIN}" scripts/mining/step1_2_download_king_hf.py
      --work "${WORK_DIR}"
      --king-dir "${KING_DIR}"
      --download-workers "${DOWNLOAD_WORKERS}"
    )
    append_optional_model_args cmd
    ;;
  *)
    echo "Unknown KING_SOURCE=${KING_SOURCE}; expected dashboard, hippius, or hf" >&2
    exit 2
    ;;
esac

run_cmd "${cmd[@]}"
