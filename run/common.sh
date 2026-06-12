#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${TEUTONIC_ROOT:-}" ]]; then
  ROOT_DIR="$(cd "${TEUTONIC_ROOT}" && pwd)"
elif [[ -f "${SCRIPT_DIR}/../pyproject.toml" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/../teutonic/pyproject.toml" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../teutonic" && pwd)"
else
  echo "Could not locate teutonic project root." >&2
  echo "Set TEUTONIC_ROOT=/path/to/teutonic and rerun." >&2
  exit 1
fi

VENV_DIR="${TEUTONIC_VENV:-${ROOT_DIR}/.venv}"
WORK_DIR="${TEUTONIC_WORK:-/workspace/teutonic-mining/work-stepwise}"
BUNDLE_DIR="${TEUTONIC_BUNDLE:-${ROOT_DIR}/scripts/training_bundle}"
PYTHON_BIN="${PYTHON_BIN:-python}"

KING_DIR="${KING_DIR:-${WORK_DIR}/king}"
LORA_OUT_DIR="${LORA_OUT_DIR:-${WORK_DIR}/lora_out}"
MERGED_DIR="${MERGED_DIR:-${WORK_DIR}/merged}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"

cd "${ROOT_DIR}"
mkdir -p "${WORK_DIR}"

if [[ -d "${VENV_DIR}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
elif [[ "${AUTO_SETUP:-0}" == "1" ]]; then
  "${SCRIPT_DIR}/setup_finetune_env.sh"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "Virtualenv not found at ${VENV_DIR}." >&2
  echo "Run: ${SCRIPT_DIR}/setup_finetune_env.sh" >&2
  exit 1
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts/mining:${PYTHONPATH:-}"

run_cmd() {
  echo
  printf '[run]'
  printf ' %q' "$@"
  echo
  "$@"
}

append_optional_model_args() {
  local -n arr_ref=$1
  if [[ -n "${KING_MODEL_URL:-}" ]]; then
    arr_ref+=(--model-url "${KING_MODEL_URL}")
  fi
  if [[ -n "${KING_REPO:-}" ]]; then
    arr_ref+=(--repo "${KING_REPO}")
  fi
  if [[ -n "${KING_REVISION:-}" ]]; then
    arr_ref+=(--revision "${KING_REVISION}")
  fi
  return 0
}

append_random_shards_arg() {
  local -n arr_ref=$1
  if [[ "${RANDOM_SHARDS:-0}" == "1" || "${RANDOM_SHARDS:-0}" == "true" ]]; then
    arr_ref+=(--random-shards)
  fi
  return 0
}

append_step2_shard_selection_arg() {
  local -n arr_ref=$1
  if [[ "${SEQUENTIAL_SHARDS:-0}" == "1" || "${SEQUENTIAL_SHARDS:-0}" == "true" || "${RANDOM_SHARDS:-1}" == "0" || "${RANDOM_SHARDS:-1}" == "false" ]]; then
    arr_ref+=(--sequential-shards)
  else
    arr_ref+=(--random-shards)
  fi
  return 0
}
