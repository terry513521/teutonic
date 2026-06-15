#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${DATASETS_CONFIG:-${SCRIPT_DIR}/datasets.json}"
CACHE="${DATASET_CACHE:-/workspace/teutonic-mining/cache/datasets}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-8}"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-auto}"
DOWNLOAD_CONNECTIONS="${DOWNLOAD_CONNECTIONS:-16}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-10}"

cmd=(
  python3 "${SCRIPT_DIR}/download_dataset_shards.py"
  --config "${CONFIG}"
  --cache "${CACHE}"
  --jobs "${DOWNLOAD_JOBS}"
  --downloader "${DOWNLOAD_BACKEND}"
  --connections "${DOWNLOAD_CONNECTIONS}"
  --retries "${DOWNLOAD_RETRIES}"
)
if [[ -n "${SEED:-}" ]]; then
  cmd+=(--seed "${SEED}")
fi

echo
printf '[run]'
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
