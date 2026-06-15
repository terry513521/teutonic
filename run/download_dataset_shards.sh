#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${DATASETS_CONFIG:-${SCRIPT_DIR}/datasets.json}"
CACHE="${DATASET_CACHE:-/workspace/teutonic-mining/cache/datasets}"

cmd=(python3 "${SCRIPT_DIR}/download_dataset_shards.py" --config "${CONFIG}" --cache "${CACHE}")
if [[ -n "${SEED:-}" ]]; then
  cmd+=(--seed "${SEED}")
fi

echo
printf '[run]'
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
