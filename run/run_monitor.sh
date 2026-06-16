#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${TEUTONIC_VENV:-${ROOT_DIR}/.venv}"
HOST="${MONITOR_HOST:-0.0.0.0}"
PORT="${MONITOR_PORT:-17888}"

cd "${ROOT_DIR}"
if [[ -d "${VENV_DIR}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
exec python -m uvicorn monitor.app:app --host "${HOST}" --port "${PORT} "
