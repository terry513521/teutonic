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
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_MINING_EXTRAS="${INSTALL_MINING_EXTRAS:-1}"

cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  echo "Activate the Vast base env first, or install uv before running this script." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] creating virtualenv: ${VENV_DIR}"
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
  echo "[setup] reusing virtualenv: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[setup] python: $(python --version)"
python - <<'PY'
import sys
print(f"[setup] executable: {sys.executable}")
PY

uv pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_TORCH}" != "0" ]]; then
  echo "[setup] installing PyTorch from ${TORCH_INDEX_URL}"
  uv pip install --index-url "${TORCH_INDEX_URL}" torch
fi

echo "[setup] installing teutonic package"
uv pip install -e .

if [[ "${INSTALL_MINING_EXTRAS}" != "0" ]]; then
  echo "[setup] installing mining/fine-tuning requirements"
  uv pip install -r scripts/mining/requirements.txt
  uv pip install nvidia-cuda-runtime-cu12
fi

python - <<'PY'
import importlib.util
import torch

mods = ["transformers", "peft", "accelerate", "huggingface_hub", "hippius_hub"]
missing = [name for name in mods if importlib.util.find_spec(name) is None]
print(f"[setup] torch: {torch.__version__}")
print(f"[setup] cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] gpu: {torch.cuda.get_device_name(0)}")
    print(f"[setup] capability: {torch.cuda.get_device_capability(0)}")
if missing:
    raise SystemExit(f"missing expected modules: {', '.join(missing)}")
PY

echo "[setup] done"
echo "Activate with: source ${VENV_DIR}/bin/activate"
