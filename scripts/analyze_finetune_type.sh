#!/usr/bin/env sh
set -eu

usage() {
  cat <<'EOF'
Usage:
  scripts/analyze_finetune_type.sh BASE_MODEL_DIR TUNED_MODEL_DIR [extra analyzer args]

Example:
  scripts/analyze_finetune_type.sh /models/original /models/finetuned
  scripts/analyze_finetune_type.sh /models/original /models/finetuned --atol 1e-5 --top 50

This wraps scripts/analyze_finetune_type.py, which compares safetensors weights
and heuristically reports likely merged LoRA vs full fine-tune.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 2 ]; then
  usage >&2
  exit 2
fi

BASE_MODEL_DIR=$1
TUNED_MODEL_DIR=$2
shift 2

if [ ! -d "$BASE_MODEL_DIR" ]; then
  echo "error: base model directory does not exist: $BASE_MODEL_DIR" >&2
  exit 2
fi

if [ ! -d "$TUNED_MODEL_DIR" ]; then
  echo "error: tuned model directory does not exist: $TUNED_MODEL_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python3}

exec "$PYTHON_BIN" "$SCRIPT_DIR/analyze_finetune_type.py" \
  --base "$BASE_MODEL_DIR" \
  --tuned "$TUNED_MODEL_DIR" \
  "$@"
