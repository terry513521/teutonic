#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

N_GPUS="${N_GPUS:-1}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1.5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
LORA_USE_DORA="${LORA_USE_DORA:-0}"
LORA_USE_RSLORA="${LORA_USE_RSLORA:-0}"
EVAL_STEPS="${EVAL_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-${EVAL_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

cmd=(
  "${PYTHON_BIN}" scripts/mining/step4_train_lora.py
  --work "${WORK_DIR}" \
  --king-dir "${KING_DIR}" \
  --bundle "${BUNDLE_DIR}" \
  --output-dir "${LORA_OUT_DIR}" \
  --n-gpus "${N_GPUS}" \
  --micro-batch "${MICRO_BATCH}" \
  --grad-accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --eval-steps "${EVAL_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
)

if [[ -n "${LORA_TARGET_MODULES}" ]]; then
  cmd+=(--lora-target-modules "${LORA_TARGET_MODULES}")
fi

if [[ "${LORA_USE_DORA}" == "1" || "${LORA_USE_DORA}" == "true" ]]; then
  cmd+=(--use-dora)
fi

if [[ "${LORA_USE_RSLORA}" == "1" || "${LORA_USE_RSLORA}" == "true" ]]; then
  cmd+=(--use-rslora)
fi

run_cmd "${cmd[@]}"
