#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

N_GPUS="${N_GPUS:-1}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
LORA_INIT="${LORA_INIT:-${INIT_LORA_WEIGHTS:-olora}}"
LORA_USE_DORA="${LORA_USE_DORA:-0}"
LORA_USE_RSLORA="${LORA_USE_RSLORA:-${USE_RSLORA:-1}}"
LORA_USE_LORAPLUS="${LORA_USE_LORAPLUS:-${USE_LORAPLUS:-1}}"
LORAPLUS_LR_RATIO="${LORAPLUS_LR_RATIO:-16}"
EVAL_STEPS="${EVAL_STEPS:-150}"
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
  --lora-init "${LORA_INIT}" \
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

if [[ "${LORA_USE_LORAPLUS}" == "1" || "${LORA_USE_LORAPLUS}" == "true" ]]; then
  cmd+=(--use-loraplus --loraplus-lr-ratio "${LORAPLUS_LR_RATIO}")
fi

run_cmd "${cmd[@]}"
