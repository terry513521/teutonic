#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

N_GPUS="${N_GPUS:-1}"
MICRO_BATCH="${MICRO_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate,up,down,w_down_proj,w_up_proj}"
LORA_INIT="${LORA_INIT:-${INIT_LORA_WEIGHTS:-olora}}"
LORA_ADAPTER_TYPE="${LORA_ADAPTER_TYPE:-lora}"
LORA_USE_ADALORA="${LORA_USE_ADALORA:-${USE_ADALORA:-0}}"
LORA_USE_VERA="${LORA_USE_VERA:-${USE_VERA:-0}}"
LORA_USE_LOHA="${LORA_USE_LOHA:-${USE_LOHA:-0}}"
LORA_USE_LOKR="${LORA_USE_LOKR:-${LORA_USE_KOKR:-${USE_LOKR:-${USE_KOKR:-0}}}}"
LORA_USE_QLORA="${LORA_USE_QLORA:-${USE_QLORA:-0}}"
QLORA_QUANT_TYPE="${QLORA_QUANT_TYPE:-nf4}"
LORA_USE_UNSLOTH="${LORA_USE_UNSLOTH:-${USE_UNSLOTH:-1}}"
UNSLOTH_GRADIENT_CHECKPOINTING="${UNSLOTH_GRADIENT_CHECKPOINTING:-unsloth}"
UNSLOTH_RANDOM_STATE="${UNSLOTH_RANDOM_STATE:-3407}"
LORA_USE_DELTA_LORA="${LORA_USE_DELTA_LORA:-${USE_DELTA_LORA:-0}}"
LORA_USE_PISSA="${LORA_USE_PISSA:-${USE_PISSA:-0}}"
LORA_USE_CORDA="${LORA_USE_CORDA:-${USE_CORDA:-0}}"
LORA_USE_EVA="${LORA_USE_EVA:-${USE_EVA:-0}}"
EVA_RHO="${EVA_RHO:-2.0}"
EVA_TAU="${EVA_TAU:-0.99}"
EVA_WHITEN="${EVA_WHITEN:-0}"
LORA_USE_LOFTQ="${LORA_USE_LOFTQ:-${USE_LOFTQ:-0}}"
LOFTQ_BITS="${LOFTQ_BITS:-4}"
LOFTQ_ITER="${LOFTQ_ITER:-1}"
LORA_USE_LORA_GA="${LORA_USE_LORA_GA:-${USE_LORA_GA:-0}}"
LORA_GA_DIRECTION="${LORA_GA_DIRECTION:-ArB2r}"
LORA_GA_SCALE="${LORA_GA_SCALE:-stable}"
LORA_GA_STABLE_GAMMA="${LORA_GA_STABLE_GAMMA:-16}"
ADALORA_TARGET_R="${ADALORA_TARGET_R:-8}"
ADALORA_INIT_R="${ADALORA_INIT_R:-12}"
ADALORA_TINIT="${ADALORA_TINIT:-0}"
ADALORA_TFINAL="${ADALORA_TFINAL:-0}"
ADALORA_DELTA_T="${ADALORA_DELTA_T:-1}"
ADALORA_BETA1="${ADALORA_BETA1:-0.85}"
ADALORA_BETA2="${ADALORA_BETA2:-0.85}"
ADALORA_ORTH_REG_WEIGHT="${ADALORA_ORTH_REG_WEIGHT:-0.5}"
VERA_PROJECTION_PRNG_KEY="${VERA_PROJECTION_PRNG_KEY:-0}"
VERA_SAVE_PROJECTION="${VERA_SAVE_PROJECTION:-1}"
VERA_D_INITIAL="${VERA_D_INITIAL:-0.1}"
LOHA_MODULE_DROPOUT="${LOHA_MODULE_DROPOUT:-0.0}"
LOKR_MODULE_DROPOUT="${LOKR_MODULE_DROPOUT:-0.0}"
LOKR_DECOMPOSE_BOTH="${LOKR_DECOMPOSE_BOTH:-0}"
LOKR_DECOMPOSE_FACTOR="${LOKR_DECOMPOSE_FACTOR:--1}"
LORA_USE_DORA="${LORA_USE_DORA:-1}"
LORA_USE_RSLORA="${LORA_USE_RSLORA:-${USE_RSLORA:-1}}"
LORA_USE_LORAPLUS="${LORA_USE_LORAPLUS:-${USE_LORAPLUS:-1}}"
LORAPLUS_LR_RATIO="${LORAPLUS_LR_RATIO:-16}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-${EVAL_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
TRAIN_DATA="${TRAIN_DATA:-${WORK_DIR}/curriculum/train.jsonl}"
VAL_DATA="${VAL_DATA:-${WORK_DIR}/curriculum/val.jsonl}"
METADATA_OUT="${METADATA_OUT:-${WORK_DIR}/adapter.json}"

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
  --weight-decay "${WEIGHT_DECAY}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --lora-init "${LORA_INIT}" \
  --adapter-type "${LORA_ADAPTER_TYPE}" \
  --qlora-quant-type "${QLORA_QUANT_TYPE}" \
  --unsloth-gradient-checkpointing "${UNSLOTH_GRADIENT_CHECKPOINTING}" \
  --unsloth-random-state "${UNSLOTH_RANDOM_STATE}" \
  --eva-rho "${EVA_RHO}" \
  --eva-tau "${EVA_TAU}" \
  --loftq-bits "${LOFTQ_BITS}" \
  --loftq-iter "${LOFTQ_ITER}" \
  --lora-ga-direction "${LORA_GA_DIRECTION}" \
  --lora-ga-scale "${LORA_GA_SCALE}" \
  --lora-ga-stable-gamma "${LORA_GA_STABLE_GAMMA}" \
  --adalora-target-r "${ADALORA_TARGET_R}" \
  --adalora-init-r "${ADALORA_INIT_R}" \
  --adalora-tinit "${ADALORA_TINIT}" \
  --adalora-tfinal "${ADALORA_TFINAL}" \
  --adalora-delta-t "${ADALORA_DELTA_T}" \
  --adalora-beta1 "${ADALORA_BETA1}" \
  --adalora-beta2 "${ADALORA_BETA2}" \
  --adalora-orth-reg-weight "${ADALORA_ORTH_REG_WEIGHT}" \
  --vera-projection-prng-key "${VERA_PROJECTION_PRNG_KEY}" \
  --vera-d-initial "${VERA_D_INITIAL}" \
  --loha-module-dropout "${LOHA_MODULE_DROPOUT}" \
  --lokr-module-dropout "${LOKR_MODULE_DROPOUT}" \
  --lokr-decompose-factor "${LOKR_DECOMPOSE_FACTOR}" \
  --eval-steps "${EVAL_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
)

if [[ -n "${LORA_TARGET_MODULES}" ]]; then
  cmd+=(--lora-target-modules "${LORA_TARGET_MODULES}")
fi

if [[ -n "${TRAIN_DATA}" ]]; then
  cmd+=(--train-data "${TRAIN_DATA}")
fi

if [[ -n "${VAL_DATA}" ]]; then
  cmd+=(--val-data "${VAL_DATA}")
fi

if [[ -n "${METADATA_OUT}" ]]; then
  cmd+=(--metadata-out "${METADATA_OUT}")
fi

if [[ "${LORA_USE_ADALORA}" == "1" || "${LORA_USE_ADALORA}" == "true" ]]; then
  cmd+=(--use-adalora)
fi

if [[ "${LORA_USE_VERA}" == "1" || "${LORA_USE_VERA}" == "true" ]]; then
  cmd+=(--use-vera)
fi

if [[ "${LORA_USE_LOHA}" == "1" || "${LORA_USE_LOHA}" == "true" ]]; then
  cmd+=(--use-loha)
fi

if [[ "${LORA_USE_LOKR}" == "1" || "${LORA_USE_LOKR}" == "true" ]]; then
  cmd+=(--use-lokr)
fi

if [[ "${LORA_USE_QLORA}" == "1" || "${LORA_USE_QLORA}" == "true" ]]; then
  cmd+=(--use-qlora)
fi

if [[ "${LORA_USE_UNSLOTH}" == "1" || "${LORA_USE_UNSLOTH}" == "true" ]]; then
  cmd+=(--use-unsloth)
fi

if [[ "${LORA_USE_DELTA_LORA}" == "1" || "${LORA_USE_DELTA_LORA}" == "true" ]]; then
  cmd+=(--use-delta-lora)
fi

if [[ "${LORA_USE_PISSA}" == "1" || "${LORA_USE_PISSA}" == "true" ]]; then
  cmd+=(--use-pissa)
fi

if [[ "${LORA_USE_CORDA}" == "1" || "${LORA_USE_CORDA}" == "true" ]]; then
  cmd+=(--use-corda)
fi

if [[ "${LORA_USE_EVA}" == "1" || "${LORA_USE_EVA}" == "true" ]]; then
  cmd+=(--use-eva)
fi

if [[ "${EVA_WHITEN}" == "1" || "${EVA_WHITEN}" == "true" ]]; then
  cmd+=(--eva-whiten)
fi

if [[ "${LORA_USE_LOFTQ}" == "1" || "${LORA_USE_LOFTQ}" == "true" ]]; then
  cmd+=(--use-loftq)
fi

if [[ "${LORA_USE_LORA_GA}" == "1" || "${LORA_USE_LORA_GA}" == "true" ]]; then
  cmd+=(--use-lora-ga)
fi

if [[ "${VERA_SAVE_PROJECTION}" == "0" || "${VERA_SAVE_PROJECTION}" == "false" ]]; then
  cmd+=(--no-vera-save-projection)
fi

if [[ "${LOKR_DECOMPOSE_BOTH}" == "1" || "${LOKR_DECOMPOSE_BOTH}" == "true" ]]; then
  cmd+=(--lokr-decompose-both)
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
