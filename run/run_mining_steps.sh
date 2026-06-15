#!/usr/bin/env bash
set -euo pipefail

random_seed_gt_100() {
  echo $((101 + RANDOM + (RANDOM << 15)))
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SEED="${SEED:-}"
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

KING_SOURCE="${KING_SOURCE:-dashboard}" # dashboard | hippius | hf
STEP2_IMPL="${STEP2_IMPL:-weighted}"     # weighted | legacy
DEVICE="${DEVICE:-cuda:0}"
STEP2_DEVICE="${STEP2_DEVICE:-auto}"
SEED="${SEED:-$(random_seed_gt_100)}"
STEP4_SEED="${STEP4_SEED:-${SEED}}"

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
SHARD_START="${SHARD_START:-0}"
RANDOM_SHARDS="${RANDOM_SHARDS:-1}"
CACHE_ONLY="${CACHE_ONLY:-0}"
DATASET_CACHE="${DATASET_CACHE:-/workspace/teutonic-mining/cache/datasets}"
STEP2_PER_DEVICE_BATCH_SIZE="${STEP2_PER_DEVICE_BATCH_SIZE:-8}"
N_SHARDS_PER_DATASET="${N_SHARDS_PER_DATASET:-10}"
N_SHARDS="${N_SHARDS:-2}"
EVAL_SHARD="${EVAL_SHARD:-10}"
N_SCORE="${N_SCORE:-120000}"
TRAIN_PER_ITER="${TRAIN_PER_ITER:-15000}"
VAL_SIZE="${VAL_SIZE:-600}"

N_GPUS="${N_GPUS:-1}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
LORA_INIT="${LORA_INIT:-${INIT_LORA_WEIGHTS:-true}}"
LORA_ADAPTER_TYPE="${LORA_ADAPTER_TYPE:-lora}"
LORA_USE_ADALORA="${LORA_USE_ADALORA:-${USE_ADALORA:-0}}"
LORA_USE_VERA="${LORA_USE_VERA:-${USE_VERA:-0}}"
LORA_USE_LOHA="${LORA_USE_LOHA:-${USE_LOHA:-0}}"
LORA_USE_LOKR="${LORA_USE_LOKR:-${LORA_USE_KOKR:-${USE_LOKR:-${USE_KOKR:-0}}}}"
LORA_USE_QLORA="${LORA_USE_QLORA:-${USE_QLORA:-0}}"
QLORA_QUANT_TYPE="${QLORA_QUANT_TYPE:-nf4}"
LORA_USE_UNSLOTH="${LORA_USE_UNSLOTH:-${USE_UNSLOTH:-0}}"
UNSLOTH_GRADIENT_CHECKPOINTING="${UNSLOTH_GRADIENT_CHECKPOINTING:-unsloth}"
UNSLOTH_RANDOM_STATE="${UNSLOTH_RANDOM_STATE:-${STEP4_SEED}}"
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
LORA_USE_DORA="${LORA_USE_DORA:-${USE_DORA:-0}}"
LORA_USE_RSLORA="${LORA_USE_RSLORA:-${USE_RSLORA:-0}}"
LORA_USE_LORAPLUS="${LORA_USE_LORAPLUS:-${USE_LORAPLUS:-0}}"
LORAPLUS_LR_RATIO="${LORAPLUS_LR_RATIO:-16}"
EVAL_STEPS="${EVAL_STEPS:-150}"
SAVE_STEPS="${SAVE_STEPS:-${EVAL_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
TRAIN_DATA="${TRAIN_DATA:-}"
VAL_DATA="${VAL_DATA:-}"
METADATA_OUT="${METADATA_OUT:-}"

MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-4.3GB}"
N_EVAL="${N_EVAL:-2000}"
EVAL_N_SHARDS_PER_DATASET="${EVAL_N_SHARDS_PER_DATASET:-1}"
EVAL_SHARD_START="${EVAL_SHARD_START:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
TRAIN_SUMMARY="${TRAIN_SUMMARY:-${WORK_DIR}/score_summary.json}"

KING_DIR="${KING_DIR:-${WORK_DIR}/king}"
LORA_OUT_DIR="${LORA_OUT_DIR:-${WORK_DIR}/lora_out}"
MERGED_DIR="${MERGED_DIR:-${WORK_DIR}/merged}"

usage() {
  cat <<'EOF'
Usage:
  run/run_mining_steps.sh [all|step1|step1_1|step1_2|step2|step2_legacy|step3|step4|step5|step6 ...]

Common environment variables:
  TEUTONIC_WORK=/workspace/teutonic-mining/work-stepwise
  TEUTONIC_VENV=/workspace/teutonic/.venv
  KING_SOURCE=dashboard|hippius|hf
  STEP2_IMPL=weighted|legacy
  N_SCORE=120000 TRAIN_PER_ITER=15000 VAL_SIZE=600 N_EVAL=2000
  N_GPUS=1 MICRO_BATCH=4 GRAD_ACCUM=4 LR=1e-5 EPOCHS=2.0 WARMUP_RATIO=0.03 WEIGHT_DECAY=0.01 LORA_DROPOUT=0.05 EVAL_STEPS=150
  LORA_INIT=true LORA_USE_DORA=0 LORA_USE_RSLORA=0 LORA_USE_LORAPLUS=0 LORAPLUS_LR_RATIO=16
  LORA_ADAPTER_TYPE=lora LORA_USE_QLORA=0 LORA_USE_UNSLOTH=0 LORA_USE_PISSA=0 LORA_USE_LOFTQ=0 LORA_USE_EVA=0
  DEVICE=cuda:0 SEED=123

Examples:
  run/setup_finetune_env.sh
  run/run_mining_steps.sh all
  N_SCORE=20000 TRAIN_PER_ITER=16000 EPOCHS=1 run/run_mining_steps.sh step2 step3 step4
  run/step2_score_samples.sh
  KING_SOURCE=hf KING_MODEL_URL=https://huggingface.co/org/model run/run_mining_steps.sh step1 step2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${ROOT_DIR}"
mkdir -p "${WORK_DIR}"

if [[ -d "${VENV_DIR}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
elif [[ "${AUTO_SETUP:-0}" == "1" ]]; then
  "${ROOT_DIR}/run/setup_finetune_env.sh"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "Virtualenv not found at ${VENV_DIR}." >&2
  echo "Run: ${ROOT_DIR}/run/setup_finetune_env.sh" >&2
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
  if [[ "${RANDOM_SHARDS}" == "1" || "${RANDOM_SHARDS}" == "true" ]]; then
    arr_ref+=(--random-shards)
  fi
  return 0
}

append_step2_shard_selection_arg() {
  local -n arr_ref=$1
  if [[ "${SEQUENTIAL_SHARDS:-0}" == "1" || "${SEQUENTIAL_SHARDS:-0}" == "true" ]]; then
    arr_ref+=(--sequential-shards)
  else
    arr_ref+=(--random-shards)
  fi
  return 0
}

step1_dashboard() {
  local cmd=(
    "${PYTHON_BIN}" scripts/mining/step1_download_king.py
    --work "${WORK_DIR}"
    --king-dir "${KING_DIR}"
    --download-workers "${DOWNLOAD_WORKERS}"
  )
  run_cmd "${cmd[@]}"
}

step1_hippius() {
  local cmd=(
    "${PYTHON_BIN}" scripts/mining/step1_1_download_king.py
    --work "${WORK_DIR}"
    --king-dir "${KING_DIR}"
    --download-workers "${DOWNLOAD_WORKERS}"
  )
  append_optional_model_args cmd
  run_cmd "${cmd[@]}"
}

step1_hf() {
  local cmd=(
    "${PYTHON_BIN}" scripts/mining/step1_2_download_king_hf.py
    --work "${WORK_DIR}"
    --king-dir "${KING_DIR}"
    --download-workers "${DOWNLOAD_WORKERS}"
  )
  append_optional_model_args cmd
  run_cmd "${cmd[@]}"
}

step1() {
  case "${KING_SOURCE}" in
    dashboard) step1_dashboard ;;
    hippius) step1_hippius ;;
    hf) step1_hf ;;
    *) echo "Unknown KING_SOURCE=${KING_SOURCE}; expected dashboard, hippius, or hf" >&2; exit 2 ;;
  esac
}

step2_weighted() {
  local cmd=(
    "${PYTHON_BIN}" scripts/mining/step2_score_samples.py
    --work "${WORK_DIR}"
    --king-dir "${KING_DIR}"
    --n-shards-per-dataset "${N_SHARDS_PER_DATASET}"
    --shard-start "${SHARD_START}"
    --n-score "${N_SCORE}"
    --device "${STEP2_DEVICE}"
    --download-workers "${DOWNLOAD_WORKERS}"
    --dataset-cache "${DATASET_CACHE}"
    --per-device-batch-size "${STEP2_PER_DEVICE_BATCH_SIZE}"
  )
  append_step2_shard_selection_arg cmd
  if [[ "${CACHE_ONLY}" == "1" || "${CACHE_ONLY}" == "true" ]]; then
    cmd+=(--cache-only)
  fi
  if [[ -n "${USER_SEED}" ]]; then
    cmd+=(--seed "${USER_SEED}")
  fi
  append_optional_model_args cmd
  if [[ -n "${DATASETS_CONFIG:-}" ]]; then
    cmd+=(--datasets-config "${DATASETS_CONFIG}")
  fi
  run_cmd "${cmd[@]}"
}

step2_legacy() {
  local cmd=(
    "${PYTHON_BIN}" scripts/mining/step2_score_samples1.py
    --work "${WORK_DIR}"
    --king-dir "${KING_DIR}"
    --n-shards "${N_SHARDS}"
    --shard-start "${SHARD_START}"
    --eval-shard "${EVAL_SHARD}"
    --n-score "${N_SCORE}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --download-workers "${DOWNLOAD_WORKERS}"
  )
  append_optional_model_args cmd
  if [[ -n "${DATASET_CONFIG:-}" ]]; then
    cmd+=(--dataset-config "${DATASET_CONFIG}")
  fi
  run_cmd "${cmd[@]}"
}

step2() {
  case "${STEP2_IMPL}" in
    weighted) step2_weighted ;;
    legacy) step2_legacy ;;
    *) echo "Unknown STEP2_IMPL=${STEP2_IMPL}; expected weighted or legacy" >&2; exit 2 ;;
  esac
}

step3() {
  run_cmd "${PYTHON_BIN}" scripts/mining/step3_build_curriculum.py \
    --work "${WORK_DIR}" \
    --train-per-iter "${TRAIN_PER_ITER}" \
    --val-size "${VAL_SIZE}" \
    --seed "${SEED}"
}

step4() {
  local cmd=(
    "${PYTHON_BIN}" scripts/mining/step4_train_lora.py
    --work "${WORK_DIR}"
    --king-dir "${KING_DIR}"
    --bundle "${BUNDLE_DIR}"
    --output-dir "${LORA_OUT_DIR}"
    --n-gpus "${N_GPUS}"
    --micro-batch "${MICRO_BATCH}"
    --grad-accum "${GRAD_ACCUM}"
    --lr "${LR}"
    --epochs "${EPOCHS}"
    --warmup-ratio "${WARMUP_RATIO}"
    --weight-decay "${WEIGHT_DECAY}"
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --lora-init "${LORA_INIT}"
    --adapter-type "${LORA_ADAPTER_TYPE}"
    --qlora-quant-type "${QLORA_QUANT_TYPE}"
    --unsloth-gradient-checkpointing "${UNSLOTH_GRADIENT_CHECKPOINTING}"
    --unsloth-random-state "${UNSLOTH_RANDOM_STATE}"
    --eva-rho "${EVA_RHO}"
    --eva-tau "${EVA_TAU}"
    --loftq-bits "${LOFTQ_BITS}"
    --loftq-iter "${LOFTQ_ITER}"
    --lora-ga-direction "${LORA_GA_DIRECTION}"
    --lora-ga-scale "${LORA_GA_SCALE}"
    --lora-ga-stable-gamma "${LORA_GA_STABLE_GAMMA}"
    --adalora-target-r "${ADALORA_TARGET_R}"
    --adalora-init-r "${ADALORA_INIT_R}"
    --adalora-tinit "${ADALORA_TINIT}"
    --adalora-tfinal "${ADALORA_TFINAL}"
    --adalora-delta-t "${ADALORA_DELTA_T}"
    --adalora-beta1 "${ADALORA_BETA1}"
    --adalora-beta2 "${ADALORA_BETA2}"
    --adalora-orth-reg-weight "${ADALORA_ORTH_REG_WEIGHT}"
    --vera-projection-prng-key "${VERA_PROJECTION_PRNG_KEY}"
    --vera-d-initial "${VERA_D_INITIAL}"
    --loha-module-dropout "${LOHA_MODULE_DROPOUT}"
    --lokr-module-dropout "${LOKR_MODULE_DROPOUT}"
    --lokr-decompose-factor "${LOKR_DECOMPOSE_FACTOR}"
    --eval-steps "${EVAL_STEPS}"
    --save-steps "${SAVE_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --seed "${STEP4_SEED}"
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
}

step5() {
  run_cmd "${PYTHON_BIN}" scripts/mining/step5_merge_lora.py \
    --work "${WORK_DIR}" \
    --king-dir "${KING_DIR}" \
    --merged-dir "${MERGED_DIR}" \
    --max-shard-size "${MAX_SHARD_SIZE}"
}

step6() {
  local cmd=(
    "${PYTHON_BIN}" scripts/mining/step6_eval_verdict.py
    --work "${WORK_DIR}"
    --king-dir "${KING_DIR}"
    --merged-dir "${MERGED_DIR}"
    --n-shards-per-dataset "${EVAL_N_SHARDS_PER_DATASET}"
    --shard-start "${EVAL_SHARD_START}"
    --n-eval "${N_EVAL}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --n-bootstrap "${N_BOOTSTRAP}"
    --train-summary "${TRAIN_SUMMARY}"
  )
  append_random_shards_arg cmd
  if [[ -n "${DATASETS_CONFIG:-}" ]]; then
    cmd+=(--datasets-config "${DATASETS_CONFIG}")
  fi
  run_cmd "${cmd[@]}"
}

steps=("$@")
if [[ ${#steps[@]} -eq 0 ]]; then
  steps=(all)
fi

for step in "${steps[@]}"; do
  case "${step}" in
    all)
      step1
      step2
      step3
      step4
      step5
      step6
      ;;
    step1) step1 ;;
    step1_1|hippius) step1_hippius ;;
    step1_2|hf) step1_hf ;;
    step2) step2 ;;
    step2_legacy|legacy) step2_legacy ;;
    step3) step3 ;;
    step4) step4 ;;
    step5) step5 ;;
    step6) step6 ;;
    *)
      echo "Unknown step: ${step}" >&2
      usage >&2
      exit 2
      ;;
  esac
done
