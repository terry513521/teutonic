#!/usr/bin/env bash
set -euo pipefail

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
SEED="${SEED:-42}"

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
SHARD_START="${SHARD_START:-0}"
RANDOM_SHARDS="${RANDOM_SHARDS:-1}"
CACHE_ONLY="${CACHE_ONLY:-0}"
N_SHARDS_PER_DATASET="${N_SHARDS_PER_DATASET:-5}"
N_SHARDS="${N_SHARDS:-2}"
EVAL_SHARD="${EVAL_SHARD:-10}"
N_SCORE="${N_SCORE:-40000}"
TRAIN_PER_ITER="${TRAIN_PER_ITER:-15000}"
VAL_SIZE="${VAL_SIZE:-600}"

N_GPUS="${N_GPUS:-1}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
EVAL_STEPS="${EVAL_STEPS:-150}"
SAVE_STEPS="${SAVE_STEPS:-${EVAL_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

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
  N_SCORE=40000 TRAIN_PER_ITER=15000 VAL_SIZE=600 N_EVAL=2000
  N_GPUS=1 MICRO_BATCH=4 GRAD_ACCUM=4 LR=1e-5 EPOCHS=2.0 WARMUP_RATIO=0.03 EVAL_STEPS=150
  DEVICE=cuda:0 SEED=42

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
    --device "${DEVICE}"
    --download-workers "${DOWNLOAD_WORKERS}"
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
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --eval-steps "${EVAL_STEPS}"
    --save-steps "${SAVE_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
  )
  if [[ -n "${LORA_TARGET_MODULES}" ]]; then
    cmd+=(--lora-target-modules "${LORA_TARGET_MODULES}")
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
