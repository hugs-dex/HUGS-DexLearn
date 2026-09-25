#!/usr/bin/env bash
set -euo pipefail

# Launch the final two-stage Human Prior training flow.
#
# Stage 1 trains the shared point-cloud encoder together with the diffusion
# pose generator, while a weak type-prior loss keeps the shared encoder useful
# for Human Prior scoring. Stage 2 loads the full Stage 1 checkpoint, freezes
# the encoder and diffusion modules, and continues training the type predictor
# on record-uniform soft-label supervision.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-output/two_stage_train_logs}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-prior_stage1}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-prior_stage2}"
STAGE1_GPU="${STAGE1_GPU:-0}"
STAGE2_GPU="${STAGE2_GPU:-${STAGE1_GPU}}"
STAGE1_CKPT_STEP="${STAGE1_CKPT_STEP:-010000}"
STAGE1_LOSS_TYPE="${STAGE1_LOSS_TYPE:-0.005}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${HUGS_OUTPUT_ROOT:-output}"

if [[ "${STAGE1_LOSS_TYPE}" == "0" || "${STAGE1_LOSS_TYPE}" == "0.0" || "${STAGE1_LOSS_TYPE}" == "0.00" || "${STAGE1_LOSS_TYPE}" == "0.000" ]]; then
  STAGE1_FREEZE_TYPE_CLASSIFIER=true
  STAGE2_STRICT_MODEL=false
  STAGE2_IGNORE_PREFIXES="[type_classifier]"
  STAGE2_TYPE_HEAD_MODE=reset
else
  STAGE1_FREEZE_TYPE_CLASSIFIER=false
  STAGE2_STRICT_MODEL=true
  STAGE2_IGNORE_PREFIXES="[]"
  STAGE2_TYPE_HEAD_MODE=continued
fi

COMMON_OVERRIDES=(
  task=train
  data=humanMulti
  algo=humanMultiHierar
  test_data=DGNMulti
  device=cuda:0
  algo.two_stage.enabled=false
  algo.model.type_objective=ce
  algo.supervision.balancing.enabled=false
  data.sampling.train_unit=record_uniform
  data.sampling.pose_group_soft_labels=true
  data.augmentation.point_dropout.enabled=true
  data.augmentation.point_dropout.ratio=0.1
)

STAGE1_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
  exp_name="${STAGE1_EXP_NAME}"
  algo.model.train_type_only=false
  algo.loss_weight.loss_diffusion=1.0
  algo.loss_weight.loss_type="${STAGE1_LOSS_TYPE}"
  algo.freeze.type_classifier="${STAGE1_FREEZE_TYPE_CLASSIFIER}"
  algo.max_iter=10000
  algo.save_every=2500
  algo.val_every=2500
  model_registry.key_features="stage1_diffusion_encoder_10000iter_save2500_loss_type${STAGE1_LOSS_TYPE}_record_uniform_soft_labels"
)

STAGE1_CKPT_PATH="${OUTPUT_ROOT}/humanMulti_humanMultiHierar_${STAGE1_EXP_NAME}/ckpts/step_${STAGE1_CKPT_STEP}.pth"

STAGE2_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
  exp_name="${STAGE2_EXP_NAME}"
  ckpt="${STAGE1_CKPT_PATH}"
  algo.model.train_type_only=true
  algo.loss_weight.loss_diffusion=0.0
  algo.loss_weight.loss_type=1.0
  algo.ckpt_load.load_optimizer=false
  algo.ckpt_load.reset_iter=true
  algo.ckpt_load.strict_model="${STAGE2_STRICT_MODEL}"
  algo.ckpt_load.ignore_prefixes="${STAGE2_IGNORE_PREFIXES}"
  algo.freeze.backbone=true
  algo.freeze.grasp_type_emb=true
  algo.freeze.output_head=true
  algo.max_iter="\${algo.two_stage.stage2.max_iter}"
  algo.save_every="\${algo.two_stage.stage2.save_every}"
  algo.val_every="\${algo.two_stage.stage2.val_every}"
  algo.lr="\${algo.two_stage.stage2.lr}"
  algo.lr_min="\${algo.two_stage.stage2.lr_min}"
  model_registry.key_features="stage2_frozen_stage1_encoder_train_yaml_configured_${STAGE2_TYPE_HEAD_MODE}_type_head_record_uniform_soft_labels"
)

run_training() {
  # Args:
  #   $1: Human-readable stage label.
  #   $2: Physical GPU id assigned through CUDA_VISIBLE_DEVICES.
  #   $3: Log file path.
  #   $@: Remaining Hydra overrides for dexlearn/main.py.
  # Return:
  #   Exits with the training command status.
  local stage_name="$1"
  local gpu_id="$2"
  local log_file="$3"
  shift 3

  echo "Launching ${stage_name} on GPU ${gpu_id}; log: ${log_file}"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    printf "  DRY_RUN CUDA_VISIBLE_DEVICES=%q" "${gpu_id}"
    printf " %q" "${PYTHON_BIN}" dexlearn/main.py "$@"
    printf "\n"
    return 0
  fi

  mkdir -p "${LOG_DIR}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" dexlearn/main.py "$@" > "${log_file}" 2>&1
}

case "${STAGE}" in
  all)
    run_training "stage1 ${STAGE1_EXP_NAME}" "${STAGE1_GPU}" "${LOG_DIR}/${STAGE1_EXP_NAME}_stage1.log" "${STAGE1_OVERRIDES[@]}"
    run_training "stage2 ${STAGE2_EXP_NAME}" "${STAGE2_GPU}" "${LOG_DIR}/${STAGE2_EXP_NAME}_stage2.log" "${STAGE2_OVERRIDES[@]}"
    ;;
  stage1)
    run_training "stage1 ${STAGE1_EXP_NAME}" "${STAGE1_GPU}" "${LOG_DIR}/${STAGE1_EXP_NAME}_stage1.log" "${STAGE1_OVERRIDES[@]}"
    ;;
  stage2)
    if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && ! -f "${STAGE1_CKPT_PATH}" ]]; then
      echo "Missing Stage 1 checkpoint: ${STAGE1_CKPT_PATH}" >&2
      exit 1
    fi
    run_training "stage2 ${STAGE2_EXP_NAME}" "${STAGE2_GPU}" "${LOG_DIR}/${STAGE2_EXP_NAME}_stage2.log" "${STAGE2_OVERRIDES[@]}"
    ;;
  *)
    echo "Unsupported STAGE=${STAGE}; expected all, stage1, or stage2." >&2
    exit 1
    ;;
esac

echo "Requested two-stage training finished successfully."
