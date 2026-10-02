#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# One distributed-platform command runs both Figure-10 stages sequentially.
# RUN_ID identifies the complete pipeline; stage outputs remain separate.
PIPELINE_RUN_ID="${RUN_ID:-ego2robot-600h-8n8g-bs32-log20-20260927}"
EGO_RUN_ID="${EGO_RUN_ID:-${PIPELINE_RUN_ID}-stage1-ego350h}"
ROBOT_RUN_ID="${ROBOT_RUN_ID:-${PIPELINE_RUN_ID}-stage2-robot250h}"
CKPT_ROOT="${CKPT_ROOT:-/media/damoxing/ckp/openwam/figure10_ego2robot_600h_8n8g_bs32_1epoch}"
LOG_ROOT="${OPENWAM_LOG_ROOT:-${ROOT}/logs}"

EGO_RUN_ROOT="${CKPT_ROOT}/${EGO_RUN_ID}"
ROBOT_RUN_ROOT="${CKPT_ROOT}/${ROBOT_RUN_ID}"

echo "EGO2ROBOT_PIPELINE=$PIPELINE_RUN_ID"
echo "STAGE1_OUTPUT_ROOT=$EGO_RUN_ROOT"
echo "STAGE2_OUTPUT_ROOT=$ROBOT_RUN_ROOT"

echo "========== Stage 1/2: ego 350h =========="
env \
  RUN_ID="$EGO_RUN_ID" \
  CKPT_ROOT="$CKPT_ROOT" \
  OPENWAM_LOG_ROOT="$LOG_ROOT" \
  OPENWAM_PLATFORM_LOG_DIR="${LOG_ROOT}/platform/${EGO_RUN_ID}" \
  WANDB_DIR="${LOG_ROOT}/wandb/${EGO_RUN_ID}" \
  WANDB_CACHE_DIR="${LOG_ROOT}/wandb-cache/${EGO_RUN_ID}" \
  WANDB_CONFIG_DIR="${LOG_ROOT}/wandb-config/${EGO_RUN_ID}" \
  WANDB_DATA_DIR="${LOG_ROOT}/wandb-data/${EGO_RUN_ID}" \
  XDG_CACHE_HOME="${LOG_ROOT}/xdg-cache/${EGO_RUN_ID}" \
  OPENWAM_WARM_START_CKPT="${OPENWAM_WARM_START_CKPT:-}" \
  OPENWAM_MAX_STEPS="${OPENWAM_MAX_STEPS:-}" \
  bash scripts/launch_figure10_ego2robot_600h_bs32_8n8g.sh ego

if [[ "${OPENWAM_DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN: stage 1 command validated; stage 2 requires the produced stage-1 checkpoint."
  exit 0
fi

# torchrun returns only after every stage-1 rank has completed. The final-save
# barrier therefore guarantees that the shared checkpoint is visible here.
OPENWAM_EGO_STAGE_CKPT="$({
  find "$EGO_RUN_ROOT" -mindepth 2 -maxdepth 2 -type f \
    -name 'checkpoint_step_*.safetensors' -printf '%T@ %h\n'
} | sort -nr | awk 'NR == 1 {$1=""; sub(/^ /, ""); print; exit}')"

if [[ -z "$OPENWAM_EGO_STAGE_CKPT" || ! -d "$OPENWAM_EGO_STAGE_CKPT" ]]; then
  echo "Stage 1 completed but no checkpoint directory was found under $EGO_RUN_ROOT" >&2
  exit 4
fi
if [[ ! -f "$OPENWAM_EGO_STAGE_CKPT/config.yaml" ]]; then
  echo "Stage-1 checkpoint is incomplete: missing $OPENWAM_EGO_STAGE_CKPT/config.yaml" >&2
  exit 4
fi
if ! compgen -G "$OPENWAM_EGO_STAGE_CKPT/checkpoint_step_*.safetensors" >/dev/null; then
  echo "Stage-1 checkpoint is incomplete: missing checkpoint weights" >&2
  exit 4
fi
export OPENWAM_EGO_STAGE_CKPT

echo "STAGE1_CHECKPOINT=$OPENWAM_EGO_STAGE_CKPT"
echo "========== Stage 2/2: robot 250h =========="
env \
  RUN_ID="$ROBOT_RUN_ID" \
  CKPT_ROOT="$CKPT_ROOT" \
  OPENWAM_EGO_STAGE_CKPT="$OPENWAM_EGO_STAGE_CKPT" \
  OPENWAM_LOG_ROOT="$LOG_ROOT" \
  OPENWAM_PLATFORM_LOG_DIR="${LOG_ROOT}/platform/${ROBOT_RUN_ID}" \
  WANDB_DIR="${LOG_ROOT}/wandb/${ROBOT_RUN_ID}" \
  WANDB_CACHE_DIR="${LOG_ROOT}/wandb-cache/${ROBOT_RUN_ID}" \
  WANDB_CONFIG_DIR="${LOG_ROOT}/wandb-config/${ROBOT_RUN_ID}" \
  WANDB_DATA_DIR="${LOG_ROOT}/wandb-data/${ROBOT_RUN_ID}" \
  XDG_CACHE_HOME="${LOG_ROOT}/xdg-cache/${ROBOT_RUN_ID}" \
  OPENWAM_WARM_START_CKPT="" \
  OPENWAM_MAX_STEPS="" \
  bash scripts/launch_figure10_ego2robot_600h_bs32_8n8g.sh robot

echo "EGO2ROBOT_COMPLETE=$PIPELINE_RUN_ID"
echo "FINAL_OUTPUT_ROOT=$ROBOT_RUN_ROOT"
