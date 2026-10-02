#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Final stage-2 checkpoint from the 350h ego -> 250h robot pretraining run.
export OPENWAM_SFT_INIT="${OPENWAM_SFT_INIT:-/media/damoxing/ckp/openwam/figure10_ego2robot_600h_8n8g_bs32_1epoch/ego2robot-600h-resume14000-20260929-stage2-robot250h/2026-09-30_12-23-01}"
# The saved stage-2 config retains this provenance interpolation. Bind it so
# any full-config resolution remains self-consistent during SFT warm-start.
export OPENWAM_EGO_STAGE_CKPT="${OPENWAM_EGO_STAGE_CKPT:-/media/damoxing/ckp/openwam/figure10_ego2robot_600h_8n8g_bs32_1epoch/ego2robot-600h-resume14000-20260929-stage1-ego350h/2026-09-30_03-27-07}"
export OPENWAM_SFT_MASK="${OPENWAM_SFT_MASK:-action_sees_video}"
export RUN_ID="${RUN_ID:-sft-robotwin10-pku-ego2robot-600h-8g-bs16-30k-20261002}"
export CKPT_ROOT="${CKPT_ROOT:-/media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k}"
export LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/sft_robotwin10/${RUN_ID}}"

if [[ ! -d "$OPENWAM_SFT_INIT" ]]; then
  echo "Missing pretraining checkpoint directory: $OPENWAM_SFT_INIT" >&2
  exit 3
fi
if [[ ! -f "$OPENWAM_SFT_INIT/config.yaml" ]]; then
  echo "Missing checkpoint config: $OPENWAM_SFT_INIT/config.yaml" >&2
  exit 3
fi
if [[ ! -d "$OPENWAM_SFT_INIT/tokenizer" ]]; then
  echo "Missing checkpoint tokenizer: $OPENWAM_SFT_INIT/tokenizer" >&2
  exit 3
fi
if [[ ! -d "$OPENWAM_EGO_STAGE_CKPT" ]]; then
  echo "Missing stage-1 provenance checkpoint: $OPENWAM_EGO_STAGE_CKPT" >&2
  exit 3
fi
if ! compgen -G "$OPENWAM_SFT_INIT/checkpoint_step_*.safetensors" >/dev/null; then
  echo "Missing checkpoint weights in: $OPENWAM_SFT_INIT" >&2
  exit 3
fi

echo "SFT_INIT=$OPENWAM_SFT_INIT"
echo "EGO_STAGE_CKPT=$OPENWAM_EGO_STAGE_CKPT"
echo "SFT_MASK=$OPENWAM_SFT_MASK"
echo "SFT_RUN_ID=$RUN_ID"
echo "SFT_OUTPUT=${CKPT_ROOT}/pku-ego2robot-600h/${RUN_ID}"

exec bash scripts/launch_sft_robotwin10_8g.sh pku-ego2robot-600h
