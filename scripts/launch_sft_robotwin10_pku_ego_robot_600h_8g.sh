#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Final OpenWAM pretraining checkpoint from the 600h ego+robot run.
export OPENWAM_SFT_INIT="${OPENWAM_SFT_INIT:-/media/damoxing/ckp/openwam/figure10_ego_robot_600h_8n8g_bs32_1epoch/ego-robot-600h-warmstart28000-20260929/2026-09-30_07-18-06}"
export OPENWAM_SFT_MASK="${OPENWAM_SFT_MASK:-mutual}"
export RUN_ID="${RUN_ID:-sft-robotwin10-pku-ego-robot-600h-8g-bs16-30k-20260930}"
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
if ! compgen -G "$OPENWAM_SFT_INIT/checkpoint_step_*.safetensors" >/dev/null; then
  echo "Missing checkpoint weights in: $OPENWAM_SFT_INIT" >&2
  exit 3
fi

echo "SFT_INIT=$OPENWAM_SFT_INIT"
echo "SFT_MASK=$OPENWAM_SFT_MASK"
echo "SFT_RUN_ID=$RUN_ID"
echo "SFT_OUTPUT=${CKPT_ROOT}/pku-ego-robot-600h/${RUN_ID}"

exec bash scripts/launch_sft_robotwin10_8g.sh pku-ego-robot-600h
