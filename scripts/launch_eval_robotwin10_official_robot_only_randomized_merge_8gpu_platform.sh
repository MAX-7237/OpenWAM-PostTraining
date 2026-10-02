#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-official-robot-only-robotwin10-randomized-20261002-$(date -u +%H%M%S)}"
export OPENWAM_EVAL_MODE=randomized ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-100}" RUN_ID
if [[ "${OPENWAM_EVAL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exec bash "$ROOT/scripts/launch_eval_robotwin10_official_8gpu_platform_common.sh" \
    openwam-official-robot-only-randomized \
    /media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/official-robot-only/sft-robotwin10-official-robot-only-8g-bs16-30k-20260927-retry/2026-09-28_00-02-42 \
    eval_robotwin10_sft_official_robot_only_8gpu.sh
fi
bash "$ROOT/scripts/launch_eval_robotwin10_official_8gpu_platform_common.sh" \
  openwam-official-robot-only-randomized \
  /media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/official-robot-only/sft-robotwin10-official-robot-only-8g-bs16-30k-20260927-retry/2026-09-28_00-02-42 \
  eval_robotwin10_sft_official_robot_only_8gpu.sh
"${OPENWAM_PYTHON:-/mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python}" "$ROOT/scripts/merge_robotwin10_clean_randomized.py" \
  --clean "$ROOT/logs/robotwin10_official/openwam-official-robot-only-robotwin10-seed0-100ep-20260930-120139/demo_clean" \
  --randomized "$ROOT/logs/robotwin10_official/$RUN_ID/demo_randomized" \
  --output "$ROOT/logs/robotwin10_official/$RUN_ID/combined"
