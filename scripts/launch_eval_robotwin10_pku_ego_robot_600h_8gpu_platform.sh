#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "$ROOT/scripts/launch_eval_robotwin10_official_8gpu_platform_common.sh" \
  pku-ego-plus-robot-600h \
  /media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/pku-ego-robot-600h/sft-robotwin10-pku-ego-robot-600h-8g-bs16-30k-20260930/2026-09-30_19-31-57 \
  eval_robotwin10_sft_official_ego_robot_mutual_8gpu.sh
