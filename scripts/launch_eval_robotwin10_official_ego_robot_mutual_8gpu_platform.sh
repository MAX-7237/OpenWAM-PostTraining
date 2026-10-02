#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/launch_eval_robotwin10_official_8gpu_platform_common.sh" \
  openwam-official-ego-plus-robot \
  /media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/official-ego-robot-mutual/sft-robotwin10-official-ego-robot-mutual-8g-bs16-30k-20260927-retry/2026-09-27_22-00-57 \
  eval_robotwin10_sft_official_ego_robot_mutual_8gpu.sh
