#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/launch_eval_robotwin10_official_8gpu_platform_common.sh" \
  openwam-official-ego2robot \
  /media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/official-ego2robot/sft-robotwin10-official-ego2robot-8g-bs16-30k-20260927/2026-09-28_01-18-27 \
  eval_robotwin10_sft_official_ego2robot_8gpu.sh
