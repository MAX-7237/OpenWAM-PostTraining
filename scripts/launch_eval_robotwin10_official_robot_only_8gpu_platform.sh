#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/launch_eval_robotwin10_official_8gpu_platform_common.sh" \
  openwam-official-robot-only \
  /media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/official-robot-only/sft-robotwin10-official-robot-only-8g-bs16-30k-20260927-retry/2026-09-28_00-02-42 \
  eval_robotwin10_sft_official_robot_only_8gpu.sh
