#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/launch_eval_robotwin10_official_8gpu_platform_common.sh" \
  openwam-official-from-scratch \
  /media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/from-scratch/sft-robotwin10-from-scratch-8g-bs16-30k-20260927-retry/2026-09-27_23-04-03 \
  eval_robotwin10_sft_official_from_scratch_8gpu.sh
