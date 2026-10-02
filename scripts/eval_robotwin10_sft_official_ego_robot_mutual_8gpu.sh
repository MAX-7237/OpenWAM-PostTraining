#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${OPENWAM_CKPT_DIR:-/media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/official-ego-robot-mutual/sft-robotwin10-official-ego-robot-mutual-8g-bs16-30k-20260927-retry/2026-09-27_22-00-57}"
exec bash "$ROOT/scripts/eval_robotwin10_sft_official_8gpu_common.sh" ego-robot-mutual \
  "$CKPT_DIR" \
  openwam-official-ego-robot-mutual
