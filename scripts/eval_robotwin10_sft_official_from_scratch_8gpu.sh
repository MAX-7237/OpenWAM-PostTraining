#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${OPENWAM_CKPT_DIR:-/media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/from-scratch/sft-robotwin10-from-scratch-8g-bs16-30k-20260927-retry/2026-09-27_23-04-03}"
exec bash "$ROOT/scripts/eval_robotwin10_sft_official_8gpu_common.sh" from-scratch \
  "$CKPT_DIR" \
  openwam-official-from-scratch
