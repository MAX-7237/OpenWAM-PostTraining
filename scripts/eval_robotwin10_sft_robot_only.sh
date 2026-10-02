#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/eval_robotwin10_sft_common.sh" \
  robot-only \
  /media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k/official-robot-only/sft-robotwin10-official-robot-only-8g-bs16-30k-20260927-retry/2026-09-28_00-02-42 \
  openwam-official-robot-only
