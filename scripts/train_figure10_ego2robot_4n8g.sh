#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"

case "$STAGE" in
  ego)
    MODE=ego2robot-ego
    ;;
  robot)
    : "${OPENWAM_EGO_STAGE_CKPT:?set OPENWAM_EGO_STAGE_CKPT to the completed ego-stage output directory}"
    MODE=ego2robot-robot
    ;;
  *)
    echo "usage: $0 {ego|robot} [Hydra overrides...]" >&2
    echo "run ego on all four nodes first; then run robot with OPENWAM_EGO_STAGE_CKPT set" >&2
    exit 2
    ;;
esac
shift

# Four nodes run each stage with NODE_RANK=0,1,2,3 respectively.
exec "$ROOT/scripts/train_figure10_pku_4n8g.sh" "$MODE" "$@"
