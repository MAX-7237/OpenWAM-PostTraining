#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OPENWAM_PYTHON="${OPENWAM_PYTHON:-python}"

case "$MODE" in
  robot-only)
    CONFIG=figure10_pku_robot_only
    ;;
  ego2robot-ego)
    CONFIG=figure10_pku_ego2robot_ego
    ;;
  ego2robot-robot)
    : "${OPENWAM_EGO_STAGE_CKPT:?set OPENWAM_EGO_STAGE_CKPT to the completed ego-stage output directory}"
    CONFIG=figure10_pku_ego2robot_robot
    ;;
  ego-robot)
    CONFIG=figure10_pku_ego_robot
    ;;
  *)
    echo "usage: $0 {robot-only|ego2robot-ego|ego2robot-robot|ego-robot} [Hydra overrides...]" >&2
    exit 2
    ;;
esac
shift

PREFLIGHT_ARGS=(--mode "$MODE")
if [[ "${OPENWAM_PREFLIGHT_SAMPLE:-0}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--sample)
fi
"$OPENWAM_PYTHON" scripts/preflight_figure10_pku.py "${PREFLIGHT_ARGS[@]}"

if [[ "${OPENWAM_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

exec "$OPENWAM_PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train.py --config-name "$CONFIG" "$@"
