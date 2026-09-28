#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
PYTHON="${OPENWAM_PYTHON:-/mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python}"
CUDNN_LIB="${OPENWAM_CUDNN_LIB:-}"
NNODES="${NNODES:-${HOST_NUM:-8}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${RANK:-}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

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

if [[ "$NNODES" != 8 || "$NPROC_PER_NODE" != 8 ]]; then
  echo "this launcher requires NNODES=8 and NPROC_PER_NODE=8" >&2
  exit 2
fi
if [[ ! "$NODE_RANK" =~ ^[0-7]$ ]]; then
  echo "NODE_RANK must be one of 0, 1, 2, 3, 4, 5, 6, 7" >&2
  exit 2
fi
: "${MASTER_ADDR:?set MASTER_ADDR to the address of node rank 0}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python is not executable: $PYTHON" >&2
  exit 1
fi
if [[ -n "$CUDNN_LIB" ]]; then
  if [[ ! -f "$CUDNN_LIB/libcudnn_graph.so.9" ]]; then
    echo "missing private cuDNN runtime: $CUDNN_LIB" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$ROOT"

export DS_ACCELERATOR=cuda
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ "${OPENWAM_SKIP_PREFLIGHT:-0}" != 1 ]]; then
  PREFLIGHT_ARGS=(--mode "$MODE")
  PATHS_PER_SOURCE="${OPENWAM_PREFLIGHT_PATHS_PER_SOURCE:-3}"
  if [[ ! "$PATHS_PER_SOURCE" =~ ^[0-9]+$ ]]; then
    echo "OPENWAM_PREFLIGHT_PATHS_PER_SOURCE must be a non-negative integer" >&2
    exit 2
  fi
  if (( PATHS_PER_SOURCE > 0 )); then
    PREFLIGHT_ARGS+=(--check-paths-per-source "$PATHS_PER_SOURCE")
  fi
  if [[ "$NODE_RANK" == 0 && "${OPENWAM_PREFLIGHT_SAMPLE:-0}" == 1 ]]; then
    PREFLIGHT_ARGS+=(--sample)
  fi
  "$PYTHON" scripts/preflight_figure10_pku.py "${PREFLIGHT_ARGS[@]}"

  GPU_COUNT=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
  if (( GPU_COUNT < NPROC_PER_NODE )); then
    echo "need 8 visible GPUs per node, found $GPU_COUNT" >&2
    exit 1
  fi
fi

CMD=(
  "$PYTHON" -m torch.distributed.run
  --nnodes="$NNODES"
  --nproc_per_node="$NPROC_PER_NODE"
  --node_rank="$NODE_RANK"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  --max_restarts=0
  scripts/train.py
  --config-name "$CONFIG"
  "$@"
)

PER_GPU_BATCH="${OPENWAM_PER_GPU_BATCH:-32}"
if [[ ! "$PER_GPU_BATCH" =~ ^[1-9][0-9]*$ ]]; then
  echo "OPENWAM_PER_GPU_BATCH must be a positive integer" >&2
  exit 2
fi
GLOBAL_BATCH=$((NNODES * NPROC_PER_NODE * PER_GPU_BATCH))
printf 'OpenWAM Figure 10 launch: mode=%s node=%s/%s GPUs/node=%s per_gpu_batch=%s global_batch=%s\n' \
  "$MODE" "$NODE_RANK" "$NNODES" "$NPROC_PER_NODE" "$PER_GPU_BATCH" "$GLOBAL_BATCH"
if [[ "${OPENWAM_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
