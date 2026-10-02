#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPERIMENT="${1:-}"
case "$EXPERIMENT" in
  T1) TASK_COUNT=2 ;;
  T2) TASK_COUNT=4 ;;
  T3) TASK_COUNT=6 ;;
  T4) TASK_COUNT=8 ;;
  *) echo "usage: $0 {T1|T2|T3|T4}" >&2; exit 2 ;;
esac

mkdir -p /mnt/world_foundational
if [ ! -e /mnt/world_foundational/datasets ]; then ln -s /media/damoxing/datasets /mnt/world_foundational/datasets; fi
PYTHON="${OPENWAM_PYTHON:-/media/damoxing/datasets/model-scaling/envs/openwam_cu124/bin/python}"
DATASET_ROOT="${ROBODOJO_B_ROOT:-/media/damoxing/datasets/model-scaling/datasets/robodojo10_B}/$EXPERIMENT"
OUTPUT_ROOT="${ROBODOJO_B_OUTPUT:-/media/damoxing/datasets/model-scaling/runs/robodojo10_joint14_B_8gpu}/$EXPERIMENT"
RUN_NAME="${ROBODOJO_B_RUN_NAME:-${EXPERIMENT}_joint14_8gpu_b4_ga3_90k}"
MAX_STEPS="${ROBODOJO_B_MAX_STEPS:-90000}"
BASE_STEP="${ROBODOJO_B_BASE_STEP:-0}"
FINETUNE_CKPT="${ROBODOJO_B_FINETUNE_CKPT:-}"
REINIT_JOINT_INTERFACES="${ROBODOJO_B_REINIT_JOINT_INTERFACES:-true}"
WANDB_DIR="${ROBODOJO_B_WANDB_DIR:-/media/damoxing/datasets/model-scaling/repos/OpenWAM/wandb/$RUN_NAME}"

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 3; }
[[ -d "$DATASET_ROOT" ]] || { echo "missing dataset root: $DATASET_ROOT" >&2; exit 3; }
[[ -d "$DATASET_ROOT/cover_blocks/arx_x5/data" ]] || { echo "missing formal RoboDojo task layout" >&2; exit 3; }
if [ -n "$FINETUNE_CKPT" ]; then
  [[ -f "$FINETUNE_CKPT" ]] || { echo "missing finetune checkpoint: $FINETUNE_CKPT" >&2; exit 3; }
else
  [[ -f "/media/damoxing/datasets/model-scaling/weights/openwam/alpha-foundation/config.yaml" ]] || {
    echo "missing Alpha foundation checkpoint config" >&2
    exit 3
  }
fi

EPISODES="$(find -L "$DATASET_ROOT" -path '*/arx_x5/data/episode_*.hdf5' -type f | wc -l)"
EXPECTED="$((TASK_COUNT * 100))"
[[ "$EPISODES" -eq "$EXPECTED" ]] || {
  echo "episode count mismatch: got $EPISODES expected $EXPECTED" >&2
  exit 4
}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-wth009x-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-openwam}"
export NETRC="${NETRC:-/run/openwam-auth/netrc}"
export WANDB_DIR
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENWAM_CUDNN_LIB="${OPENWAM_CUDNN_LIB:-$(dirname "$PYTHON")/../cudnn/9.5.1/lib}"
export LD_LIBRARY_PATH="$OPENWAM_CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUTPUT_ROOT" "$WANDB_DIR"

echo "B_GROUP=$EXPERIMENT TASK_COUNT=$TASK_COUNT EPISODES=$EPISODES"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "GLOBAL_BATCH=96 (8 GPUs x batch 4 x grad accumulation 3)"
echo "TOTAL_STEPS=$MAX_STEPS SAVE_STEPS=1000 REINIT_JOINT_INTERFACES=$REINIT_JOINT_INTERFACES"
echo "BASE_STEP=$BASE_STEP TOTAL_TARGET_MICRO_STEPS=$((BASE_STEP + MAX_STEPS))"
if [ -n "$FINETUNE_CKPT" ]; then
  echo "FINETUNE_CKPT=$FINETUNE_CKPT"
else
  echo "INIT=/media/damoxing/datasets/model-scaling/weights/openwam/alpha-foundation"
fi

TRAIN_ARGS=(
  scripts/train.py --config-name robodojo_B_8gpu
  "dataloader.dataset_dir=$DATASET_ROOT"
  "training.output_path=$OUTPUT_ROOT"
  "training.max_steps=$MAX_STEPS"
  "training.reinitialize_joint_interfaces=$REINIT_JOINT_INTERFACES"
  "project.wandb.run_name=$RUN_NAME"
)
if [ -n "$FINETUNE_CKPT" ]; then
  TRAIN_ARGS+=("training.finetune_ckpt_path=$FINETUNE_CKPT" "training.resume_ckpt_path=null")
fi

exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "${TRAIN_ARGS[@]}"