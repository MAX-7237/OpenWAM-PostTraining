#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The validated venv was created under /mnt/world_foundational/datasets and
# contains absolute links to that prefix. Recreate the equivalent mount alias
# when the platform exposes the datasource under /media/damoxing/datasets.
MEDIA_DATASETS=/media/damoxing/datasets
CANONICAL_DATASETS=/mnt/world_foundational/datasets
if [[ -d "$MEDIA_DATASETS/model-scaling" && ! -e "$CANONICAL_DATASETS" ]]; then
  mkdir -p "$(dirname "$CANONICAL_DATASETS")"
  ln -s "$MEDIA_DATASETS" "$CANONICAL_DATASETS"
fi
if [[ -d "$CANONICAL_DATASETS/model-scaling" && ! -e "$MEDIA_DATASETS" ]]; then
  mkdir -p "$(dirname "$MEDIA_DATASETS")"
  ln -s "$CANONICAL_DATASETS" "$MEDIA_DATASETS"
fi

# PKU indexes contain payload paths rooted at /media/datasets/pku-data.
LEGACY_DATASETS=/media/datasets
if [[ -d "$CANONICAL_DATASETS/pku-data" && ! -e "$LEGACY_DATASETS" && ! -L "$LEGACY_DATASETS" ]]; then
  mkdir -p "$(dirname "$LEGACY_DATASETS")"
  ln -s "$CANONICAL_DATASETS" "$LEGACY_DATASETS"
fi
if [[ -d "$CANONICAL_DATASETS/pku-data" && -d "$LEGACY_DATASETS" && ! -e "$LEGACY_DATASETS/pku-data" ]]; then
  ln -s "$CANONICAL_DATASETS/pku-data" "$LEGACY_DATASETS/pku-data"
fi
if [[ ! -d "$LEGACY_DATASETS/pku-data" ]]; then
  echo "PKU payload path is unavailable: $LEGACY_DATASETS/pku-data" >&2
  echo "Expected datasource: $CANONICAL_DATASETS/pku-data" >&2
  exit 3
fi

export RUN_ID="${RUN_ID:-ego-robot-600h-abs-eef-q99-8n8g-bs32-1epoch-log20-20260926}"
export CKPT_ROOT="${CKPT_ROOT:-/media/damoxing/ckp/openwam/figure10_ego_robot_600h_8n8g_bs32_1epoch}"
export OUTPUT_BASE="${CKPT_ROOT}/${RUN_ID}"
export OPENWAM_PER_GPU_BATCH="${OPENWAM_PER_GPU_BATCH:-32}"

if [[ "$OPENWAM_PER_GPU_BATCH" != 32 ]]; then
  echo "This launcher is fixed to per-GPU batch 32; got OPENWAM_PER_GPU_BATCH=$OPENWAM_PER_GPU_BATCH" >&2
  exit 2
fi

# Use only the environment that passed the local OpenWAM GPU smoke.
if [[ -z "${OPENWAM_PYTHON:-}" ]]; then
  for candidate in \
    /mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python \
    /media/damoxing/datasets/model-scaling/envs/openwam_cu124/bin/python \
    /mnt/dataset/share/model-scaling/envs/openwam_cu124/bin/python; do
    if [[ -x "$candidate" ]]; then
      export OPENWAM_PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -z "${OPENWAM_PYTHON:-}" || ! -x "$OPENWAM_PYTHON" ]]; then
  echo "Missing smoke-tested OpenWAM Python." >&2
  echo "Mount the complete model-scaling/envs tree at:" >&2
  echo "  /mnt/world_foundational/datasets/model-scaling/envs" >&2
  exit 3
fi

SMOKE_ENV_ROOT="$(cd "$(dirname "$OPENWAM_PYTHON")/.." && pwd)"
export OPENWAM_CUDNN_LIB="${OPENWAM_CUDNN_LIB:-${SMOKE_ENV_ROOT}/cudnn/9.5.1/lib}"

if [[ -z "${OPENWAM_WAN22_PATH:-}" ]]; then
  for candidate in \
    /mnt/world_foundational/datasets/cscsx_projects/models/Wan2.2-TI2V-5B \
    /media/damoxing/datasets/cscsx_projects/models/Wan2.2-TI2V-5B \
    /mnt/world_foundational/datasets/model-scaling/weights/shared/Wan2.2-TI2V-5B \
    /media/damoxing/datasets/model-scaling/weights/shared/Wan2.2-TI2V-5B \
    /mnt/dataset/share/model-scaling/weights/shared/Wan2.2-TI2V-5B; do
    if [[ -f "$candidate/Wan2.2_VAE.pth" && -f "$candidate/diffusion_pytorch_model-00003-of-00003.safetensors" ]]; then
      export OPENWAM_WAN22_PATH="$candidate"
      break
    fi
  done
fi

export NNODES="${NNODES:-${HOST_NUM:-8}}"
export NPROC_PER_NODE=8
export NODE_RANK="${NODE_RANK:-${RANK:-}}"
export MASTER_ADDR="${MASTER_ADDR:-}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
# Keep all runtime logs and W&B metadata on the mounted OpenWAM repository.
# This avoids W&B's attempt to create /root/.cache, which can be a file rather
# than a directory in distributed platform containers.
export OPENWAM_LOG_ROOT="${OPENWAM_LOG_ROOT:-${ROOT}/logs}"
export OPENWAM_PLATFORM_LOG_DIR="${OPENWAM_PLATFORM_LOG_DIR:-${OPENWAM_LOG_ROOT}/platform}"
export WANDB_DIR="${WANDB_DIR:-${OPENWAM_LOG_ROOT}/wandb/${RUN_ID}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OPENWAM_LOG_ROOT}/wandb-cache/${RUN_ID}}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${OPENWAM_LOG_ROOT}/wandb-config/${RUN_ID}}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${OPENWAM_LOG_ROOT}/wandb-data/${RUN_ID}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENWAM_LOG_ROOT}/xdg-cache/${RUN_ID}}"
export PYTHONUNBUFFERED=1

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENWAM_PREFLIGHT_PATHS_PER_SOURCE="${OPENWAM_PREFLIGHT_PATHS_PER_SOURCE:-3}"
export OPENWAM_PREFLIGHT_SAMPLE="${OPENWAM_PREFLIGHT_SAMPLE:-1}"

export HF_HOME="${HF_HOME:-${CKPT_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "$NNODES" != 8 ]]; then
  echo "Expected 8 nodes, got NNODES=$NNODES" >&2
  exit 2
fi
if [[ ! "$NODE_RANK" =~ ^[0-7]$ ]]; then
  echo "NODE_RANK/RANK must be one of 0..7; got '${NODE_RANK}'" >&2
  exit 2
fi
if [[ -z "$MASTER_ADDR" ]]; then
  echo "MASTER_ADDR is not set by the distributed platform" >&2
  exit 2
fi
if [[ -z "${OPENWAM_WAN22_PATH:-}" || ! -f "$OPENWAM_WAN22_PATH/Wan2.2_VAE.pth" ]]; then
  echo "Missing complete smoke-tested Wan2.2-TI2V-5B weights." >&2
  echo "Mount the model-scaling weights or cscsx_projects model datasource." >&2
  exit 3
fi

mkdir -p \
  "$OUTPUT_BASE" \
  "$OPENWAM_PLATFORM_LOG_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" \
  "$XDG_CACHE_HOME" \
  "$HF_DATASETS_CACHE"

echo "RUN_ID=$RUN_ID"
echo "NODE_RANK=$NODE_RANK/$NNODES"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "CHECKPOINT_BASE=$OUTPUT_BASE"
echo "LOG_ROOT=$OPENWAM_LOG_ROOT"
echo "PLATFORM_LOG=$OPENWAM_PLATFORM_LOG_DIR/node_${NODE_RANK}.log"
echo "WANDB_DIR=$WANDB_DIR"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "PER_GPU_BATCH=$OPENWAM_PER_GPU_BATCH"
echo "GLOBAL_BATCH=$((NNODES * NPROC_PER_NODE * OPENWAM_PER_GPU_BATCH))"
echo "LOG_EVERY=20"
echo "PYTHON=$OPENWAM_PYTHON"
echo "CUDNN=$OPENWAM_CUDNN_LIB"
echo "WAN22=$OPENWAM_WAN22_PATH"
"$OPENWAM_PYTHON" -c 'import torch; print(f"TORCH={torch.__version__} CUDA={torch.version.cuda} GPUS={torch.cuda.device_count()}")'

set -o pipefail
bash scripts/train_figure10_ego_robot_8n8g.sh \
  training.num_epochs=1 \
  training.max_steps=null \
  training.batch_size=32 \
  training.gradient_accumulation_steps=1 \
  training.log_every=20 \
  training.use_gradient_checkpointing=true \
  training.use_gradient_checkpointing_offload=false \
  training.save_steps=2000 \
  training.keep_last_k_ckpts=1 \
  training.output_path="$OUTPUT_BASE" \
  project.wandb.run_name="$RUN_ID" \
  2>&1 | tee -a "$OPENWAM_PLATFORM_LOG_DIR/node_${NODE_RANK}.log"
