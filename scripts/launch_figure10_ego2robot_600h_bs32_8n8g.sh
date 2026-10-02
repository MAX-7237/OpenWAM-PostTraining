#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -eq 0 ]]; then
  exec bash scripts/launch_figure10_ego2robot_600h_bs32_8n8g_all.sh
fi

STAGE="${1:-}"
[[ "$STAGE" == ego || "$STAGE" == robot ]] || { echo "usage: $0 {ego|robot}" >&2; exit 2; }

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

LEGACY_DATASETS=/media/datasets
if [[ -d "$CANONICAL_DATASETS/pku-data" && ! -e "$LEGACY_DATASETS" && ! -L "$LEGACY_DATASETS" ]]; then
  mkdir -p "$(dirname "$LEGACY_DATASETS")"
  ln -s "$CANONICAL_DATASETS" "$LEGACY_DATASETS"
fi
if [[ -d "$CANONICAL_DATASETS/pku-data" && -d "$LEGACY_DATASETS" && ! -e "$LEGACY_DATASETS/pku-data" ]]; then
  ln -s "$CANONICAL_DATASETS/pku-data" "$LEGACY_DATASETS/pku-data"
fi
[[ -d "$LEGACY_DATASETS/pku-data" ]] || { echo "PKU payload path is unavailable: $LEGACY_DATASETS/pku-data" >&2; exit 3; }

export NNODES="${NNODES:-${HOST_NUM:-8}}" NPROC_PER_NODE=8
export NODE_RANK="${NODE_RANK:-${RANK:-}}" MASTER_ADDR="${MASTER_ADDR:-}" MASTER_PORT="${MASTER_PORT:-29500}"
export OPENWAM_PER_GPU_BATCH="${OPENWAM_PER_GPU_BATCH:-32}"
[[ "$OPENWAM_PER_GPU_BATCH" == 32 ]] || { echo "This launcher requires per-GPU batch 32" >&2; exit 2; }

if [[ -z "${OPENWAM_PYTHON:-}" ]]; then
  for candidate in \
    /mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python \
    /media/damoxing/datasets/model-scaling/envs/openwam_cu124/bin/python \
    /mnt/dataset/share/model-scaling/envs/openwam_cu124/bin/python; do
    [[ ! -x "$candidate" ]] || { export OPENWAM_PYTHON="$candidate"; break; }
  done
fi
if [[ -z "${OPENWAM_PYTHON:-}" || ! -x "$OPENWAM_PYTHON" ]]; then
  echo "Missing smoke-tested OpenWAM Python" >&2; exit 3
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
export CKPT_ROOT="${CKPT_ROOT:-/media/damoxing/ckp/openwam/figure10_ego2robot_600h_8n8g_bs32_1epoch}"
export OPENWAM_WARM_START_CKPT="${OPENWAM_WARM_START_CKPT:-}"
export OPENWAM_MAX_STEPS="${OPENWAM_MAX_STEPS:-}"

if [[ -n "$OPENWAM_WARM_START_CKPT" ]]; then
  [[ -d "$OPENWAM_WARM_START_CKPT" ]] || {
    echo "Missing warm-start checkpoint directory: $OPENWAM_WARM_START_CKPT" >&2
    exit 3
  }
  [[ -f "$OPENWAM_WARM_START_CKPT/config.yaml" ]] || {
    echo "Warm-start checkpoint is missing config.yaml: $OPENWAM_WARM_START_CKPT" >&2
    exit 3
  }
  if ! compgen -G "$OPENWAM_WARM_START_CKPT/checkpoint_step_*.safetensors" >/dev/null; then
    echo "Warm-start checkpoint has no checkpoint_step_*.safetensors: $OPENWAM_WARM_START_CKPT" >&2
    exit 3
  fi
fi
if [[ -n "$OPENWAM_MAX_STEPS" && ! "$OPENWAM_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "OPENWAM_MAX_STEPS must be a positive integer" >&2
  exit 2
fi

if [[ "$STAGE" == ego ]]; then
  export RUN_ID="${RUN_ID:-ego2robot-stage1-ego350h-8n8g-bs32-log20-20260927}"
  MODE=ego2robot-ego
else
  : "${OPENWAM_EGO_STAGE_CKPT:?set OPENWAM_EGO_STAGE_CKPT to completed stage-1 output}"
  [[ -d "$OPENWAM_EGO_STAGE_CKPT" ]] || { echo "Missing: $OPENWAM_EGO_STAGE_CKPT" >&2; exit 3; }
  if ! compgen -G "$OPENWAM_EGO_STAGE_CKPT/checkpoint_step_*.safetensors" >/dev/null; then
    echo "No checkpoint_step_*.safetensors in OPENWAM_EGO_STAGE_CKPT=$OPENWAM_EGO_STAGE_CKPT" >&2
    echo "Point it to the timestamped stage-1 run directory, not its parent RUN_ID directory" >&2
    exit 3
  fi
  export RUN_ID="${RUN_ID:-ego2robot-stage2-robot250h-8n8g-bs32-log20-20260927}"
  MODE=ego2robot-robot
fi

export OUTPUT_BASE="${CKPT_ROOT}/${RUN_ID}"
export OPENWAM_LOG_ROOT="${OPENWAM_LOG_ROOT:-${ROOT}/logs}"
export OPENWAM_PLATFORM_LOG_DIR="${OPENWAM_PLATFORM_LOG_DIR:-${OPENWAM_LOG_ROOT}/platform/${RUN_ID}}"
export WANDB_DIR="${WANDB_DIR:-${OPENWAM_LOG_ROOT}/wandb/${RUN_ID}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OPENWAM_LOG_ROOT}/wandb-cache/${RUN_ID}}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${OPENWAM_LOG_ROOT}/wandb-config/${RUN_ID}}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${OPENWAM_LOG_ROOT}/wandb-data/${RUN_ID}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENWAM_LOG_ROOT}/xdg-cache/${RUN_ID}}"

if [[ "$NNODES" != 8 || "$NPROC_PER_NODE" != 8 || ! "$NODE_RANK" =~ ^[0-7]$ || -z "$MASTER_ADDR" ]]; then
  echo "Require 8 nodes x 8 GPUs, NODE_RANK/RANK=0..7, and MASTER_ADDR" >&2; exit 2
fi
if [[ -z "${OPENWAM_WAN22_PATH:-}" || ! -f "$OPENWAM_WAN22_PATH/Wan2.2_VAE.pth" || ! -f "$OPENWAM_WAN22_PATH/diffusion_pytorch_model-00003-of-00003.safetensors" ]]; then
  echo "Complete smoke-tested Wan2.2 weights are unavailable" >&2; exit 3
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false WANDB_MODE="${WANDB_MODE:-offline}" PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENWAM_PREFLIGHT_PATHS_PER_SOURCE="${OPENWAM_PREFLIGHT_PATHS_PER_SOURCE:-3}"
export OPENWAM_PREFLIGHT_SAMPLE="${OPENWAM_PREFLIGHT_SAMPLE:-1}"
export HF_HOME="${HF_HOME:-${CKPT_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$OUTPUT_BASE" "$OPENWAM_PLATFORM_LOG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$XDG_CACHE_HOME" "$HF_DATASETS_CACHE"
echo "RUN_ID=$RUN_ID STAGE=$STAGE NODE_RANK=$NODE_RANK/$NNODES GLOBAL_BATCH=2048"
echo "CHECKPOINT_BASE=$OUTPUT_BASE"
echo "PLATFORM_LOG=$OPENWAM_PLATFORM_LOG_DIR/node_${NODE_RANK}.log"
[[ "$STAGE" != robot ]] || echo "EGO_STAGE_CKPT=$OPENWAM_EGO_STAGE_CKPT"
echo "PYTHON=$OPENWAM_PYTHON"
echo "WAN22=$OPENWAM_WAN22_PATH"
[[ -z "$OPENWAM_WARM_START_CKPT" ]] || echo "WARM_START_CKPT=$OPENWAM_WARM_START_CKPT"
[[ -z "$OPENWAM_MAX_STEPS" ]] || echo "MAX_STEPS=$OPENWAM_MAX_STEPS"
"$OPENWAM_PYTHON" -c 'import torch; print(f"TORCH={torch.__version__} CUDA={torch.version.cuda} GPUS={torch.cuda.device_count()}")'

set -o pipefail
TRAIN_OVERRIDES=(
  training.num_epochs=1 training.max_steps=null training.batch_size=32 \
  training.gradient_accumulation_steps=1 training.log_every=20 \
  training.use_gradient_checkpointing=true training.use_gradient_checkpointing_offload=false \
  training.save_steps=2000 training.keep_last_k_ckpts=1 training.save_full_states_for_resume=false \
  training.output_path="$OUTPUT_BASE" project.wandb.run_name="$RUN_ID"
)
if [[ -n "$OPENWAM_WARM_START_CKPT" ]]; then
  TRAIN_OVERRIDES+=(training.finetune_ckpt_path="$OPENWAM_WARM_START_CKPT")
fi
if [[ -n "$OPENWAM_MAX_STEPS" ]]; then
  TRAIN_OVERRIDES+=(training.num_epochs=null training.max_steps="$OPENWAM_MAX_STEPS")
fi

bash scripts/train_figure10_pku_8n8g.sh "$MODE" "${TRAIN_OVERRIDES[@]}" \
  2>&1 | tee -a "$OPENWAM_PLATFORM_LOG_DIR/node_${NODE_RANK}.log"
