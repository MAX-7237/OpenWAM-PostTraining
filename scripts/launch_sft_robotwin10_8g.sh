#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VARIANT="${1:-}"
case "$VARIANT" in
  from-scratch)
    INIT=""
    MASK=action_sees_video
    ;;
  official-robot-only)
    INIT=/mnt/world_foundational/datasets/model-scaling/weights/openwam/study-pretrain/pretrain_robot_only
    MASK=action_sees_video
    ;;
  official-ego2robot)
    INIT=/mnt/world_foundational/datasets/model-scaling/weights/openwam/study-pretrain/pretrain_two_stage_robot
    MASK=action_sees_video
    ;;
  official-ego-robot-mutual)
    INIT=/mnt/world_foundational/datasets/model-scaling/weights/openwam/study-pretrain/pretrain_ego_robot_cotrain_mutual
    MASK=mutual
    ;;
  pku-ego-robot-600h)
    : "${OPENWAM_SFT_INIT:?set OPENWAM_SFT_INIT to a self-contained pretraining checkpoint directory}"
    INIT="$OPENWAM_SFT_INIT"
    MASK="${OPENWAM_SFT_MASK:-mutual}"
    ;;
  pku-ego2robot-600h)
    : "${OPENWAM_SFT_INIT:?set OPENWAM_SFT_INIT to a self-contained pretraining checkpoint directory}"
    INIT="$OPENWAM_SFT_INIT"
    MASK="${OPENWAM_SFT_MASK:-action_sees_video}"
    ;;
  pku-ego-robot-step6000)
    INIT=/media/damoxing/ckp/openwam/sft_inits/pku_ego_robot_step6000
    MASK=action_sees_video
    ;;
  *)
    echo "usage: $0 {from-scratch|official-robot-only|official-ego2robot|official-ego-robot-mutual|pku-ego-robot-600h|pku-ego2robot-600h|pku-ego-robot-step6000}" >&2
    exit 2
    ;;
esac

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

# Platform mounts are not uniform. Accept a root only after checking an actual
# episode, rather than accepting an existing but unrelated /media/datasets.
robotwin_candidates=()
if [[ -n "${ROBOTWIN_DATASET_ROOT:-}" ]]; then
  robotwin_candidates+=("$ROBOTWIN_DATASET_ROOT")
fi
robotwin_candidates+=(
  /mnt/world_foundational/datasets/RoboTwin2_0/dataset
  /media/damoxing/datasets/RoboTwin2_0/dataset
  /media/datasets/RoboTwin2_0/dataset
  /mnt/dataset/share/RoboTwin2_0/dataset
)
unset ROBOTWIN_DATASET_ROOT
for candidate in "${robotwin_candidates[@]}"; do
  if compgen -G "$candidate/adjust_bottle/aloha-agilex_clean_50/data/episode*.hdf5" >/dev/null; then
    export ROBOTWIN_DATASET_ROOT="$candidate"
    break
  fi
done

index_candidates=()
if [[ -n "${ROBOTWIN10_INDEX_ROOT:-}" ]]; then
  index_candidates+=("$ROBOTWIN10_INDEX_ROOT")
fi
index_candidates+=(
  /media/unify/unify_pretrain/data_indexes/simulation/robotwin/clean50_20260922
  /mnt/world_foundational/datasets/unify/unify_pretrain/data_indexes/simulation/robotwin/clean50_20260922
  /media/damoxing/datasets/unify/unify_pretrain/data_indexes/simulation/robotwin/clean50_20260922
  /mnt/dataset/share/unify/unify_pretrain/data_indexes/simulation/robotwin/clean50_20260922
)
unset ROBOTWIN10_INDEX_ROOT
for candidate in "${index_candidates[@]}"; do
  if [[ -f "$candidate/split_manifest.json" ]]; then
    export ROBOTWIN10_INDEX_ROOT="$candidate"
    break
  fi
done

export ROBOTWIN10_STATS_PATH="${ROBOTWIN10_STATS_PATH:-/mnt/world_foundational/datasets/model-scaling/manifests/robotwin/clean50_10task_20260922/robotwin_clean50_10task_normalization_stats.npy}"
export RUN_ID="${RUN_ID:-${VARIANT}-robotwin10-bs16-30k-$(date -u +%Y%m%d-%H%M%S)}"
export CKPT_ROOT="${CKPT_ROOT:-/media/damoxing/ckp/openwam/robotwin10_sft_8g_bs16_30k}"
export OPENWAM_SFT_OUTPUT="${CKPT_ROOT}/${VARIANT}/${RUN_ID}"

[[ -n "${OPENWAM_PYTHON:-}" && -x "$OPENWAM_PYTHON" ]] || { echo "missing smoke-tested OpenWAM Python" >&2; exit 3; }
[[ -n "${OPENWAM_WAN22_PATH:-}" ]] || { echo "missing complete Wan2.2-TI2V-5B assets" >&2; exit 3; }
[[ -n "${ROBOTWIN_DATASET_ROOT:-}" ]] || {
  echo "missing RoboTwin payload: expected adjust_bottle/aloha-agilex_clean_50/data/episode*.hdf5" >&2
  printf 'checked: %s\n' "${robotwin_candidates[@]}" >&2
  exit 3
}
[[ -n "${ROBOTWIN10_INDEX_ROOT:-}" ]] || {
  echo "missing RoboTwin 10-task split_manifest.json" >&2
  printf 'checked: %s\n' "${index_candidates[@]}" >&2
  exit 3
}

ENV_ROOT="$(cd "$(dirname "$OPENWAM_PYTHON")/.." && pwd)"
export OPENWAM_CUDNN_LIB="${OPENWAM_CUDNN_LIB:-${ENV_ROOT}/cudnn/9.5.1/lib}"
export LD_LIBRARY_PATH="${OPENWAM_CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/sft_robotwin10/${RUN_ID}}"
export WANDB_DIR="${WANDB_DIR:-${LOG_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${LOG_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${LOG_ROOT}/wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${LOG_ROOT}/wandb-data}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOG_ROOT}/xdg-cache}"
export HF_HOME="${HF_HOME:-${CKPT_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OPENWAM_SFT_OUTPUT" "$LOG_ROOT" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$XDG_CACHE_HOME" "$HF_DATASETS_CACHE"

echo "PYTHON=$OPENWAM_PYTHON"
echo "WAN22=$OPENWAM_WAN22_PATH"
echo "ROBOTWIN_DATASET_ROOT=$ROBOTWIN_DATASET_ROOT"
echo "ROBOTWIN10_INDEX_ROOT=$ROBOTWIN10_INDEX_ROOT"
echo "ROBOTWIN10_STATS_PATH=$ROBOTWIN10_STATS_PATH"

PREFLIGHT=(
  "$OPENWAM_PYTHON" scripts/preflight_sft_robotwin10.py
  --dataset-root "$ROBOTWIN_DATASET_ROOT"
  --manifest "$ROBOTWIN10_INDEX_ROOT/split_manifest.json"
  --stats "$ROBOTWIN10_STATS_PATH"
  --wan-path "$OPENWAM_WAN22_PATH"
  --sample
)
if [[ "${OPENWAM_DRY_RUN:-0}" != 1 ]]; then
  PREFLIGHT+=(--require-gpus 8)
fi
if [[ -n "$INIT" ]]; then
  PREFLIGHT+=(--init "$INIT")
fi
"${PREFLIGHT[@]}"

CMD=(
  "$OPENWAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8
  scripts/train.py --config-name sft_robotwin10
  "model.architecture.attention_mask_mode=${MASK}"
  "training.finetune_ckpt_path=${INIT:-null}"
  training.num_epochs=null
  training.max_steps=30000
  training.batch_size=16
  training.gradient_accumulation_steps=1
  training.log_every=20
  training.use_gradient_checkpointing=true
  training.use_gradient_checkpointing_offload=false
  training.save_steps=2000
  training.save_full_states_for_resume=false
  training.keep_last_k_ckpts=1
  "training.output_path=${OPENWAM_SFT_OUTPUT}"
  "project.wandb.run_name=${RUN_ID}"
)

printf 'VARIANT=%s\nINIT=%s\nMASK=%s\nOUTPUT=%s\nGLOBAL_BATCH=128\n' \
  "$VARIANT" "${INIT:-Wan2.2 base}" "$MASK" "$OPENWAM_SFT_OUTPUT"
printf 'COMMAND:'; printf ' %q' "${CMD[@]}"; printf '\n'
if [[ "${OPENWAM_DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi
"${CMD[@]}" 2>&1 | tee -a "$LOG_ROOT/train.log"
