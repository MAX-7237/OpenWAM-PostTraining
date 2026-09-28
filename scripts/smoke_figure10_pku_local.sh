#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-ego-robot}"
PYTHON="${OPENWAM_PYTHON:-/mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python}"
GPU="${OPENWAM_SMOKE_GPU:-1}"
CUDNN_LIB="${OPENWAM_CUDNN_LIB:-/mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/cudnn/9.5.1/lib}"

case "$MODE" in
  robot-only)
    CONFIG=figure10_pku_robot_only
    ;;
  ego2robot-ego)
    CONFIG=figure10_pku_ego2robot_ego
    ;;
  ego2robot-robot)
    : "${OPENWAM_EGO_STAGE_CKPT:?set OPENWAM_EGO_STAGE_CKPT to a completed ego-stage checkpoint directory}"
    CONFIG=figure10_pku_ego2robot_robot
    ;;
  ego-robot)
    CONFIG=figure10_pku_ego_robot
    ;;
  *)
    echo "usage: $0 {robot-only|ego2robot-ego|ego2robot-robot|ego-robot}" >&2
    exit 2
    ;;
esac

OUTPUT="${OPENWAM_SMOKE_OUTPUT:-/tmp/openwam_figure10_${MODE//-/_}_smoke}"

cd "$ROOT"

if [[ ! -f "$CUDNN_LIB/libcudnn_graph.so.9" ]]; then
  echo "missing private cuDNN runtime: $CUDNN_LIB" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/preflight_figure10_pku.py \
  --mode "$MODE" --sample

export CUDA_VISIBLE_DEVICES="$GPU"
export DS_ACCELERATOR=cuda
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/train.py --config-name "$CONFIG" \
  training.debug=false \
  training.num_epochs=null \
  training.max_steps=1 \
  training.batch_size=1 \
  training.gradient_accumulation_steps=1 \
  training.dataset_num_workers=0 \
  training.save_steps=null \
  training.keep_last_k_ckpts=0 \
  training.use_gradient_checkpointing=true \
  training.initialize_model_on_cpu=true \
  training.offload_optimizer_device=cpu \
  training.output_path="$OUTPUT" \
  project.wandb.project=null
