#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
PYTHON="${OPENWAM_PYTHON:-/mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python}"
GPU="${OPENWAM_SMOKE_GPU:-0}"
WAN="${OPENWAM_WAN22_PATH:-/mnt/world_foundational/datasets/cscsx_projects/models/Wan2.2-TI2V-5B}"
ENV_ROOT="$(cd "$(dirname "$PYTHON")/.." && pwd)"
export LD_LIBRARY_PATH="${OPENWAM_CUDNN_LIB:-${ENV_ROOT}/cudnn/9.5.1/lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OPENWAM_PYTHON="$PYTHON" OPENWAM_WAN22_PATH="$WAN"
export ROBOTWIN_DATASET_ROOT="${ROBOTWIN_DATASET_ROOT:-/media/datasets/RoboTwin2_0/dataset}"
export ROBOTWIN10_INDEX_ROOT="${ROBOTWIN10_INDEX_ROOT:-/media/unify/unify_pretrain/data_indexes/simulation/robotwin/clean50_20260922}"
export ROBOTWIN10_STATS_PATH="${ROBOTWIN10_STATS_PATH:-/mnt/world_foundational/datasets/model-scaling/manifests/robotwin/clean50_10task_20260922/robotwin_clean50_10task_normalization_stats.npy}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_ACCELERATOR=cuda

variants=(from-scratch official-robot-only official-ego2robot official-ego-robot-mutual pku-ego-robot-step6000)
if [[ "$MODE" != all ]]; then
  variants=("$MODE")
fi

for variant in "${variants[@]}"; do
  case "$variant" in
    from-scratch) init=""; mask=action_sees_video ;;
    official-robot-only) init=/mnt/world_foundational/datasets/model-scaling/weights/openwam/study-pretrain/pretrain_robot_only; mask=action_sees_video ;;
    official-ego2robot) init=/mnt/world_foundational/datasets/model-scaling/weights/openwam/study-pretrain/pretrain_two_stage_robot; mask=action_sees_video ;;
    official-ego-robot-mutual) init=/mnt/world_foundational/datasets/model-scaling/weights/openwam/study-pretrain/pretrain_ego_robot_cotrain_mutual; mask=mutual ;;
    pku-ego-robot-step6000) init=/media/damoxing/ckp/openwam/sft_inits/pku_ego_robot_step6000; mask=action_sees_video ;;
    *) echo "unknown smoke variant: $variant" >&2; exit 2 ;;
  esac

  output="${OPENWAM_SMOKE_OUTPUT_ROOT:-/tmp/openwam_sft_robotwin10_smoke}/${variant}-$(date -u +%Y%m%d-%H%M%S)"
  preflight=("$PYTHON" scripts/preflight_sft_robotwin10.py --dataset-root "$ROBOTWIN_DATASET_ROOT" \
    --manifest "$ROBOTWIN10_INDEX_ROOT/split_manifest.json" --stats "$ROBOTWIN10_STATS_PATH" \
    --wan-path "$WAN" --sample --require-gpus 1)
  [[ -z "$init" ]] || preflight+=(--init "$init")
  CUDA_VISIBLE_DEVICES="$GPU" "${preflight[@]}"

  echo "SMOKE START: $variant"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    scripts/train.py --config-name sft_robotwin10 \
    "model.architecture.attention_mask_mode=${mask}" \
    "training.finetune_ckpt_path=${init:-null}" \
    training.num_epochs=null training.max_steps=1 training.batch_size=1 \
    training.gradient_accumulation_steps=1 training.log_every=1 \
    training.dataset_num_workers=0 training.save_steps=null training.keep_last_k_ckpts=0 \
    training.use_gradient_checkpointing=true training.initialize_model_on_cpu=true \
    training.offload_optimizer_device=cpu "training.output_path=${output}" \
    project.wandb.project=null
  echo "SMOKE PASS: $variant"
done
