#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/damoxing/datasets/model-scaling
EXPERIMENT=${1:?Usage: train_robodojo_A.sh Baseline|E1|E2|E3|E4 [--dry-run]}
shift
case "$EXPERIMENT" in Baseline|E1|E2|E3|E4) ;; *) exit 2;; esac
PY=$ROOT/envs/openwam/bin/python
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MICRO_BATCH=${MICRO_BATCH:-1}
NNODES=${NNODES:-1}
SEED=${SEED:-42}
[[ $NPROC_PER_NODE =~ ^[1-9][0-9]*$ && $MICRO_BATCH =~ ^[1-9][0-9]*$ ]] || exit 2
((96 % (NNODES * NPROC_PER_NODE * MICRO_BATCH) == 0)) || { echo 'Global batch 96 must be divisible by GPU count * micro batch'; exit 2; }
ACCUM=$((96 / (NNODES * NPROC_PER_NODE * MICRO_BATCH)))
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=false
# Prefer the matching cuDNN shipped with the training PyTorch over image libraries.
CUDNN_LIB=$ROOT/envs/cudnn-9.1.0.70/nvidia/cudnn/lib
[[ -f "$CUDNN_LIB/libcudnn.so.9" ]] || { echo "cuDNN repair is not installed; refusing to start" >&2; exit 1; }
export LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CUDNN_LIB/libcudnn.so.9${LD_PRELOAD:+:$LD_PRELOAD}"
cd "$ROOT/repos/OpenWAM"
ARGS=(dataloader=robodojo "dataloader.dataset_dir=$ROOT/datasets/robodojo10_A/$EXPERIMENT"
 "training.finetune_ckpt_path=$ROOT/weights/openwam/alpha-foundation"
 "model.video_backbone.model_path=$ROOT/weights/shared/Wan2.2-TI2V-5B"
 "model.video_backbone.encoder.model_path=$ROOT/weights/shared/Wan2.2-TI2V-5B"
 training.max_steps=30000 training.num_epochs=null training.debug=false
 "training.batch_size=$MICRO_BATCH" "training.gradient_accumulation_steps=$ACCUM"
 training.save_steps=1000 training.keep_last_k_ckpts=31 training.save_full_states_for_resume=true
 training.zero_stage=2 training.mixed_precision=bf16
 "training.output_path=$ROOT/runs/robodojo10_A/${EXPERIMENT}_seed${SEED}"
 "project.seed=$SEED" "project.wandb.run_name=${EXPERIMENT}_seed${SEED}")
if [[ ${1:-} == --dry-run ]]; then
 exec "$PY" scripts/train.py --cfg job "${ARGS[@]}"
fi
"$PY" -c "import torch; assert torch.cuda.device_count() >= $NPROC_PER_NODE, 'Not enough visible GPUs'"
exec "$PY" -m torch.distributed.run --nnodes="$NNODES" --node_rank="${NODE_RANK:-${RANK:-0}}" --master_addr="${MASTER_ADDR:-127.0.0.1}" --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-29500}" scripts/train.py "${ARGS[@]}" "$@"
