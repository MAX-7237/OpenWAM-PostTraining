#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/damoxing/datasets/model-scaling
EXPERIMENT=${1:?Usage: train_robodojo_joint_A.sh Baseline|E1|E2|E3|E4 micro|optimizer [overrides...]}
UNIT=${2:?Specify micro or optimizer explicitly}
shift 2
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MICRO_BATCH=${MICRO_BATCH:-1}
NNODES=${NNODES:-1}
[[ $NPROC_PER_NODE =~ ^[1-9][0-9]*$ && $MICRO_BATCH =~ ^[1-9][0-9]*$ ]] || exit 2
((96 % (NNODES * NPROC_PER_NODE * MICRO_BATCH) == 0)) || exit 2
ACCUM=$((96 / (NNODES * NPROC_PER_NODE * MICRO_BATCH)))
case "$UNIT" in
  micro) MAX_MICRO_STEPS=30000 ;;
  optimizer) MAX_MICRO_STEPS=$((30000 * ACCUM)) ;;
  *) echo 'Step unit must be micro or optimizer' >&2; exit 2 ;;
esac
export NPROC_PER_NODE MICRO_BATCH NNODES
# Checkpoint cadence remains the explicitly requested 1000 micro-steps.
exec bash "$ROOT/scripts/train_robodojo_A.sh" "$EXPERIMENT" \
  dataloader=robodojo_joint \
  model.architecture.action_dim=14 model.architecture.state_dim=14 \
  +training.reinitialize_joint_interfaces=true \
  "training.max_steps=$MAX_MICRO_STEPS" \
  "training.output_path=$ROOT/runs/robodojo10_joint_A/${EXPERIMENT}_${UNIT}_seed${SEED:-42}" \
  "project.wandb.run_name=${EXPERIMENT}_joint14_${UNIT}_seed${SEED:-42}" \
  "$@"
