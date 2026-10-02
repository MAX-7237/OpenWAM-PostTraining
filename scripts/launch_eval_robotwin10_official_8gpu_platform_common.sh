#!/usr/bin/env bash
set -euo pipefail

# Platform-safe launcher for the four official-pretrain RobotWin10 SFT models.
# Keeps the OpenWAM Python 3.12 server environment isolated from the RoboTwin
# Python 3.10 simulator environment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LABEL="${1:?missing experiment label}"
CKPT_DIR="${2:?missing checkpoint directory}"
ENTRYPOINT="${3:?missing eval entrypoint}"

export OPENWAM_PYTHON="${OPENWAM_PYTHON:-/mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/world_foundational/datasets/model-scaling/envs/wth_Robotwin/bin/python}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/mnt/world_foundational/datasets/model-scaling/repos/RoboTwin-Official-Eval}"
export ROBOTWIN_SHARED_ROOT="${ROBOTWIN_SHARED_ROOT:-/mnt/world_foundational/datasets/model-scaling/envs/wth_Robotwin_shared}"
export OPENWAM_CKPT_DIR="${OPENWAM_CKPT_DIR:-$CKPT_DIR}"
# cuRobo's source checkout may be mounted without git tags. This is consumed
# by setuptools-scm during import and does not install or download anything.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO:-0.7.0}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO}}"

# Never expose RoboTwin's Python 3.10 packages to the OpenWAM Python 3.12
# policy server. The common eval script constructs both PYTHONPATHs separately.
unset PYTHONPATH
unset ROBOTWIN_EXTRA_PYTHONPATH

export OPENWAM_GPU_COUNT="${OPENWAM_GPU_COUNT:-8}"
export OPENWAM_SERVERS_PER_GPU="${OPENWAM_SERVERS_PER_GPU:-2}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"
export ROBOTWIN_PORT_BASE="${ROBOTWIN_PORT_BASE:-9500}"
export ROBOTWIN_SERVER_TIMEOUT="${ROBOTWIN_SERVER_TIMEOUT:-900}"
export OPENWAM_EVAL_MODE="${OPENWAM_EVAL_MODE:-all}"
export OPENWAM_EVAL_PREFLIGHT_ONLY="${OPENWAM_EVAL_PREFLIGHT_ONLY:-0}"
export RUN_ID="${RUN_ID:-${LABEL}-robotwin10-seed0-100ep-$(date -u +%Y%m%d-%H%M%S)}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

[[ -x "$OPENWAM_PYTHON" ]] || fail "missing OpenWAM Python: $OPENWAM_PYTHON"
[[ -x "$ROBOTWIN_PYTHON" ]] || fail "missing RoboTwin Python: $ROBOTWIN_PYTHON"
[[ -d "$ROBOTWIN_SHARED_ROOT/site-packages" ]] || fail "missing RoboTwin shared site-packages: $ROBOTWIN_SHARED_ROOT"
[[ -f "$ROBOTWIN_PATH/script/eval_policy.py" ]] || fail "missing RoboTwin eval_policy.py: $ROBOTWIN_PATH"
ROBOTWIN_CUROBO_SRC="${ROBOTWIN_CUROBO_SRC:-$ROBOTWIN_PATH/envs/curobo/src}"
[[ -f "$ROBOTWIN_CUROBO_SRC/curobo/__init__.py" ]] || fail "missing RoboTwin curobo source: $ROBOTWIN_CUROBO_SRC"
[[ -f "$ROBOTWIN_PATH/assets/objects/objaverse/list.json" ]] || fail "missing RoboTwin object assets: $ROBOTWIN_PATH"
[[ -d "$ROBOTWIN_PATH/assets/embodiments/aloha-agilex" ]] || fail "missing aloha-agilex assets: $ROBOTWIN_PATH"
for texture_split in seen unseen; do
  texture_dir="$ROBOTWIN_PATH/assets/background_texture/$texture_split"
  [[ -d "$texture_dir" ]] || fail "missing RoboTwin background textures: $texture_dir; download background_texture.zip"
  compgen -G "$texture_dir/*" >/dev/null || fail "empty RoboTwin background textures: $texture_dir"
done
[[ -f "$OPENWAM_CKPT_DIR/config.yaml" ]] || fail "missing checkpoint config: $OPENWAM_CKPT_DIR/config.yaml"
[[ -f "$OPENWAM_CKPT_DIR/checkpoint_step_30000.safetensors" ]] || fail "missing 30k checkpoint: $OPENWAM_CKPT_DIR"
[[ -f "$ROOT/scripts/$ENTRYPOINT" ]] || fail "missing eval entrypoint: $ENTRYPOINT"

echo "[INFO] label=$LABEL"
echo "[INFO] run_id=$RUN_ID"
echo "[INFO] checkpoint=$OPENWAM_CKPT_DIR"
echo "[INFO] openwam_python=$OPENWAM_PYTHON"
echo "[INFO] robotwin_python=$ROBOTWIN_PYTHON"
echo "[INFO] robotwin_path=$ROBOTWIN_PATH"
echo "[INFO] robotwin_curobo_src=$ROBOTWIN_CUROBO_SRC"
echo "[INFO] protocol=seed0 mode=${OPENWAM_EVAL_MODE} 10tasks ${ROBOTWIN_TEST_NUM}episodes"
echo "[INFO] topology=${OPENWAM_GPU_COUNT}gpus x ${OPENWAM_SERVERS_PER_GPU}servers_per_gpu"

exec bash "$ROOT/scripts/$ENTRYPOINT"
