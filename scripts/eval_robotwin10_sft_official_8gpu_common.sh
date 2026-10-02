#!/usr/bin/env bash
set -euo pipefail

# Official-style 10-task evaluation: seed=0, 100 episodes/task, clean then
# randomized. Keep two-server-per-GPU as the requested capacity, but launch
# only one active server per task and distribute those servers round-robin
# across all visible GPUs. This avoids leaving the last GPUs idle because the
# protocol has ten tasks rather than sixteen.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VARIANT="${1:?missing variant}"
CKPT_DIR="${2:?missing checkpoint directory}"
LABEL="${3:?missing label}"
GPU_COUNT="${OPENWAM_GPU_COUNT:-8}"
SERVERS_PER_GPU="${OPENWAM_SERVERS_PER_GPU:-2}"
GPU_START="${OPENWAM_GPU_START:-0}"
PORT_BASE="${ROBOTWIN_PORT_BASE:-9200}"
EPISODES="${ROBOTWIN_TEST_NUM:-100}"
SERVER_TIMEOUT="${ROBOTWIN_SERVER_TIMEOUT:-900}"
RUN_ID="${RUN_ID:-${LABEL}-10task-seed0-100ep-$(date -u +%Y%m%d-%H%M%S)}"
EVAL_ROOT="${ROBOTWIN_EVAL_ROOT:-${ROOT}/logs/robotwin10_official/${RUN_ID}}"
EVAL_MODE="${OPENWAM_EVAL_MODE:-all}"
SERVER_CAPACITY=$((GPU_COUNT * SERVERS_PER_GPU))
TASK_FILE="$ROOT/configs/dataloader/robotwin10_clean50_tasks.txt"
POLICY_CONFIG="$ROOT/benchmarks/robotwin/policy_config.yml"

die() { echo "[ERROR] $*" >&2; exit 3; }
[[ -x "${OPENWAM_PYTHON:-}" ]] || for p in \
  /mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python \
  /media/damoxing/datasets/model-scaling/envs/openwam_cu124/bin/python; do
  [[ -x "$p" ]] && OPENWAM_PYTHON="$p" && break
done
[[ -x "${ROBOTWIN_PYTHON:-}" ]] || for p in \
  /mnt/world_foundational/datasets/model-scaling/envs/wth_Robotwin/bin/python \
  /media/damoxing/datasets/model-scaling/envs/wth_Robotwin/bin/python \
  /mnt/world_foundational_model/datasets/qingpowuwu/envs/wth_Robotwin/bin/python \
  /mnt/world_foundational_model/weixiaobao_file/miniconda3/envs/RoboTwin/bin/python; do
  [[ -x "$p" ]] && ROBOTWIN_PYTHON="$p" && break
done
if [[ -z "${ROBOTWIN_PATH:-}" ]]; then
  for p in \
    /mnt/world_foundational/datasets/model-scaling/repos/RoboTwin-Official-Eval \
    /media/damoxing/datasets/model-scaling/repos/RoboTwin-Official-Eval \
    /media/damoxing/fileset/RoboTwin-Official \
    /media/damoxing/fileset/RoboTwin; do
    if [[ -f "$p/script/eval_policy.py" && -f "$p/assets/objects/objaverse/list.json" ]]; then
      ROBOTWIN_PATH="$p"
      break
    fi
  done
fi
[[ -x "${OPENWAM_PYTHON:-}" ]] || die "missing OPENWAM_PYTHON"
[[ -x "${ROBOTWIN_PYTHON:-}" ]] || die "missing ROBOTWIN_PYTHON"
[[ -n "${ROBOTWIN_PATH:-}" && -f "$ROBOTWIN_PATH/script/eval_policy.py" ]] || die "invalid or missing ROBOTWIN_PATH=${ROBOTWIN_PATH:-unset}"
[[ -f "$ROBOTWIN_PATH/assets/objects/objaverse/list.json" ]] || die "RoboTwin assets missing under $ROBOTWIN_PATH"
for texture_split in seen unseen; do
  texture_dir="$ROBOTWIN_PATH/assets/background_texture/$texture_split"
  [[ -d "$texture_dir" ]] || die "missing RoboTwin background textures: $texture_dir; download background_texture.zip"
  compgen -G "$texture_dir/*" >/dev/null || die "empty RoboTwin background textures: $texture_dir"
done
[[ -f "$CKPT_DIR/config.yaml" ]] || die "missing checkpoint config=$CKPT_DIR/config.yaml"
compgen -G "$CKPT_DIR/checkpoint_step_*.safetensors" >/dev/null || die "missing checkpoint weights"
[[ -f "$TASK_FILE" ]] || die "missing task file=$TASK_FILE"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "ROBOTWIN_TEST_NUM must be positive"
[[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ && "$SERVERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "invalid GPU count"
case "$EVAL_MODE" in
  all|clean|randomized) ;;
  *) die "OPENWAM_EVAL_MODE must be all, clean, or randomized; got $EVAL_MODE" ;;
esac
mapfile -t TASKS < <(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "$TASK_FILE")
(( ${#TASKS[@]} == 10 )) || die "expected 10 tasks, got ${#TASKS[@]}"
(( ${#TASKS[@]} <= SERVER_CAPACITY )) || die "10 tasks exceed server capacity=${SERVER_CAPACITY}"
# There is one policy request stream per task. Starting additional idle policy
# servers only consumes memory and cannot increase throughput.
SERVER_COUNT=${#TASKS[@]}

mkdir -p "$EVAL_ROOT/servers" "$EVAL_ROOT/demo_clean" "$EVAL_ROOT/demo_randomized"
export ROBOTWIN_PATH ROBOTWIN_PYTHON POLICY_CONFIG_PATH="$POLICY_CONFIG"
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_MODE=offline
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$EVAL_ROOT/xdg-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$EVAL_ROOT/matplotlib}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
ENV_ROOT="$(cd "$(dirname "$OPENWAM_PYTHON")/.." && pwd)"
export OPENWAM_CUDNN_LIB="${OPENWAM_CUDNN_LIB:-$ENV_ROOT/cudnn/9.5.1/lib}"
OPENWAM_SERVER_PYTHONPATH="$ROOT"
OPENWAM_SERVER_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -d "$OPENWAM_CUDNN_LIB" ]]; then
  OPENWAM_SERVER_LD_LIBRARY_PATH="$OPENWAM_CUDNN_LIB${OPENWAM_SERVER_LD_LIBRARY_PATH:+:$OPENWAM_SERVER_LD_LIBRARY_PATH}"
fi
ROBOTWIN_SHARED_ROOT="${ROBOTWIN_SHARED_ROOT:-/mnt/world_foundational/datasets/model-scaling/envs/wth_Robotwin_shared}"
# cuRobo is shipped as a source dependency by RoboTwin.  Some platform
# mounts expose an empty shared-env/curobo directory, so resolve the package
# from the checkout first and fail before starting policy servers if absent.
ROBOTWIN_CUROBO_SRC="${ROBOTWIN_CUROBO_SRC:-}"
for candidate in \
  "$ROBOTWIN_PATH/envs/curobo/src" \
  "$ROBOTWIN_SHARED_ROOT/curobo/src" \
  "/mnt/world_foundational_model/weixiaobao_file/RoboTwin/envs/curobo/src"; do
  if [[ -f "$candidate/curobo/__init__.py" ]]; then
    ROBOTWIN_CUROBO_SRC="$candidate"
    break
  fi
done
[[ -n "$ROBOTWIN_CUROBO_SRC" ]] || die "missing RoboTwin curobo source; expected <ROBOTWIN_PATH>/envs/curobo/src"
# Platform copies may contain cuRobo's .git directory without release tags.
# setuptools-scm supports this non-invasive runtime fallback; it avoids any
# package installation and keeps the RoboTwin client environment isolated.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO:-0.7.0}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO}}"
ROBOTWIN_CLIENT_PYTHONPATH="${ROBOTWIN_EXTRA_PYTHONPATH:-$ROBOTWIN_SHARED_ROOT/site-packages}:$ROBOTWIN_CUROBO_SRC:$ROBOTWIN_PATH"

echo "[INFO] OpenWAM Python: $OPENWAM_PYTHON"
echo "[INFO] RoboTwin Python: $ROBOTWIN_PYTHON"
echo "[INFO] RoboTwin checkout: $ROBOTWIN_PATH"
echo "[INFO] RoboTwin curobo source: $ROBOTWIN_CUROBO_SRC"
echo "[INFO] checkpoint: $CKPT_DIR"
openwam_torch="$(PYTHONPATH="$OPENWAM_SERVER_PYTHONPATH" "$OPENWAM_PYTHON" -c 'import torch; print(torch.__file__)')" || die "OpenWAM Python cannot import its own torch"
robotwin_torch="$(PYTHONPATH="$ROBOTWIN_CLIENT_PYTHONPATH" "$ROBOTWIN_PYTHON" -c 'import torch; print(torch.__file__)')" || die "RoboTwin Python cannot import its own torch"
echo "[INFO] OpenWAM torch: $openwam_torch"
echo "[INFO] RoboTwin torch: $robotwin_torch"
echo "[INFO] checking RoboTwin Python imports"
if ! PYTHONPATH="$ROBOTWIN_CLIENT_PYTHONPATH" "$ROBOTWIN_PYTHON" -c 'import websockets' >/dev/null 2>&1; then
  ws_site="$(PYTHONPATH="$OPENWAM_SERVER_PYTHONPATH" "$OPENWAM_PYTHON" -c 'import os,site; print(next((p for p in site.getsitepackages() if os.path.isfile(os.path.join(p,"websockets","__init__.py"))),""))')"
  [[ -n "$ws_site" ]] || die "websockets is unavailable in both Python environments"
  ws_bridge="$EVAL_ROOT/python-bridge"
  mkdir -p "$ws_bridge"
  ln -sfn "$ws_site/websockets" "$ws_bridge/websockets"
  ROBOTWIN_CLIENT_PYTHONPATH="$ws_bridge:$ROBOTWIN_CLIENT_PYTHONPATH"
  echo "[INFO] reusing websockets through $ws_bridge/websockets"
fi
PYTHONPATH="$ROBOTWIN_CLIENT_PYTHONPATH" "$ROBOTWIN_PYTHON" -c 'import sapien,websockets,curobo' >/dev/null 2>&1 || {
  echo "[ERROR] RoboTwin Python lacks required sapien/websockets/curobo imports" >&2
  PYTHONPATH="$ROBOTWIN_CLIENT_PYTHONPATH" "$ROBOTWIN_PYTHON" -c 'import sapien,websockets,curobo' >&2 || true
  die "RoboTwin evaluation environment is incomplete"
}
echo "[INFO] RoboTwin Python preflight passed"

cp "$TASK_FILE" "$EVAL_ROOT/tasks.txt"
cp "$POLICY_CONFIG" "$EVAL_ROOT/policy_config.yml"
printf 'variant=%q\nlabel=%q\nrun_id=%q\nseed=0\nmode=%q\nepisodes=%q\ngpu_count=%q\nservers_per_gpu=%q\nport_base=%q\ncheckpoint=%q\nrobotwin=%q\nrobotwin_curobo_src=%q\n' \
  "$VARIANT" "$LABEL" "$RUN_ID" "$EVAL_MODE" "$EPISODES" "$GPU_COUNT" "$SERVERS_PER_GPU" "$PORT_BASE" "$CKPT_DIR" "$ROBOTWIN_PATH" "$ROBOTWIN_CUROBO_SRC" > "$EVAL_ROOT/run.env"
echo "[INFO] active server slots=$SERVER_COUNT capacity=${GPU_COUNT}x${SERVERS_PER_GPU}; task servers are round-robin across GPUs"

if [[ "${OPENWAM_EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "[INFO] evaluation preflight complete: $EVAL_ROOT"
  exit 0
fi

declare -a SERVER_PIDS SERVER_PORTS SERVER_GPUS SERVER_LOGS
kill_tree() { local pid="$1" sig="${2:-TERM}" c; [[ -z "$pid" ]] && return; while read -r c; do [[ -n "$c" ]] && kill_tree "$c" "$sig"; done < <(pgrep -P "$pid" 2>/dev/null || true); kill -"$sig" "$pid" 2>/dev/null || true; }
cleanup() { trap - EXIT INT TERM; for p in "${SERVER_PIDS[@]:-}"; do kill_tree "$p" TERM; done; sleep 2; for p in "${SERVER_PIDS[@]:-}"; do kill_tree "$p" KILL; done; }
trap cleanup EXIT INT TERM

for ((slot=0; slot<SERVER_COUNT; slot++)); do
  gpu=$((GPU_START + slot % GPU_COUNT)); port=$((PORT_BASE + slot))
  server_log="$EVAL_ROOT/servers/server_${slot}_gpu${gpu}_port${port}.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$OPENWAM_SERVER_PYTHONPATH" LD_LIBRARY_PATH="$OPENWAM_SERVER_LD_LIBRARY_PATH" \
    "$OPENWAM_PYTHON" scripts/deploy.py --ckpt-dir "$CKPT_DIR" --device cuda:0 --port "$port" \
    >"$server_log" 2>&1 &
  SERVER_PIDS+=("$!"); SERVER_PORTS+=("$port"); SERVER_GPUS+=("$gpu"); SERVER_LOGS+=("$server_log")
  echo "[INFO] server slot=$slot gpu=$gpu port=$port pid=${SERVER_PIDS[-1]}"
done

for i in "${!SERVER_PIDS[@]}"; do
  ready=0
  for _ in $(seq 1 "$SERVER_TIMEOUT"); do
    if "$OPENWAM_PYTHON" - "${SERVER_PORTS[$i]}" <<'PY' >/dev/null 2>&1
import socket,sys
with socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=1): pass
PY
    then ready=1; break; fi
    if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
      tail -120 "${SERVER_LOGS[$i]}" >&2 || true
      die "server ${i} exited before ready; log=${SERVER_LOGS[$i]}"
    fi
    sleep 1
  done
  if (( ready != 1 )); then
    tail -120 "${SERVER_LOGS[$i]}" >&2 || true
    die "server ${i} timeout; log=${SERVER_LOGS[$i]}"
  fi
done

run_mode() {
  local mode="$1" root="$EVAL_ROOT/$1"; mkdir -p "$root"; declare -a pids=()
  for ((i=0; i<10; i++)); do
    task="${TASKS[$i]}"; slot="$i"; gpu="${SERVER_GPUS[$slot]}"; port="${SERVER_PORTS[$slot]}"
    (
      export ROBOTWIN_TEST_NUM="$EPISODES" ROBOTWIN_PORT="$port" ROBOTWIN_POLICY_HOST=127.0.0.1
      export ROBOTWIN_RUNTIME_ROOT="$root/runtime_$slot" XDG_CACHE_HOME="$root/cache_$slot" MPLCONFIGDIR="$root/mpl_$slot"
      export PYTHONPATH="$ROBOTWIN_CLIENT_PYTHONPATH"
      mkdir -p "$ROBOTWIN_RUNTIME_ROOT" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
      CUDA_VISIBLE_DEVICES="$gpu" bash benchmarks/robotwin/single_eval.sh "$task" "$mode" "$LABEL" "$gpu" "$port" 127.0.0.1
    ) >"$root/${task}.log" 2>&1 & pids+=("$!")
    echo "[INFO] client mode=$mode task=$task gpu=$gpu port=$port pid=${pids[-1]}"
  done
  failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
  find "$root" -type f -name '*.log' -print0 | xargs -0r grep --color=never 'Success rate' > "$root/success_rates.txt" || true
  (( failed == 0 ))
}

case "$EVAL_MODE" in
  all)
    run_mode demo_clean
    run_mode demo_randomized
    ;;
  clean)
    run_mode demo_clean
    ;;
  randomized)
    run_mode demo_randomized
    ;;
esac
echo "[INFO] complete: $EVAL_ROOT"
