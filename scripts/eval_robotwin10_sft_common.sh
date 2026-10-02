#!/usr/bin/env bash
set -euo pipefail

# Fixed 10-task evaluation for the Robotwin10 SFT comparison.
# The policy server owns the checkpoint; the RoboTwin client only runs the
# simulator and talks to the server over WebSocket.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VARIANT="${1:?missing variant}"
CKPT_DIR="${2:?missing checkpoint directory}"
LABEL="${3:?missing run label}"

case "$VARIANT" in
  from-scratch|robot-only|ego2robot|ego-robot-mutual) ;;
  *) echo "unsupported variant: $VARIANT" >&2; exit 2 ;;
esac

if [[ -z "${OPENWAM_PYTHON:-}" ]]; then
  for candidate in \
    /mnt/world_foundational/datasets/model-scaling/envs/openwam_cu124/bin/python \
    /media/damoxing/datasets/model-scaling/envs/openwam_cu124/bin/python \
    /mnt/dataset/share/model-scaling/envs/openwam_cu124/bin/python \
    /media/damoxing/datasets/model-scaling/envs/openwam_cu12.x/bin/python \
    /media/damoxing/datasets/model-scaling/envs/calib_cu121/bin/python; do
    if [[ -x "$candidate" ]]; then OPENWAM_PYTHON="$candidate"; break; fi
  done
fi

if [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
  for candidate in \
    /media/damoxing/fileset/RoboTwin/.conda/envs/robotwin/bin/python \
    /media/damoxing/fileset/RoboTwin/envs/robotwin/bin/python \
    /media/damoxing/datasets/model-scaling/envs/robotwin/bin/python \
    /mnt/world_foundational/datasets/model-scaling/envs/robotwin/bin/python \
    /mnt/world_foundational_model/datasets/qingpowuwu/envs/wth_Robotwin/bin/python \
    /mnt/world_foundational_model/weixiaobao_file/miniconda3/envs/RoboTwin/bin/python \
    /opt/conda/envs/robotwin/bin/python; do
    if [[ -x "$candidate" ]]; then ROBOTWIN_PYTHON="$candidate"; break; fi
  done
fi

if [[ -z "${ROBOTWIN_PATH:-}" ]]; then
  if [[ -f /media/damoxing/fileset/RoboTwin-Official/assets/objects/objaverse/list.json ]]; then
    ROBOTWIN_PATH=/media/damoxing/fileset/RoboTwin-Official
  else
    ROBOTWIN_PATH=/media/damoxing/fileset/RoboTwin
  fi
fi
PORT="${ROBOTWIN_PORT:-8848}"
GPU="${ROBOTWIN_EVAL_GPU:-0}"
EPISODES="${ROBOTWIN_TEST_NUM:-1}"
MODE="${ROBOTWIN_EVAL_MODE:-demo_clean}"
SERVER_TIMEOUT="${ROBOTWIN_SERVER_TIMEOUT:-600}"
RUN_ID="${RUN_ID:-${LABEL}-10task-${MODE}-$(date -u +%Y%m%d-%H%M%S)}"
EVAL_ROOT="${ROBOTWIN_EVAL_ROOT:-${ROOT}/logs/robotwin10_eval/${RUN_ID}}"
SERVER_LOG="${EVAL_ROOT}/policy_server.log"

TASK_FILE="$ROOT/configs/dataloader/robotwin10_clean50_tasks.txt"
POLICY_CONFIG="$ROOT/benchmarks/robotwin/policy_config.yml"

mkdir -p "$EVAL_ROOT"

die() { echo "[ERROR] $*" >&2; exit 3; }
[[ -x "${OPENWAM_PYTHON:-}" ]] || die "missing OPENWAM_PYTHON; set it to the smoke-tested OpenWAM environment"
[[ -x "${ROBOTWIN_PYTHON:-}" ]] || die "missing ROBOTWIN_PYTHON; set it to the RoboTwin environment python"
[[ -d "$ROBOTWIN_PATH" && -f "$ROBOTWIN_PATH/script/eval_policy.py" ]] || die "invalid ROBOTWIN_PATH: $ROBOTWIN_PATH"
[[ -d "$CKPT_DIR" ]] || die "checkpoint directory not found: $CKPT_DIR"
[[ -f "$CKPT_DIR/config.yaml" ]] || die "checkpoint config missing: $CKPT_DIR/config.yaml"
compgen -G "$CKPT_DIR/checkpoint_step_*.safetensors" >/dev/null || die "checkpoint weights missing: $CKPT_DIR"
[[ -f "$TASK_FILE" ]] || die "task list missing: $TASK_FILE"
[[ "$MODE" == demo_clean || "$MODE" == demo_randomized ]] || die "invalid ROBOTWIN_EVAL_MODE=$MODE"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "ROBOTWIN_TEST_NUM must be a positive integer"
[[ "$SERVER_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "ROBOTWIN_SERVER_TIMEOUT must be a positive integer"

export PYTHONPATH="$ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOTWIN_PATH ROBOTWIN_PYTHON ROBOTWIN_TEST_NUM="$EPISODES"
export ROBOTWIN_PORT="$PORT" ROBOTWIN_POLICY_HOST="${ROBOTWIN_POLICY_HOST:-127.0.0.1}"
export POLICY_CONFIG_PATH="$POLICY_CONFIG"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ENV_ROOT="$(cd "$(dirname "$OPENWAM_PYTHON")/.." && pwd)"
export OPENWAM_CUDNN_LIB="${OPENWAM_CUDNN_LIB:-${ENV_ROOT}/cudnn/9.5.1/lib}"
if [[ -d "$OPENWAM_CUDNN_LIB" ]]; then
  export LD_LIBRARY_PATH="$OPENWAM_CUDNN_LIB${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${EVAL_ROOT}/xdg-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${EVAL_ROOT}/matplotlib}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# The local RoboTwin env contains SAPIEN/cuRobo, while the OpenWAM env already
# contains the pure-Python websocket client dependency. Reuse it when the
# RoboTwin interpreter does not provide websockets; no package is installed.
if ! "$ROBOTWIN_PYTHON" -c 'import websockets' >/dev/null 2>&1; then
  websocket_site="$($OPENWAM_PYTHON - <<'PY'
import os, site
for path in site.getsitepackages():
    if os.path.isfile(os.path.join(path, "websockets", "__init__.py")):
        print(path)
        break
PY
)"
  if [[ -n "$websocket_site" ]]; then
    export PYTHONPATH="$websocket_site:$PYTHONPATH"
    echo "[INFO] reusing websockets from $websocket_site"
  fi
fi
"$ROBOTWIN_PYTHON" -c 'import sapien, websockets' >/dev/null 2>&1 || \
  die "ROBOTWIN_PYTHON lacks required sapien/websockets imports: $ROBOTWIN_PYTHON"

latest_ckpt="$($OPENWAM_PYTHON - "$CKPT_DIR" <<'PY'
import glob, os, re, sys
root = sys.argv[1]
items = []
for path in glob.glob(os.path.join(root, "checkpoint_step_*.safetensors")):
    m = re.search(r"checkpoint_step_(\d+)\.safetensors$", path)
    if m:
        items.append((int(m.group(1)), path))
if not items:
    raise SystemExit("no checkpoint_step_<int>.safetensors found")
print(max(items)[1])
PY
)"

cp "$TASK_FILE" "$EVAL_ROOT/tasks.txt"
cp "$POLICY_CONFIG" "$EVAL_ROOT/policy_config.yml"
cp "$CKPT_DIR/config.yaml" "$EVAL_ROOT/checkpoint_config.yaml"
{
  printf 'variant=%q\nlabel=%q\nrun_id=%q\nmode=%q\nepisodes=%q\nserver_timeout=%q\n' "$VARIANT" "$LABEL" "$RUN_ID" "$MODE" "$EPISODES" "$SERVER_TIMEOUT"
  printf 'checkpoint_dir=%q\ncheckpoint_file=%q\nrobotwin_path=%q\nrobotwin_python=%q\nopenwam_python=%q\n' \
    "$CKPT_DIR" "$latest_ckpt" "$ROBOTWIN_PATH" "$ROBOTWIN_PYTHON" "$OPENWAM_PYTHON"
  git -C "$ROBOTWIN_PATH" rev-parse HEAD 2>/dev/null | sed 's/^/robotwin_commit=/' || true
  git -C "$ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/openwam_commit=/' || true
} > "$EVAL_ROOT/run.env"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    echo "[INFO] stopping policy server pid=$server_pid"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] variant=$VARIANT mode=$MODE episodes=$EPISODES"
echo "[INFO] checkpoint=$latest_ckpt"
echo "[INFO] robotwin=$ROBOTWIN_PATH python=$ROBOTWIN_PYTHON"
echo "[INFO] tasks=$(tr '\n' ' ' < "$TASK_FILE")"

"$OPENWAM_PYTHON" scripts/deploy.py \
  --ckpt-dir "$CKPT_DIR" \
  --device "cuda:$GPU" \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!
echo "[INFO] policy server pid=$server_pid log=$SERVER_LOG"

healthy=0
for _ in $(seq 1 "$SERVER_TIMEOUT"); do
  if "$ROBOTWIN_PYTHON" - "$ROBOTWIN_POLICY_HOST" "$PORT" <<'PY' >/dev/null 2>&1
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=1):
    pass
PY
  then healthy=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -80 "$SERVER_LOG" >&2 || true
    die "policy server exited before becoming ready"
  fi
  sleep 1
done
(( healthy == 1 )) || { tail -80 "$SERVER_LOG" >&2 || true; die "policy server did not open port $PORT"; }

export ROBOTWIN_LOG_ROOT="$EVAL_ROOT/client"
mkdir -p "$ROBOTWIN_LOG_ROOT"
bash benchmarks/robotwin/multi_eval.sh \
  -m "$MODE" \
  -n "$LABEL" \
  -d "$CKPT_DIR" \
  --host "$ROBOTWIN_POLICY_HOST" \
  --port "$PORT" \
  -g "$GPU" \
  "$TASK_FILE" 2>&1 | tee "$EVAL_ROOT/eval.log"

find "$EVAL_ROOT/client" -type f -name '*.log' -print0 2>/dev/null \
  | xargs -0r grep --color=never 'Success rate' \
  | tee "$EVAL_ROOT/success_rates.txt" || true
echo "[INFO] evaluation complete: $EVAL_ROOT"
