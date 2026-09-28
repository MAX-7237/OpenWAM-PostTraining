#!/usr/bin/env bash
# Read-only readiness and command template for the 10-task RoboDojo eval.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_SCALING_ROOT:-/media/damoxing/datasets/model-scaling}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${MODEL_ROOT}/repos/RoboDojo}"
DATA_ROOT="${ROBODOJO_DATA_ROOT:-${MODEL_ROOT}/datasets/RoboDojo}"
MANIFEST="${ROBODOJO_MANIFEST:-${MODEL_ROOT}/manifests/robodojo_10task_v1.json}"
CKPT="${OPENWAM_CKPT:-${MODEL_ROOT}/weights/openwam/alpha-foundation/checkpoint_step_154000.safetensors}"
ENV_PY="${OPENWAM_PYTHON:-${MODEL_ROOT}/envs/openwam/bin/python}"
POLICY_DIR="${XPL_OPENWAM_POLICY_DIR:-${MODEL_ROOT}/repos/XPolicyLab/policy/OpenWAM}"

TASKS=(
  arrange_largest_number
  cover_blocks
  fasten_screws
  fold_clothes
  match_and_pick_from_conveyor
  play_tic_tac_toe
  pour_balls_into_vase
  organize_table
  stack_bowls
  classify_objects
)

pass=0; warn=0; fail=0
ok() { printf '[PASS] %s\n' "$*"; pass=$((pass + 1)); }
warning() { printf '[WARN] %s\n' "$*"; warn=$((warn + 1)); }
bad() { printf '[FAIL] %s\n' "$*"; fail=$((fail + 1)); }
check_file() { [[ -f "$1" ]] && ok "$2: $1" || bad "$2 missing: $1"; }
check_dir() { [[ -d "$1" ]] && ok "$2: $1" || bad "$2 missing: $1"; }

printf '%s\n' "RoboDojo 10-task evaluation preflight (read-only)"
printf 'model_root=%s\nrobodojo_root=%s\ndata_root=%s\nmanifest=%s\ncheckpoint=%s\n' \
  "$MODEL_ROOT" "$ROBODOJO_ROOT" "$DATA_ROOT" "$MANIFEST" "$CKPT"

check_file "$MANIFEST" "fixed manifest"
check_file "$ROBODOJO_ROOT/scripts/robodojo.sh" "RoboDojo launcher"
check_file "$ROBODOJO_ROOT/scripts/eval_policy.sh" "RoboDojo eval client"
check_file "$ROBODOJO_ROOT/env_cfg/arx_x5.yml" "ARX X5 environment config"
check_file "$ENV_PY" "OpenWAM Python"
check_file "$CKPT" "OpenWAM checkpoint"
check_dir "$DATA_ROOT" "RoboDojo HDF5 data root"

if [[ -d "$ROBODOJO_ROOT/Assets/Robots" && -d "$ROBODOJO_ROOT/Assets/Object/RoboDojo" && \
      -d "$ROBODOJO_ROOT/Assets/Eval_Layout/RoboDojo" ]]; then
  ok "RoboDojo simulator assets"
else
  bad "RoboDojo simulator assets incomplete (run scripts/init_assets.sh)"
fi

if [[ -d "$POLICY_DIR" && -f "$POLICY_DIR/eval.sh" && -f "$POLICY_DIR/setup_eval_policy_server.sh" ]]; then
  ok "OpenWAM XPolicyLab adapter: $POLICY_DIR"
else
  bad "OpenWAM XPolicyLab adapter missing: expected eval.sh and setup_eval_policy_server.sh under $POLICY_DIR"
fi

if [[ -f "$MANIFEST" ]]; then
  if "${ENV_PY}" - "$MANIFEST" "${TASKS[@]}" <<'PY'
import json, sys
p = sys.argv[1]
wanted = sys.argv[2:]
d = json.load(open(p, encoding="utf-8"))
actual = list(d.get("tasks", {}))
missing = [t for t in wanted if t not in actual]
if missing:
    print("[FAIL] manifest missing tasks: " + ", ".join(missing))
    raise SystemExit(1)
episodes = d.get("episodes", [])
seen = {e.get("split") for e in episodes}
print(f"[PASS] manifest tasks={len(actual)} episodes={len(episodes)} splits={sorted(seen)}")
PY
  then pass=$((pass + 1)); else fail=$((fail + 1)); fi
fi

printf '\nTask list for ID/OOD smoke:\n'
printf '  %s\n' "${TASKS[@]}"
printf '\nOfficial client dry-run templates (requires an OpenWAM adapter):\n'
for task in "${TASKS[@]}"; do
  printf '  EVAL_NUM=1 bash %q eval --policy-dir %q --task %q --ckpt %q --policy-env %q --eval-env RoboDojo --env-cfg arx_x5 --action-type ee --env-gpu 0 --policy-gpu 1 --dry-run\n' \
    "$ROBODOJO_ROOT/scripts/robodojo.sh" "$POLICY_DIR" "$task" "$CKPT" "$ENV_PY"
done
printf '\nID/OOD protocol:\n'
printf '  ID: fixed official task configs, seeds 0/1/2, EVAL_NUM=native; report per-task success_rate and mean score.\n'
printf '  OOD: same 10 task families with held-out/random variants only where official config exists; record exact variant/config and seed.\n'
printf '  Smoke: use EVAL_NUM=1 for one episode per task before full evaluation.\n'

printf '\nSummary: pass=%d warn=%d fail=%d\n' "$pass" "$warn" "$fail"
if (( fail > 0 )); then
  printf 'Readiness: BLOCKED\n'
  exit 1
fi
printf 'Readiness: READY FOR SMOKE (subject to policy adapter and simulator runtime)\n'
