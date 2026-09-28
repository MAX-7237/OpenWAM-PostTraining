#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Four nodes run this same command with NODE_RANK=0,1,2,3 respectively.
exec "$ROOT/scripts/train_figure10_pku_4n8g.sh" robot-only "$@"
