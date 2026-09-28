#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Eight nodes run this same command with NODE_RANK=0..7 respectively.
exec "$ROOT/scripts/train_figure10_pku_8n8g.sh" ego-robot "$@"
