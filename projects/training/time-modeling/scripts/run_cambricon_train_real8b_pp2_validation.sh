#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "PP training validation is disabled in hanwu-proj. Running TP two-card validation instead."
exec bash "$ROOT/scripts/run_cambricon_train_tp_multi.sh" "$@"
