#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Deprecated PP1 validation entry. Running TP single-card baseline instead."
exec bash "$ROOT/scripts/run_cambricon_train_tp_single.sh" "$@"
