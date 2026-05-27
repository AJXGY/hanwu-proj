#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/projects/training/time-modeling"
exec bash scripts/run_cambricon_train_tp_single.sh "$@"
