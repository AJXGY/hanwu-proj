#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/projects/inference/time-modeling"
exec bash scripts/run_cambricon_infer_tp_single.sh "$@"
