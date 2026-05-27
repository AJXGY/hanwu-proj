#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Rebuilding TP training profile tables..."
OUTPUT_DIR="${OUTPUT_DIR_SINGLE:-/workspace/reports/profile_rebuild_tp_single}" \
PROFILE_DB="${PROFILE_DB_SINGLE:-/workspace/database/train_component_profile_cambricon_mlu580_tp_single.jsonl}" \
bash "$ROOT/scripts/run_cambricon_train_tp_single.sh"

OUTPUT_DIR="${OUTPUT_DIR_TP:-/workspace/reports/profile_rebuild_tp2}" \
PROFILE_DB="${PROFILE_DB_TP:-/workspace/database/train_component_profile_cambricon_mlu580_tp2.jsonl}" \
bash "$ROOT/scripts/run_cambricon_train_tp_multi.sh"

echo "TP training profile tables rebuilt."
