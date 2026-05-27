#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The training smoke path is TP-first.  It keeps a single-card baseline and a
# two-card tensor-parallel run, both using the train-infer-estimation style of
# reporting measured vs estimated training iteration time.
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
MICROBATCH_COUNT="${MICROBATCH_COUNT:-1}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-sgd}"

echo "Running TP single-card training baseline..."
OUTPUT_DIR="${OUTPUT_DIR_SINGLE:-/workspace/reports/tp_single_training_smoke}" \
PROFILE_DB="${PROFILE_DB_SINGLE:-/workspace/database/train_component_profile_cambricon_mlu580_tp_single.jsonl}" \
SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
MICROBATCH_COUNT="$MICROBATCH_COUNT" \
OPTIMIZER_TYPE="$OPTIMIZER_TYPE" \
bash "$ROOT/scripts/run_cambricon_train_tp_single.sh"

echo "Running TP two-card training smoke..."
OUTPUT_DIR="${OUTPUT_DIR_TP:-/workspace/reports/tp_multi_training_smoke}" \
PROFILE_DB="${PROFILE_DB_TP:-/workspace/database/train_component_profile_cambricon_mlu580_tp2.jsonl}" \
SEQUENCE_LENGTH="$SEQUENCE_LENGTH" \
MICROBATCH_COUNT="$MICROBATCH_COUNT" \
OPTIMIZER_TYPE="$OPTIMIZER_TYPE" \
bash "$ROOT/scripts/run_cambricon_train_tp_multi.sh"

echo "Training smoke reports:"
echo "  $ROOT/reports/tp_single_training_smoke/report.json"
echo "  $ROOT/reports/tp_multi_training_smoke/report.json"
