#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310"

mkdir -p "${ROOT_DIR}/results/raw" "${ROOT_DIR}/results/processed" "${ROOT_DIR}/figure/strict"
rm -f "${ROOT_DIR}/results/raw/compute_op_bench.csv"
rm -f "${ROOT_DIR}/results/processed/compute_op_space_model.json"
rm -f "${ROOT_DIR}/results/processed/compute_op_validation_points.csv"
rm -f "${ROOT_DIR}/results/processed/compute_op_validation_report.csv"
rm -f "${ROOT_DIR}/figure/compute_op_validation_overview.png"
rm -f "${ROOT_DIR}/figure/strict/"*.png

docker run --rm \
  --privileged \
  --net=host \
  --pid=host \
  --ipc=host \
  --cgroupns=host \
  --shm-size 64gb \
  -e CAMBRICON_VISIBLE_DEVICES=all \
  -e MLU_VISIBLE_DEVICE=all \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v "${ROOT_DIR}:/workspace" \
  -v /data:/data \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    cd /workspace
    python3 scripts/compute_op_microbench.py \
      --output results/raw/compute_op_bench.csv \
      --dtype fp16 \
      --warmup 2 \
      --repeats 5
    python3 scripts/compute_op_space_tool.py \
      build \
      --input results/raw/compute_op_bench.csv \
      --model-output results/processed/compute_op_space_model.json \
      --target-max-error-pct 20.0
    python3 scripts/compute_op_space_tool.py \
      evaluate \
      --model results/processed/compute_op_space_model.json \
      --input results/raw/compute_op_bench.csv \
      --summary-output results/processed/compute_op_validation_points.csv \
      --report-output results/processed/compute_op_validation_report.csv \
      --plot-dir figure/strict \
      --overview-plot figure/compute_op_validation_overview.png \
      --target-max-error-pct 20.0
  '

echo "Completed compute-intensive operator space-model validation."
