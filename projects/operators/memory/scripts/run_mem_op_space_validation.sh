#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310"

mkdir -p "${ROOT_DIR}/results/raw" "${ROOT_DIR}/results/processed" "${ROOT_DIR}/figure/strict"

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
    python3 scripts/mem_op_microbench.py \
      --output results/raw/mem_op_bench.csv \
      --dtype fp16 \
      --warmup 2 \
      --repeats 15
    python3 scripts/mem_op_space_tool.py \
      build \
      --input results/raw/mem_op_bench.csv \
      --model-output results/processed/mem_op_space_model.json
    python3 scripts/mem_op_space_tool.py \
      evaluate \
      --model results/processed/mem_op_space_model.json \
      --input results/raw/mem_op_bench.csv \
      --summary-output results/processed/mem_op_validation_points.csv \
      --report-output results/processed/mem_op_validation_report.csv \
      --plot-dir figure/strict \
      --overview-plot figure/mem_op_validation_overview.png
  '

echo "Completed memory-intensive operator space-model validation."
