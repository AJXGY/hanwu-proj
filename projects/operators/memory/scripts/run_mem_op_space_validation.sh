#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310"

mkdir -p "${ROOT_DIR}/results/raw" "${ROOT_DIR}/results/processed" "${ROOT_DIR}/figure/strict"

echo "Starting memory-intensive operator validation in Docker..."
echo "  workspace: ${ROOT_DIR}"
echo "  image: ${IMAGE}"

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
    source /torch/venv3/pytorch/bin/activate
    cd /workspace
    echo "[mem-op] Running microbenchmark..."
    python3 scripts/mem_op_microbench.py \
      --output results/raw/mem_op_bench.csv \
      --dtype fp16 \
      --warmup 2 \
      --repeats 15
    echo "[mem-op] Building space model..."
    python3 scripts/mem_op_space_tool.py \
      build \
      --input results/raw/mem_op_bench.csv \
      --model-output results/processed/mem_op_space_model.json
    echo "[mem-op] Evaluating validation points..."
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
