#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/o_mabin/LLM/models/Llama-3.1-8B}"
MODEL_DIR="${MODEL_DIR:-/model}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/validation_reports/cambricon_tp_single_smoke}"
TABLE_DB="${TABLE_DB:-/workspace/database/module_profile_table_cambricon_mlu580.jsonl}"
PROMPT="${PROMPT:-alpha alpha alpha alpha alpha alpha alpha alpha}"

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
  -v "$ROOT:/workspace" \
  -v "$HOST_MODEL_DIR:$MODEL_DIR:ro" \
  -v /data:/data \
  "$IMAGE" \
  bash -lc "
    cd /workspace && \
    python torch_infer_mvp.py \
      --model-path '$MODEL_DIR' \
      --prompt '$PROMPT' \
      --max-new-tokens 2 \
      --dtype fp16 \
      --device mlu:0 \
      --parallel-mode single \
      --physical-devices 0 \
      --world-size 1 \
      --tp-size 1 \
      --warmup 0 \
      --benchmark-repeat 1 \
      --profile-repeat 1 \
      --estimate-mode online \
      --table-db-path '$TABLE_DB' \
      --table-writeback \
      --output-dir '$OUTPUT_DIR'
  "

echo "TP single-card inference report: ${OUTPUT_DIR/#\/workspace/$ROOT}/report.json"
