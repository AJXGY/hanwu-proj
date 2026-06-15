#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/hanwu-time-modeling}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/o_mabin/LLM/models/Llama-3.1-8B}"
MODEL_DIR="${MODEL_DIR:-/model}"
OUTPUT_DIR="${OUTPUT_DIR:-$CONTAINER_ROOT/validation_reports/cambricon_tp2_smoke}"
TABLE_DB="${TABLE_DB:-$CONTAINER_ROOT/database/module_profile_table_cambricon_mlu580.jsonl}"
PROMPT="${PROMPT:-alpha alpha alpha alpha alpha alpha alpha alpha}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29502}"

echo "Starting TP two-card inference validation in Docker..."
echo "  model: $HOST_MODEL_DIR"
echo "  output: ${OUTPUT_DIR/#$CONTAINER_ROOT/$ROOT}/report.json"
echo "  master: $MASTER_ADDR:$MASTER_PORT"

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
  -v "$ROOT:$CONTAINER_ROOT" \
  -v "$HOST_MODEL_DIR:$MODEL_DIR:ro" \
  -v /data:/data \
  "$IMAGE" \
  bash -lc "
    source /torch/venv3/pytorch_infer/bin/activate && \
    echo '[infer] Running two-card tensor-parallel validation...' && \
    cd '$CONTAINER_ROOT' && \
    python -m torch.distributed.run --nproc_per_node 2 \
      --master_addr '$MASTER_ADDR' \
      --master_port '$MASTER_PORT' \
      torch_infer_mvp.py \
      --model-path '$MODEL_DIR' \
      --prompt '$PROMPT' \
      --max-new-tokens 2 \
      --dtype fp16 \
      --device mlu:0 \
      --parallel-mode tp \
      --physical-devices 0,1 \
      --world-size 2 \
      --tp-size 2 \
      --warmup 0 \
      --benchmark-repeat 1 \
      --profile-repeat 1 \
      --estimate-mode online \
      --table-db-path '$TABLE_DB' \
      --table-writeback \
      --output-dir '$OUTPUT_DIR'
  "

echo "TP two-card inference report: ${OUTPUT_DIR/#$CONTAINER_ROOT/$ROOT}/report.json"
echo "Model path: $HOST_MODEL_DIR"
