#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/o_mabin/LLM/models/Llama-3.1-8B}"
MODEL_DIR="${MODEL_DIR:-/model}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/reports/tp_single_training}"
PROFILE_DB="${PROFILE_DB:-/workspace/database/train_component_profile_cambricon_mlu580_tp_single.jsonl}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-sgd}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
MICROBATCH_COUNT="${MICROBATCH_COUNT:-1}"

echo "Starting TP single-card training validation in Docker..."
echo "  model: $HOST_MODEL_DIR"
echo "  output: ${OUTPUT_DIR/#\/workspace/$ROOT}/report.json"

docker run --rm \
  --privileged \
  --net=host \
  --pid=host \
  --ipc=host \
  --cgroupns=host \
  --shm-size 64gb \
  -e CAMBRICON_VISIBLE_DEVICES=all \
  -e MLU_VISIBLE_DEVICE=all \
  -e PYTORCH_MLU_ALLOC_CONF=expandable_segments:True \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v "$ROOT:/workspace" \
  -v "$HOST_MODEL_DIR:$MODEL_DIR:ro" \
  -v /data:/data \
  "$IMAGE" \
  bash -lc "
    source /torch/venv3/pytorch/bin/activate && \
    echo '[train] Running single-card TP baseline...' && \
    cd /workspace && \
    python torch_train_tp_mvp.py \
      --model-path '$MODEL_DIR' \
      --dtype bf16 \
      --device mlu:0 \
      --parallel-mode single \
      --physical-devices 0 \
      --world-size 1 \
      --tp-size 1 \
      --microbatch-count '$MICROBATCH_COUNT' \
      --microbatch-size 1 \
      --sequence-length '$SEQUENCE_LENGTH' \
      --optimizer-type '$OPTIMIZER_TYPE' \
      --estimate-mode online \
      --profile-db-path '$PROFILE_DB' \
      --write-profile-db \
      --warmup 1 \
      --benchmark-repeat 3 \
      --profile-repeat 3 \
      --output-dir '$OUTPUT_DIR' \
      --enable-gradient-checkpointing
  "

echo "TP single-card training report: ${OUTPUT_DIR/#\/workspace/$ROOT}/report.json"
