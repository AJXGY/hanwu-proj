#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/o_mabin/LLM/models/Llama-3.1-8B}"
MODEL_DIR="${MODEL_DIR:-/model}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/reports/tp_multi_training}"
PROFILE_DB="${PROFILE_DB:-/workspace/database/train_component_profile_cambricon_mlu580_tp2.jsonl}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-sgd}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
MICROBATCH_COUNT="${MICROBATCH_COUNT:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

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
    cd /workspace && \
    python -m torch.distributed.run --nproc_per_node 2 \
      --master_addr '$MASTER_ADDR' \
      --master_port '$MASTER_PORT' \
      torch_train_tp_mvp.py \
      --model-path '$MODEL_DIR' \
      --dtype bf16 \
      --device mlu:0 \
      --parallel-mode tp \
      --physical-devices 0,1 \
      --world-size 2 \
      --tp-size 2 \
      --nproc-per-node 2 \
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

echo "TP multi-card training report: ${OUTPUT_DIR/#\/workspace/$ROOT}/report.json"
