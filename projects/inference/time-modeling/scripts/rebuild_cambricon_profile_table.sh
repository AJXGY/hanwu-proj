#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/o_mabin/LLM/models/Llama-3.1-8B}"
MODEL_PATH="${MODEL_PATH:-/model}"
TABLE_DB="${TABLE_DB:-/workspace/database/module_profile_table_cambricon_mlu580.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/validation_reports/cambricon_profile_rebuild}"

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
  -v "$HOST_MODEL_DIR:$MODEL_PATH:ro" \
  -v /data:/data \
  "$IMAGE" \
  bash -lc "
    cd /workspace && \
    rm -f '$TABLE_DB' && \
    python -m torch.distributed.run --standalone --nproc_per_node 2 torch_infer_mvp.py \
      --model-path '$MODEL_PATH' \
      --prompt 'alpha alpha alpha alpha alpha alpha alpha alpha' \
      --max-new-tokens 2 \
      --dtype fp16 \
      --device mlu:0 \
      --parallel-mode tp \
      --physical-devices 0,1 \
      --world-size 2 \
      --tp-size 2 \
      --warmup 1 \
      --benchmark-repeat 2 \
      --profile-repeat 2 \
      --estimate-mode online \
      --table-db-path '$TABLE_DB' \
      --table-writeback \
      --output-dir '$OUTPUT_DIR/run_0' && \
    python -m torch.distributed.run --standalone --nproc_per_node 2 torch_infer_mvp.py \
      --model-path '$MODEL_PATH' \
      --prompt 'alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha' \
      --max-new-tokens 2 \
      --dtype fp16 \
      --device mlu:0 \
      --parallel-mode tp \
      --physical-devices 0,1 \
      --world-size 2 \
      --tp-size 2 \
      --warmup 1 \
      --benchmark-repeat 2 \
      --profile-repeat 2 \
      --estimate-mode online \
      --table-db-path '$TABLE_DB' \
      --table-writeback \
      --output-dir '$OUTPUT_DIR/run_1' && \
    python -m torch.distributed.run --standalone --nproc_per_node 2 torch_infer_mvp.py \
      --model-path '$MODEL_PATH' \
      --prompt 'explain what a runtime estimator needs to measure .' \
      --max-new-tokens 2 \
      --dtype fp16 \
      --device mlu:0 \
      --parallel-mode tp \
      --physical-devices 0,1 \
      --world-size 2 \
      --tp-size 2 \
      --warmup 1 \
      --benchmark-repeat 2 \
      --profile-repeat 2 \
      --estimate-mode online \
      --table-db-path '$TABLE_DB' \
      --table-writeback \
      --output-dir '$OUTPUT_DIR/run_2'
  "

echo "Cambricon table rebuilt at: $ROOT/database/module_profile_table_cambricon_mlu580.jsonl"
echo "Model path: $HOST_MODEL_DIR"
