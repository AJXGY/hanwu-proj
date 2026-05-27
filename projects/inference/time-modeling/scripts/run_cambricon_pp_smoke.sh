#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/home/o_mabin/LLM/models/Llama-3.1-8B}"
MODEL_DIR="${MODEL_DIR:-/model}"
OUTPUT_ROOT="/workspace/validation_reports/cambricon_pp_smoke"
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
    python torch_infer_pipeline_mvp.py \
      --model-path '$MODEL_DIR' \
      --prompt '$PROMPT' \
      --dtype fp16 \
      --device mlu:0 \
      --pp-size 1 \
      --microbatch-count 2 \
      --physical-devices 0 \
      --warmup 1 \
      --benchmark-repeat 2 \
      --output-dir '$OUTPUT_ROOT/pp1_mb2' && \
    python -m torch.distributed.run --standalone --nproc_per_node 2 torch_infer_pipeline_mvp.py \
      --model-path '$MODEL_DIR' \
      --prompt '$PROMPT' \
      --dtype fp16 \
      --device mlu:0 \
      --pp-size 2 \
      --microbatch-count 1 \
      --physical-devices 0,1 \
      --warmup 1 \
      --benchmark-repeat 2 \
      --output-dir '$OUTPUT_ROOT/pp2_mb1' && \
    python -m torch.distributed.run --standalone --nproc_per_node 2 torch_infer_pipeline_mvp.py \
      --model-path '$MODEL_DIR' \
      --prompt '$PROMPT' \
      --dtype fp16 \
      --device mlu:0 \
      --pp-size 2 \
      --microbatch-count 2 \
      --physical-devices 0,1 \
      --warmup 1 \
      --benchmark-repeat 2 \
      --output-dir '$OUTPUT_ROOT/pp2_mb2' && \
    python -m torch.distributed.run --standalone --nproc_per_node 2 torch_infer_pipeline_mvp.py \
      --model-path '$MODEL_DIR' \
      --prompt '$PROMPT' \
      --dtype fp16 \
      --device mlu:0 \
      --pp-size 2 \
      --microbatch-count 4 \
      --physical-devices 0,1 \
      --warmup 1 \
      --benchmark-repeat 2 \
      --output-dir '$OUTPUT_ROOT/pp2_mb4'
  "

echo "Pipeline smoke reports:"
echo "  $ROOT/validation_reports/cambricon_pp_smoke/pp1_mb2/report.json"
echo "  $ROOT/validation_reports/cambricon_pp_smoke/pp2_mb1/report.json"
echo "  $ROOT/validation_reports/cambricon_pp_smoke/pp2_mb2/report.json"
echo "  $ROOT/validation_reports/cambricon_pp_smoke/pp2_mb4/report.json"
echo "Model path: $HOST_MODEL_DIR"
