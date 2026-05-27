#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310}"
MODEL_REPO="${MODEL_REPO:-meta-llama/Meta-Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/models/Llama-3.1-8B}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required."
  echo "The Hugging Face account behind this token must already have access to the official gated Meta Llama 3.1 repo."
  exit 1
fi

mkdir -p "$ROOT/models"

docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$ROOT:/workspace" \
  "$IMAGE" \
  bash -lc "
    source /torch/venv3/pytorch/bin/activate && \
    python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='$MODEL_REPO',
    local_dir='$OUTPUT_DIR',
    token='$HF_TOKEN',
    resume_download=True,
    local_dir_use_symlinks=False,
)
print('Downloaded to:', '$OUTPUT_DIR')
PY
  "

echo "Official model directory:"
echo "  $OUTPUT_DIR"
