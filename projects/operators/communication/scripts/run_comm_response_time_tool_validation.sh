#!/usr/bin/env bash
# Purpose:
#   Run official Cambricon communication benchmarks, build the standalone
#   response-time analysis tool model, and evaluate strict D-F outputs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310"

mkdir -p "${ROOT_DIR}/results/raw" "${ROOT_DIR}/results/processed" "${ROOT_DIR}/figure/strict"

echo "Starting communication response-time validation in Docker..."
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
    export PATH=/usr/local/neuware/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH
    COUNTS="256,1024,4096,16384,65536,262144,1048576,4194304"
    for op in allreduce sendrecv allgather alltoall reducescatter broadcast reduce; do
      echo "[comm-op] Running official benchmark: ${op}"
      /usr/local/neuware/bin/${op} \
        --special_count "${COUNTS}" \
        --threads 2 -l 5 -w 1 > "/workspace/results/raw/${op}_bench.log" 2>&1
    done

    cd /workspace
    : > results/processed/comm_bench_combined.csv
    first_csv=1
    for op in allreduce sendrecv allgather alltoall reducescatter broadcast reduce; do
      echo "[comm-op] Parsing benchmark log: ${op}"
      case "${op}" in
        allreduce) operator_name="all_reduce" ;;
        sendrecv) operator_name="send_recv" ;;
        allgather) operator_name="all_gather" ;;
        alltoall) operator_name="all_to_all" ;;
        reducescatter) operator_name="reduce_scatter" ;;
        broadcast) operator_name="broadcast" ;;
        reduce) operator_name="reduce" ;;
      esac
      python scripts/parse_cncl_benchmark.py \
        --input "results/raw/${op}_bench.log" \
        --output "results/processed/${op}_bench.csv" \
        --operator "${operator_name}"
      if [ "${first_csv}" -eq 1 ]; then
        cp "results/processed/${op}_bench.csv" results/processed/comm_bench_combined.csv
        first_csv=0
      else
        tail -n +2 "results/processed/${op}_bench.csv" >> results/processed/comm_bench_combined.csv
      fi
    done

    echo "[comm-op] Building response-time model..."
    python scripts/comm_response_time_tool.py build \
      --input results/processed/comm_bench_combined.csv \
      --model-output results/processed/comm_space_model.json

    echo "[comm-op] Evaluating validation points..."
    python scripts/comm_response_time_tool.py evaluate \
      --model results/processed/comm_space_model.json \
      --input results/processed/comm_bench_combined.csv \
      --summary-output results/processed/comm_model_validation_strict.csv \
      --report-output results/processed/comm_model_validation_report.csv \
      --plot-dir figure/strict
  '

echo "Completed strict response-time tool validation. Check results/processed and figure/strict."
