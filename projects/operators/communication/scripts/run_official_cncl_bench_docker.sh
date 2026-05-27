#!/usr/bin/env bash
# Purpose:
#   Run the official Cambricon CNCL benchmarks in Docker, then parse logs and
#   generate processed CSV summaries and comparison figures in this project.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310"

mkdir -p "${ROOT_DIR}/results/raw" "${ROOT_DIR}/results/processed" "${ROOT_DIR}/figure"

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
    export PATH=/usr/local/neuware/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH
    COUNTS="256,1024,4096,16384,65536,262144,1048576,4194304"
    for op in allreduce sendrecv allgather alltoall reducescatter broadcast reduce; do
      /usr/local/neuware/bin/${op} \
        --special_count "${COUNTS}" \
        --threads 2 -l 5 -w 1 > "/workspace/results/raw/${op}_bench.log" 2>&1
    done
    cd /workspace
    : > results/processed/comm_bench_combined.csv
    first_csv=1
    for op in allreduce sendrecv allgather alltoall reducescatter broadcast reduce; do
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
    python scripts/analyze_comm_results.py \
      --input results/processed/comm_bench_combined.csv \
      --summary-output results/processed/comm_model_summary.csv \
      --plot-output figure/comm_model_vs_real.png
  '

echo "Completed benchmark pipeline. Check results/raw, results/processed, and figure/."
