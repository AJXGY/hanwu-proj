#!/usr/bin/env python3
"""PyTorch CNCL communication microbenchmark prototype for Cambricon MLU.

This script was used to explore a torch.distributed-based path for measuring
AllReduce and Send/Recv latency on a single machine with two MLUs.
"""

import argparse
import csv
import os
import statistics
import time
from typing import Dict, List

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark communication-intensive operators on Cambricon MLU."
    )
    parser.add_argument(
        "--output",
        default="results/processed/comm_bench_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations per message size.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Measured iterations per message size.",
    )
    parser.add_argument(
        "--message-sizes",
        default="1024,4096,16384,65536,262144,1048576,4194304,16777216",
        help="Comma-separated message sizes in bytes.",
    )
    return parser.parse_args()


def get_rank() -> int:
    return int(os.environ["RANK"])


def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def device_for_rank(rank: int) -> str:
    if not hasattr(torch, "mlu") or not torch.mlu.is_available():
        raise RuntimeError("MLU is not available in this environment.")
    device_count = torch.mlu.device_count()
    if device_count < 2:
        raise RuntimeError(f"Expected 2 MLU devices, found {device_count}.")
    device_index = get_local_rank()
    torch.mlu.set_device(device_index)
    return f"mlu:{device_index}"


def sync_device() -> None:
    if hasattr(torch, "mlu") and torch.mlu.is_available():
        torch.mlu.synchronize()


def barrier() -> None:
    dist.barrier()
    sync_device()


def gather_max_time(local_ms: float) -> float:
    collected: List[float] = [0.0 for _ in range(get_world_size())]
    dist.all_gather_object(collected, local_ms)
    return max(collected)


def bench_all_reduce(num_bytes: int, device: str, warmup: int, iters: int) -> Dict[str, float]:
    elem_count = max(1, num_bytes // 4)
    tensor = torch.ones(elem_count, dtype=torch.float32, device=device)
    for _ in range(warmup):
        dist.all_reduce(tensor)
        barrier()
    times_ms = []
    for _ in range(iters):
        barrier()
        start = time.perf_counter()
        dist.all_reduce(tensor)
        sync_device()
        local_ms = (time.perf_counter() - start) * 1000.0
        barrier()
        times_ms.append(gather_max_time(local_ms))
    return summarize_times(times_ms)


def bench_send_recv(num_bytes: int, device: str, warmup: int, iters: int) -> Dict[str, float]:
    elem_count = max(1, num_bytes // 4)
    send_tensor = torch.ones(elem_count, dtype=torch.float32, device=device)
    recv_tensor = torch.zeros(elem_count, dtype=torch.float32, device=device)

    def run_once() -> float:
        barrier()
        start = time.perf_counter()
        if get_rank() == 0:
            dist.send(send_tensor, dst=1)
        elif get_rank() == 1:
            dist.recv(recv_tensor, src=0)
        sync_device()
        local_ms = (time.perf_counter() - start) * 1000.0
        barrier()
        return gather_max_time(local_ms)

    for _ in range(warmup):
        run_once()
    times_ms = [run_once() for _ in range(iters)]
    return summarize_times(times_ms)


def summarize_times(times_ms: List[float]) -> Dict[str, float]:
    avg_ms = statistics.mean(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
    }


def write_rows(output_path: str, rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "operator",
        "message_bytes",
        "avg_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "world_size",
        "device_type",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="cncl")
    rank = get_rank()
    world_size = get_world_size()
    if world_size != 2:
        raise RuntimeError(f"Expected WORLD_SIZE=2, found {world_size}.")
    device = device_for_rank(rank)
    if rank == 0:
        print(
            f"Using world_size={world_size}, local_rank={get_local_rank()}, "
            f"device_count={torch.mlu.device_count()}"
        )
    message_sizes = [int(item) for item in args.message_sizes.split(",") if item.strip()]

    rows: List[Dict[str, float]] = []
    operators = [
        ("all_reduce", bench_all_reduce),
        ("send_recv", bench_send_recv),
    ]

    for operator_name, bench_fn in operators:
        for num_bytes in message_sizes:
            result = bench_fn(num_bytes, device, args.warmup, args.iters)
            if rank == 0:
                rows.append(
                    {
                        "operator": operator_name,
                        "message_bytes": num_bytes,
                        "avg_ms": round(result["avg_ms"], 6),
                        "min_ms": round(result["min_ms"], 6),
                        "max_ms": round(result["max_ms"], 6),
                        "std_ms": round(result["std_ms"], 6),
                        "world_size": world_size,
                        "device_type": "MLU580",
                    }
                )

    if rank == 0:
        write_rows(args.output, rows)
        print(f"Wrote benchmark results to {args.output}")

    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
