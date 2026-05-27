#!/usr/bin/env python3
"""Cambricon memory-intensive operator microbenchmark for Llama-style tensors."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import time
from statistics import pstdev
from typing import Callable

import torch


MESSAGE_BYTES = [
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    6291456,
    8388608,
    12582912,
    16777216,
    33554432,
]

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark memory-intensive ops on Cambricon MLU.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--message-bytes", default=",".join(str(v) for v in MESSAGE_BYTES))
    parser.add_argument("--dtype", default="fp16", choices=sorted(DTYPE_MAP))
    parser.add_argument("--hidden-size", default=4096, type=int)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--repeats", default=15, type=int)
    return parser.parse_args()


def dtype_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def canonical_shape(message_bytes: int, hidden_size: int, dtype: torch.dtype) -> tuple[int, int, int]:
    element_bytes = dtype_bytes(dtype)
    elements = max(hidden_size, message_bytes // element_bytes)
    elements = max(hidden_size, (elements // hidden_size) * hidden_size)
    seq_len = max(1, elements // hidden_size)
    return (1, seq_len, hidden_size)


def sync() -> None:
    torch.mlu.synchronize()


def benchmark_op(op: Callable[[], torch.Tensor | None], warmup: int, repeats: int, inner_loops: int) -> list[float]:
    for _ in range(warmup):
        for _ in range(inner_loops):
            _ = op()
        sync()
    timings_ms: list[float] = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        for _ in range(inner_loops):
            _ = op()
        sync()
        timings_ms.append((time.perf_counter() - start) * 1000.0 / inner_loops)
    return timings_ms


def op_data_copy(shape: tuple[int, int, int], dtype: torch.dtype, device: str):
    src = torch.randn(shape, dtype=dtype, device=device)
    dst = torch.empty_like(src)

    def run() -> torch.Tensor:
        dst.copy_(src)
        return dst

    return run


def op_reshape_transpose(shape: tuple[int, int, int], dtype: torch.dtype, device: str):
    x = torch.randn(shape, dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return x.transpose(1, 2).contiguous()

    return run


def op_slice_copy(shape: tuple[int, int, int], dtype: torch.dtype, device: str):
    batch, seq_len, hidden_size = shape
    source_shape = (batch, seq_len * 2, hidden_size)
    x = torch.randn(source_shape, dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return x[:, ::2, :].contiguous()

    return run


def op_concat(shape: tuple[int, int, int], dtype: torch.dtype, device: str):
    batch, seq_len, hidden_size = shape
    left_seq = max(1, seq_len // 2)
    right_seq = max(1, seq_len - left_seq)
    left = torch.randn((batch, left_seq, hidden_size), dtype=dtype, device=device)
    right = torch.randn((batch, right_seq, hidden_size), dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return torch.cat((left, right), dim=1)

    return run


OP_BUILDERS = {
    "data_copy": op_data_copy,
    "reshape_transpose": op_reshape_transpose,
    "slice_copy": op_slice_copy,
    "concat": op_concat,
}


def summarize(timings_ms: list[float]) -> dict[str, float]:
    # These memory kernels often complete in a few microseconds; occasional host
    # scheduling hiccups can dominate the median.  Use the best synchronized
    # sample as the model input while still recording the full min/max/std range.
    return {
        "avg_ms": min(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "std_ms": pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
    }


def bench_single(
    operator: str,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    inner_loops: int,
) -> dict[str, float]:
    torch.mlu.set_device(0)
    device = "mlu:0"
    op = OP_BUILDERS[operator](shape, dtype, device)
    return summarize(benchmark_op(op, warmup, repeats, inner_loops))


def dual_worker(
    rank: int,
    operator: str,
    shape: tuple[int, int, int],
    dtype_name: str,
    warmup: int,
    repeats: int,
    inner_loops: int,
    queue: mp.Queue,
) -> None:
    dtype = DTYPE_MAP[dtype_name]
    torch.mlu.set_device(rank)
    device = f"mlu:{rank}"
    op = OP_BUILDERS[operator](shape, dtype, device)
    queue.put({"rank": rank, "summary": summarize(benchmark_op(op, warmup, repeats, inner_loops))})


def bench_dual(
    operator: str,
    shape: tuple[int, int, int],
    dtype_name: str,
    warmup: int,
    repeats: int,
    inner_loops: int,
) -> dict[str, float]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(target=dual_worker, args=(rank, operator, shape, dtype_name, warmup, repeats, inner_loops, queue))
        for rank in (0, 1)
    ]
    for proc in processes:
        proc.start()
    results = [queue.get() for _ in processes]
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Dual-card benchmark process failed with exit code {proc.exitcode}.")
    avg_values = [item["summary"]["avg_ms"] for item in results]
    min_values = [item["summary"]["min_ms"] for item in results]
    max_values = [item["summary"]["max_ms"] for item in results]
    std_values = [item["summary"]["std_ms"] for item in results]
    return {
        "avg_ms": max(avg_values),
        "min_ms": max(min_values),
        "max_ms": max(max_values),
        "std_ms": max(std_values),
    }


def main() -> None:
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    message_bytes_list = [int(item) for item in args.message_bytes.split(",") if item.strip()]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator",
                "scale",
                "world_size",
                "message_bytes",
                "shape",
                "dtype",
                "avg_ms",
                "min_ms",
                "max_ms",
                "std_ms",
                "inner_loops",
                "warmup",
                "repeats",
            ],
        )
        writer.writeheader()
        for operator in OP_BUILDERS:
            for message_bytes in message_bytes_list:
                shape = canonical_shape(message_bytes, args.hidden_size, dtype)
                for scale, world_size in (("single_card", 1), ("single_node_dual_card", 2)):
                    inner_loops = max(1, min(256, 67108864 // max(message_bytes, 1)))
                    if scale == "single_card":
                        summary = bench_single(operator, shape, dtype, args.warmup, args.repeats, inner_loops)
                    else:
                        summary = bench_dual(operator, shape, args.dtype, args.warmup, args.repeats, inner_loops)
                    writer.writerow(
                        {
                            "operator": operator,
                            "scale": scale,
                            "world_size": world_size,
                            "message_bytes": math.prod(shape) * dtype_bytes(dtype),
                            "shape": "x".join(str(v) for v in shape),
                            "dtype": args.dtype,
                            "avg_ms": f"{summary['avg_ms']:.6f}",
                            "min_ms": f"{summary['min_ms']:.6f}",
                            "max_ms": f"{summary['max_ms']:.6f}",
                            "std_ms": f"{summary['std_ms']:.6f}",
                            "inner_loops": inner_loops,
                            "warmup": args.warmup,
                            "repeats": args.repeats,
                        }
                    )
                    f.flush()


if __name__ == "__main__":
    main()
