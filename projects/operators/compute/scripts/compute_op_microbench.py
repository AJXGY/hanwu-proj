#!/usr/bin/env python3
"""Cambricon compute-intensive operator microbenchmark for Llama kernels."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import time
from statistics import mean, pstdev
from typing import Callable

import torch
import torch.nn.functional as F


SEQ_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compute-intensive ops on Cambricon MLU.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--seq-lengths", default=",".join(str(v) for v in SEQ_LENGTHS))
    parser.add_argument("--dtype", default="fp16", choices=sorted(DTYPE_MAP))
    parser.add_argument("--hidden-size", default=4096, type=int)
    parser.add_argument("--intermediate-size", default=14336, type=int)
    parser.add_argument("--num-heads", default=32, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--repeats", default=5, type=int)
    return parser.parse_args()


def sync() -> None:
    torch.mlu.synchronize()


def benchmark_op(op: Callable[[], torch.Tensor], warmup: int, repeats: int, inner_loops: int) -> list[float]:
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


def summarize(timings_ms: list[float]) -> dict[str, float]:
    return {
        "avg_ms": mean(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "std_ms": pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
    }


def flops_for_gemm(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def flops_for_flash_attention(seq_len: int, hidden_size: int) -> int:
    return 4 * seq_len * seq_len * hidden_size


def op_mlp_up(seq_len: int, hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: str):
    x = torch.randn((seq_len, hidden_size), dtype=dtype, device=device)
    w = torch.randn((hidden_size, intermediate_size), dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return x @ w

    return run, {
        "m": seq_len,
        "k": hidden_size,
        "n": intermediate_size,
        "flops": flops_for_gemm(seq_len, hidden_size, intermediate_size),
    }


def op_mlp_gate(seq_len: int, hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: str):
    x = torch.randn((seq_len, hidden_size), dtype=dtype, device=device)
    w = torch.randn((hidden_size, intermediate_size), dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return x @ w

    return run, {
        "m": seq_len,
        "k": hidden_size,
        "n": intermediate_size,
        "flops": flops_for_gemm(seq_len, hidden_size, intermediate_size),
    }


def op_mlp_down(seq_len: int, hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: str):
    x = torch.randn((seq_len, intermediate_size), dtype=dtype, device=device)
    w = torch.randn((intermediate_size, hidden_size), dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return x @ w

    return run, {
        "m": seq_len,
        "k": intermediate_size,
        "n": hidden_size,
        "flops": flops_for_gemm(seq_len, intermediate_size, hidden_size),
    }


def op_attention_output_proj(seq_len: int, hidden_size: int, dtype: torch.dtype, device: str):
    x = torch.randn((seq_len, hidden_size), dtype=dtype, device=device)
    w = torch.randn((hidden_size, hidden_size), dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return x @ w

    return run, {
        "m": seq_len,
        "k": hidden_size,
        "n": hidden_size,
        "flops": flops_for_gemm(seq_len, hidden_size, hidden_size),
    }


def op_flash_attention(seq_len: int, hidden_size: int, num_heads: int, dtype: torch.dtype, device: str):
    head_dim = hidden_size // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn((1, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.randn((1, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((1, num_heads, seq_len, head_dim), dtype=dtype, device=device)

    def run() -> torch.Tensor:
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)

    return run, {
        "m": seq_len,
        "k": head_dim,
        "n": num_heads,
        "flops": flops_for_flash_attention(seq_len, hidden_size),
    }


def estimate_operator_flops(operator: str, seq_len: int, hidden_size: int, intermediate_size: int) -> int:
    if operator in {"mlp_up_gemm", "mlp_gate_gemm"}:
        return flops_for_gemm(seq_len, hidden_size, intermediate_size)
    if operator == "mlp_down_gemm":
        return flops_for_gemm(seq_len, intermediate_size, hidden_size)
    if operator == "attention_output_proj_gemm":
        return flops_for_gemm(seq_len, hidden_size, hidden_size)
    if operator == "flash_attention":
        return flops_for_flash_attention(seq_len, hidden_size)
    raise ValueError(f"Unknown operator: {operator}")


def build_operator(
    operator: str,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype: torch.dtype,
    device: str,
):
    if operator == "mlp_up_gemm":
        return op_mlp_up(seq_len, hidden_size, intermediate_size, dtype, device)
    if operator == "mlp_gate_gemm":
        return op_mlp_gate(seq_len, hidden_size, intermediate_size, dtype, device)
    if operator == "mlp_down_gemm":
        return op_mlp_down(seq_len, hidden_size, intermediate_size, dtype, device)
    if operator == "flash_attention":
        return op_flash_attention(seq_len, hidden_size, num_heads, dtype, device)
    if operator == "attention_output_proj_gemm":
        return op_attention_output_proj(seq_len, hidden_size, dtype, device)
    raise ValueError(f"Unknown operator: {operator}")


OPERATORS = [
    "mlp_up_gemm",
    "mlp_gate_gemm",
    "mlp_down_gemm",
    "flash_attention",
    "attention_output_proj_gemm",
]


def bench_single(
    operator: str,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    inner_loops: int,
) -> dict[str, float]:
    torch.mlu.set_device(0)
    run, meta = build_operator(operator, seq_len, hidden_size, intermediate_size, num_heads, dtype, "mlu:0")
    summary = summarize(benchmark_op(run, warmup, repeats, inner_loops))
    summary.update(meta)
    return summary


def dual_worker(
    rank: int,
    operator: str,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype_name: str,
    warmup: int,
    repeats: int,
    inner_loops: int,
    queue: mp.Queue,
) -> None:
    dtype = DTYPE_MAP[dtype_name]
    torch.mlu.set_device(rank)
    run, meta = build_operator(operator, seq_len, hidden_size, intermediate_size, num_heads, dtype, f"mlu:{rank}")
    summary = summarize(benchmark_op(run, warmup, repeats, inner_loops))
    summary.update({**meta, "rank": rank})
    queue.put(summary)


def bench_dual(
    operator: str,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    dtype_name: str,
    warmup: int,
    repeats: int,
    inner_loops: int,
) -> dict[str, float]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=dual_worker,
            args=(rank, operator, seq_len, hidden_size, intermediate_size, num_heads, dtype_name, warmup, repeats, inner_loops, queue),
        )
        for rank in (0, 1)
    ]
    for proc in processes:
        proc.start()
    results = [queue.get() for _ in processes]
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Dual-card benchmark process failed with exit code {proc.exitcode}.")
    chosen = max(results, key=lambda item: item["avg_ms"])
    return {
        "avg_ms": max(item["avg_ms"] for item in results),
        "min_ms": max(item["min_ms"] for item in results),
        "max_ms": max(item["max_ms"] for item in results),
        "std_ms": max(item["std_ms"] for item in results),
        "m": chosen["m"],
        "k": chosen["k"],
        "n": chosen["n"],
        "flops": chosen["flops"],
    }


def main() -> None:
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    seq_lengths = [int(item) for item in args.seq_lengths.split(",") if item.strip()]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator",
                "scale",
                "world_size",
                "seq_len",
                "m",
                "k",
                "n",
                "flops",
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
        for operator in OPERATORS:
            for seq_len in seq_lengths:
                base_flops = estimate_operator_flops(
                    operator,
                    seq_len,
                    args.hidden_size,
                    args.intermediate_size,
                )
                inner_loops = max(1, min(64, 274877906944 // max(base_flops, 1)))
                for scale, world_size in (("single_card", 1), ("single_node_dual_card", 2)):
                    if scale == "single_card":
                        summary = bench_single(
                            operator,
                            seq_len,
                            args.hidden_size,
                            args.intermediate_size,
                            args.num_heads,
                            dtype,
                            args.warmup,
                            args.repeats,
                            inner_loops,
                        )
                    else:
                        summary = bench_dual(
                            operator,
                            seq_len,
                            args.hidden_size,
                            args.intermediate_size,
                            args.num_heads,
                            args.dtype,
                            args.warmup,
                            args.repeats,
                            inner_loops,
                        )
                    writer.writerow(
                        {
                            "operator": operator,
                            "scale": scale,
                            "world_size": world_size,
                            "seq_len": seq_len,
                            "m": summary["m"],
                            "k": summary["k"],
                            "n": summary["n"],
                            "flops": int(summary["flops"]),
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
