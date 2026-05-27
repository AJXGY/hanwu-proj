from __future__ import annotations

import statistics

import torch

from mvp_graph import dtype_num_bytes
from mvp_types import HardwareCalibration


def benchmark_linear_tflops(dtype: torch.dtype, device: torch.device) -> float:
    shapes = [
        (2048, 2048, 2048),
        (2048, 8192, 2048),
        (4096, 2048, 2048),
    ]
    scores = []
    for m, n, k in shapes:
        x = torch.randn((m, k), device=device, dtype=dtype)
        w = torch.randn((n, k), device=device, dtype=dtype)
        for _ in range(3):
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            torch.nn.functional.linear(x, w)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 10.0
        flops = 2.0 * m * n * k
        scores.append(flops / (elapsed_ms / 1.0e3) / 1.0e12)
        del x, w
    return max(1.0, statistics.median(scores))


def benchmark_attention_tflops(dtype: torch.dtype, device: torch.device) -> float:
    cases = [
        (1, 32, 64, 64, 64),
        (1, 32, 128, 128, 64),
        (1, 32, 1, 128, 64),
        (1, 32, 1, 256, 64),
    ]
    scores = []
    for batch, heads, q_len, kv_len, head_dim in cases:
        q = torch.randn((batch, heads, q_len, head_dim), device=device, dtype=dtype)
        k = torch.randn((batch, heads, kv_len, head_dim), device=device, dtype=dtype)
        v = torch.randn((batch, heads, kv_len, head_dim), device=device, dtype=dtype)
        for _ in range(3):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 20.0
        flops = 4.0 * batch * heads * q_len * kv_len * head_dim
        scores.append(flops / (elapsed_ms / 1.0e3) / 1.0e12)
        del q, k, v
    return max(1.0, statistics.median(scores))


def benchmark_memory_bandwidth_gbps(dtype: torch.dtype, device: torch.device) -> float:
    elements = 16 * 1024 * 1024
    x = torch.randn((elements,), device=device, dtype=dtype)
    y = torch.randn((elements,), device=device, dtype=dtype)
    for _ in range(5):
        x + y
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        x + y
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / 20.0
    total_bytes = elements * dtype_num_bytes(dtype) * 3
    return max(1.0, total_bytes / (elapsed_ms / 1.0e3) / 1.0e9)


def benchmark_launch_overhead_ms(device: torch.device) -> float:
    x = torch.ones((1,), device=device)
    iterations = 4000
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        x = x + 1
    end.record()
    torch.cuda.synchronize()
    return max(0.001, start.elapsed_time(end) / iterations)


def build_calibration(dtype: torch.dtype, device: torch.device) -> HardwareCalibration:
    gemm_tflops = benchmark_linear_tflops(dtype, device)
    attention_tflops = benchmark_attention_tflops(dtype, device)
    memory_bandwidth_gbps = benchmark_memory_bandwidth_gbps(dtype, device)
    launch_overhead_ms = benchmark_launch_overhead_ms(device)
    props = torch.cuda.get_device_properties(device)
    return HardwareCalibration(
        device_name=props.name,
        device_index=device.index or 0,
        gemm_tflops=gemm_tflops,
        attention_tflops=attention_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        launch_overhead_ms=launch_overhead_ms,
    )
