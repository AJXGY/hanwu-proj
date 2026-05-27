from __future__ import annotations

import statistics
import time

import torch
import torch.distributed as dist

from mvp_backend import event, get_device_properties, synchronize
from mvp_graph import dtype_num_bytes
from mvp_types import ExecutionConfig, HardwareCalibration


def benchmark_linear_tflops(dtype: torch.dtype, device: torch.device) -> float:
    accelerator_kind = device.type
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
        synchronize(accelerator_kind)
        start = event(accelerator_kind, enable_timing=True)
        end = event(accelerator_kind, enable_timing=True)
        start.record()
        for _ in range(10):
            torch.nn.functional.linear(x, w)
        end.record()
        synchronize(accelerator_kind)
        elapsed_ms = start.elapsed_time(end) / 10.0
        flops = 2.0 * m * n * k
        scores.append(flops / (elapsed_ms / 1.0e3) / 1.0e12)
        del x, w
    return max(1.0, statistics.median(scores))


def benchmark_attention_tflops(dtype: torch.dtype, device: torch.device) -> float:
    accelerator_kind = device.type
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
        synchronize(accelerator_kind)
        start = event(accelerator_kind, enable_timing=True)
        end = event(accelerator_kind, enable_timing=True)
        start.record()
        for _ in range(20):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        end.record()
        synchronize(accelerator_kind)
        elapsed_ms = start.elapsed_time(end) / 20.0
        flops = 4.0 * batch * heads * q_len * kv_len * head_dim
        scores.append(flops / (elapsed_ms / 1.0e3) / 1.0e12)
        del q, k, v
    return max(1.0, statistics.median(scores))


def benchmark_memory_bandwidth_gbps(dtype: torch.dtype, device: torch.device) -> float:
    accelerator_kind = device.type
    elements = 16 * 1024 * 1024
    x = torch.randn((elements,), device=device, dtype=dtype)
    y = torch.randn((elements,), device=device, dtype=dtype)
    for _ in range(5):
        x + y
    synchronize(accelerator_kind)
    start = event(accelerator_kind, enable_timing=True)
    end = event(accelerator_kind, enable_timing=True)
    start.record()
    for _ in range(20):
        x + y
    end.record()
    synchronize(accelerator_kind)
    elapsed_ms = start.elapsed_time(end) / 20.0
    total_bytes = elements * dtype_num_bytes(dtype) * 3
    return max(1.0, total_bytes / (elapsed_ms / 1.0e3) / 1.0e9)


def benchmark_launch_overhead_ms(device: torch.device) -> float:
    accelerator_kind = device.type
    x = torch.ones((1,), device=device)
    iterations = 4000
    synchronize(accelerator_kind)
    start = event(accelerator_kind, enable_timing=True)
    end = event(accelerator_kind, enable_timing=True)
    start.record()
    for _ in range(iterations):
        x = x + 1
    end.record()
    synchronize(accelerator_kind)
    return max(0.001, start.elapsed_time(end) / iterations)


def benchmark_collective_link(
    execution: ExecutionConfig,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int = 2,
    repeat: int = 6,
) -> dict[str, object] | None:
    if execution.parallel_mode != "tp" or execution.tp_size <= 1:
        return None
    if not dist.is_initialized():
        return None

    accelerator_kind = device.type
    ring_factor = 2.0 * (execution.tp_size - 1) / execution.tp_size
    sizes_bytes = [256, 1024, 4096, 16384, 65536]
    rows: list[dict[str, object]] = []

    for num_bytes in sizes_bytes:
        numel = max((num_bytes + dtype_num_bytes(dtype) - 1) // dtype_num_bytes(dtype), 1)
        tensor = torch.ones((numel,), device=device, dtype=dtype)
        for _ in range(warmup):
            dist.barrier()
            synchronize(accelerator_kind)
            dist.all_reduce(tensor)
            synchronize(accelerator_kind)

        worst_case_samples: list[float] = []
        for _ in range(repeat):
            dist.barrier()
            synchronize(accelerator_kind)
            started = time.perf_counter()
            dist.all_reduce(tensor)
            synchronize(accelerator_kind)
            elapsed_ms = (time.perf_counter() - started) * 1.0e3
            gathered: list[float | None] = [None for _ in range(execution.world_size)]
            dist.all_gather_object(gathered, elapsed_ms)
            worst_case_samples.append(max(float(item or 0.0) for item in gathered))
        rows.append(
            {
                "bytes": int(num_bytes),
                "mean_ms": statistics.mean(worst_case_samples),
                "median_ms": statistics.median(worst_case_samples),
                "samples_ms": list(worst_case_samples),
            }
        )

    x_values = [ring_factor * float(row["bytes"]) for row in rows]
    y_values = [float(row["median_ms"]) for row in rows]
    count = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xx = sum(value * value for value in x_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    denominator = count * sum_xx - sum_x * sum_x
    if denominator <= 0:
        slope = 1.0e-6 / 1.0
        intercept = max(min(y_values), 0.001)
    else:
        slope = max((count * sum_xy - sum_x * sum_y) / denominator, 1.0e-12)
        intercept = max((sum_y - slope * sum_x) / count, 0.001)
    bandwidth_gbps = max(0.01, min(1.0e-6 / slope, 500.0))
    return {
        "bandwidth_gbps": bandwidth_gbps,
        "latency_ms": intercept,
        "samples": rows,
    }


def build_calibration(dtype: torch.dtype, device: torch.device) -> HardwareCalibration:
    gemm_tflops = benchmark_linear_tflops(dtype, device)
    attention_tflops = benchmark_attention_tflops(dtype, device)
    memory_bandwidth_gbps = benchmark_memory_bandwidth_gbps(dtype, device)
    launch_overhead_ms = benchmark_launch_overhead_ms(device)
    props = get_device_properties(device.type, device)
    return HardwareCalibration(
        accelerator_kind=device.type,
        device_name=props.name,
        device_index=device.index or 0,
        gemm_tflops=gemm_tflops,
        attention_tflops=attention_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        launch_overhead_ms=launch_overhead_ms,
    )
