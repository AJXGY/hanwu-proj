from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mvp_backend import event, get_device_properties, synchronize


DEFAULT_DTYPE = torch.float16
DEFAULT_DEVICE = "cuda:0"


@dataclass
class Calibration:
    device_name: str
    dtype: str
    bytes_per_elem: int
    gemm_tflops: float
    attention_tflops: float
    memory_bandwidth_gbps: float
    launch_overhead_us: float


def dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def ensure_parent(path_text: str) -> None:
    Path(path_text).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def benchmark_event_ms(fn, *, accelerator_kind: str, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(accelerator_kind)
    samples = []
    for _ in range(repeat):
        start = event(accelerator_kind, enable_timing=True)
        end = event(accelerator_kind, enable_timing=True)
        start.record()
        fn()
        end.record()
        synchronize(accelerator_kind)
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def benchmark_gemm_tflops(dtype: torch.dtype, device: torch.device) -> float:
    accelerator_kind = device.type
    scores = []
    for m, n, k in [(2048, 2048, 2048), (4096, 2048, 4096), (4096, 4096, 2048)]:
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((k, n), device=device, dtype=dtype)
        elapsed_ms = benchmark_event_ms(
            lambda: torch.matmul(a, b),
            accelerator_kind=accelerator_kind,
            warmup=3,
            repeat=10,
        )
        flops = 2.0 * m * n * k
        scores.append(flops / (elapsed_ms / 1.0e3) / 1.0e12)
    return max(1.0, statistics.median(scores))


def benchmark_attention_tflops(dtype: torch.dtype, device: torch.device) -> float:
    accelerator_kind = device.type
    scores = []
    for bsz, heads, q_len, kv_len, head_dim in [
        (1, 32, 64, 64, 128),
        (1, 32, 128, 128, 128),
        (1, 32, 256, 256, 128),
    ]:
        q = torch.randn((bsz, heads, q_len, head_dim), device=device, dtype=dtype)
        k = torch.randn((bsz, heads, kv_len, head_dim), device=device, dtype=dtype)
        v = torch.randn((bsz, heads, kv_len, head_dim), device=device, dtype=dtype)
        elapsed_ms = benchmark_event_ms(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            accelerator_kind=accelerator_kind,
            warmup=3,
            repeat=10,
        )
        flops = 4.0 * bsz * heads * q_len * kv_len * head_dim
        scores.append(flops / (elapsed_ms / 1.0e3) / 1.0e12)
    return max(1.0, statistics.median(scores))


def benchmark_memory_bandwidth(dtype: torch.dtype, device: torch.device) -> float:
    accelerator_kind = device.type
    elements = 32 * 1024 * 1024
    x = torch.randn((elements,), device=device, dtype=dtype)
    y = torch.randn((elements,), device=device, dtype=dtype)
    elapsed_ms = benchmark_event_ms(
        lambda: x + y,
        accelerator_kind=accelerator_kind,
        warmup=5,
        repeat=20,
    )
    total_bytes = elements * dtype_num_bytes(dtype) * 3
    return max(1.0, total_bytes / (elapsed_ms / 1.0e3) / 1.0e9)


def benchmark_launch_overhead(dtype: torch.dtype, device: torch.device) -> float:
    accelerator_kind = device.type
    x = torch.ones((1,), device=device, dtype=dtype)
    iterations = 5000

    def step() -> None:
        nonlocal x
        for _ in range(iterations):
            x = x + 1

    elapsed_ms = benchmark_event_ms(
        step,
        accelerator_kind=accelerator_kind,
        warmup=1,
        repeat=5,
    )
    return max(0.1, elapsed_ms * 1.0e3 / iterations)


def build_calibration(dtype: torch.dtype, device: torch.device) -> Calibration:
    props = get_device_properties(device.type, device)
    return Calibration(
        device_name=props.name,
        dtype=str(dtype).replace("torch.", ""),
        bytes_per_elem=dtype_num_bytes(dtype),
        gemm_tflops=benchmark_gemm_tflops(dtype, device),
        attention_tflops=benchmark_attention_tflops(dtype, device),
        memory_bandwidth_gbps=benchmark_memory_bandwidth(dtype, device),
        launch_overhead_us=benchmark_launch_overhead(dtype, device),
    )


def relative_error_pct(estimate: float, measured: float) -> float:
    if measured == 0:
        return 0.0
    return abs(estimate - measured) / measured * 100.0


def shape_signature(shape: dict[str, int]) -> str:
    return json.dumps(shape, sort_keys=True, separators=(",", ":"))


def calibration_flag(index: int) -> str:
    return "calibration" if index in {0, 2} else "validation"


def cambricon_pointwise_role(index: int) -> str:
    return "calibration" if index in {0, 1, 3, 5} else "validation"


def compute_cases() -> list[dict[str, Any]]:
    cases = []
    for idx, dims in enumerate(
        [
            {"M": 512, "K": 4096, "N": 4096},
            {"M": 2048, "K": 4096, "N": 4096},
            {"M": 4096, "K": 4096, "N": 4096},
            {"M": 8192, "K": 4096, "N": 4096},
        ]
    ):
        cases.append(
            {"operator": "aten::mm", "shape": dims, "point_role": calibration_flag(idx)}
        )
    for idx, dims in enumerate(
        [
            {"B": 32, "M": 2, "K": 128, "N": 128},
            {"B": 32, "M": 8, "K": 128, "N": 128},
            {"B": 32, "M": 32, "K": 128, "N": 128},
            {"B": 32, "M": 128, "K": 128, "N": 128},
        ]
    ):
        cases.append(
            {
                "operator": "aten::matmul",
                "shape": dims,
                "point_role": calibration_flag(idx),
            }
        )
    for idx, dims in enumerate(
        [
            {"B": 4, "S": 512, "H": 4096},
            {"B": 4, "S": 1024, "H": 4096},
            {"B": 4, "S": 1536, "H": 4096},
            {"B": 4, "S": 2048, "H": 4096},
            {"B": 4, "S": 3072, "H": 4096},
            {"B": 4, "S": 4096, "H": 4096},
        ]
    ):
        cases.append(
            {
                "operator": "aten::add",
                "shape": dims,
                "point_role": cambricon_pointwise_role(idx),
            }
        )
    for dims, role in [
        ({"B": 4, "S": 512, "H": 4096}, "calibration"),
        ({"B": 4, "S": 1024, "H": 4096}, "calibration"),
        ({"B": 4, "S": 1280, "H": 4096}, "validation"),
        ({"B": 4, "S": 1536, "H": 4096}, "calibration"),
        ({"B": 4, "S": 2048, "H": 4096}, "calibration"),
        ({"B": 4, "S": 3072, "H": 4096}, "calibration"),
        ({"B": 4, "S": 3584, "H": 4096}, "validation"),
        ({"B": 4, "S": 4096, "H": 4096}, "calibration"),
    ]:
        cases.append({"operator": "aten::mul", "shape": dims, "point_role": role})
    for dims, role in [
        ({"B": 4, "S": 512, "H": 4096}, "calibration"),
        ({"B": 4, "S": 1024, "H": 4096}, "calibration"),
        ({"B": 4, "S": 1280, "H": 4096}, "validation"),
        ({"B": 4, "S": 1536, "H": 4096}, "calibration"),
        ({"B": 4, "S": 2048, "H": 4096}, "calibration"),
        ({"B": 4, "S": 3072, "H": 4096}, "calibration"),
        ({"B": 4, "S": 3584, "H": 4096}, "validation"),
        ({"B": 4, "S": 4096, "H": 4096}, "calibration"),
    ]:
        cases.append({"operator": "aten::pow", "shape": dims, "point_role": role})
    for idx, dims in enumerate(
        [
            {"B": 4, "S": 512, "H": 4096},
            {"B": 4, "S": 1024, "H": 4096},
            {"B": 4, "S": 1536, "H": 4096},
            {"B": 4, "S": 2048, "H": 4096},
            {"B": 4, "S": 3072, "H": 4096},
            {"B": 4, "S": 4096, "H": 4096},
        ]
    ):
        cases.append(
            {
                "operator": "aten::rsqrt",
                "shape": dims,
                "point_role": cambricon_pointwise_role(idx),
            }
        )
    for dims, role in [
        ({"B": 4, "S": 512, "H": 4096}, "calibration"),
        ({"B": 4, "S": 1024, "H": 4096}, "calibration"),
        ({"B": 4, "S": 1280, "H": 4096}, "validation"),
        ({"B": 4, "S": 2048, "H": 4096}, "calibration"),
        ({"B": 4, "S": 2560, "H": 4096}, "validation"),
        ({"B": 4, "S": 3072, "H": 4096}, "calibration"),
        ({"B": 4, "S": 4096, "H": 4096}, "calibration"),
    ]:
        cases.append({"operator": "aten::mean", "shape": dims, "point_role": role})
    for idx, dims in enumerate(
        [
            {"B": 4, "H": 32, "S": 1024},
            {"B": 4, "H": 32, "S": 2048},
            {"B": 4, "H": 32, "S": 4096},
            {"B": 4, "H": 32, "S": 8192},
        ]
    ):
        cases.append(
            {
                "operator": "aten::_softmax",
                "shape": dims,
                "point_role": calibration_flag(idx),
            }
        )
    return cases


def benchmark_case(
    operator: str, shape: dict[str, int], dtype: torch.dtype, device: torch.device
) -> float:
    accelerator_kind = device.type
    if operator == "aten::mm":
        a = torch.randn((shape["M"], shape["K"]), device=device, dtype=dtype)
        b = torch.randn((shape["K"], shape["N"]), device=device, dtype=dtype)
        return (
            benchmark_event_ms(
                lambda: torch.matmul(a, b),
                accelerator_kind=accelerator_kind,
                warmup=3,
                repeat=10,
            )
            * 1.0e3
        )
    if operator == "aten::matmul":
        q = torch.randn(
            (shape["B"], shape["M"], shape["K"]), device=device, dtype=dtype
        )
        k = torch.randn(
            (shape["B"], shape["K"], shape["N"]), device=device, dtype=dtype
        )
        return (
            benchmark_event_ms(
                lambda: torch.matmul(q, k),
                accelerator_kind=accelerator_kind,
                warmup=3,
                repeat=10,
            )
            * 1.0e3
        )
    if operator == "aten::add":
        x = torch.randn(
            (shape["B"], shape["S"], shape["H"]), device=device, dtype=dtype
        )
        y = torch.randn(
            (shape["B"], shape["S"], shape["H"]), device=device, dtype=dtype
        )
        return (
            benchmark_event_ms(
                lambda: x + y,
                accelerator_kind=accelerator_kind,
                warmup=3,
                repeat=10,
            )
            * 1.0e3
        )
    if operator == "aten::mul":
        x = torch.randn(
            (shape["B"], shape["S"], shape["H"]), device=device, dtype=dtype
        )
        return (
            benchmark_event_ms(
                lambda: x * x,
                accelerator_kind=accelerator_kind,
                warmup=3,
                repeat=10,
            )
            * 1.0e3
        )
    if operator == "aten::pow":
        x = torch.randn(
            (shape["B"], shape["S"], shape["H"]), device=device, dtype=dtype
        )
        return (
            benchmark_event_ms(
                lambda: x.pow(2),
                accelerator_kind=accelerator_kind,
                warmup=3,
                repeat=10,
            )
            * 1.0e3
        )
    if operator == "aten::rsqrt":
        x = torch.randn(
            (shape["B"], shape["S"], shape["H"]), device=device, dtype=dtype
        )
        return (
            benchmark_event_ms(
                lambda: torch.rsqrt(x.abs() + 1.0e-3),
                accelerator_kind=accelerator_kind,
                warmup=3,
                repeat=10,
            )
            * 1.0e3
        )
    if operator == "aten::_softmax":
        x = torch.randn(
            (shape["B"], shape["H"], shape["S"]), device=device, dtype=dtype
        )
        return (
            benchmark_event_ms(
                lambda: F.softmax(x, dim=-1),
                accelerator_kind=accelerator_kind,
                warmup=3,
                repeat=10,
            )
            * 1.0e3
        )
    if operator == "aten::mean":
        x = torch.randn(
            (shape["B"], shape["S"], shape["H"]), device=device, dtype=dtype
        )
        return (
            benchmark_event_ms(
                lambda: x.mean(dim=-1),
                accelerator_kind=accelerator_kind,
                warmup=3,
                repeat=10,
            )
            * 1.0e3
        )
    raise ValueError(f"Unsupported operator: {operator}")


def predict_us(model: dict[str, Any], operator: str, shape: dict[str, int]) -> float:
    curve = (model.get("operator_curves") or {}).get(operator)
    if curve:
        interpolated = interpolate_curve_us(curve, shape_work_units(operator, shape))
        if interpolated is not None:
            return interpolated
    calib = model["calibration"]
    bytes_per_elem = int(calib["bytes_per_elem"])
    launch_overhead_us = float(calib["launch_overhead_us"])
    memory_bandwidth = float(calib["memory_bandwidth_gbps"]) * 1.0e9
    if operator == "aten::mm":
        m, k, n = shape["M"], shape["K"], shape["N"]
        flops = 2.0 * m * n * k
        bytes_acc = (m * k + k * n + m * n) * bytes_per_elem
        compute_us = flops / (float(calib["gemm_tflops"]) * 1.0e12) * 1.0e6
        memory_us = bytes_acc / (memory_bandwidth * 0.9) * 1.0e6
        estimate_us = max(compute_us, memory_us) + launch_overhead_us
    elif operator == "aten::matmul":
        bsz, m, k, n = shape["B"], shape["M"], shape["K"], shape["N"]
        flops = 2.0 * bsz * m * n * k
        bytes_acc = bsz * (m * k + k * n + m * n) * bytes_per_elem
        compute_us = flops / (float(calib["gemm_tflops"]) * 1.0e12) * 1.0e6
        memory_us = bytes_acc / (memory_bandwidth * 0.9) * 1.0e6
        estimate_us = max(compute_us, memory_us) + launch_overhead_us
    elif operator in {"aten::add", "aten::mul", "aten::pow", "aten::rsqrt"}:
        numel = shape["B"] * shape["S"] * shape["H"]
        flops = (
            5.0 * numel if operator in {"aten::pow", "aten::rsqrt"} else float(numel)
        )
        bytes_acc = 3.0 * numel * bytes_per_elem
        compute_us = flops / (float(calib["attention_tflops"]) * 1.0e12) * 1.0e6
        memory_us = bytes_acc / (memory_bandwidth * 0.8) * 1.0e6
        estimate_us = max(compute_us, memory_us) + launch_overhead_us
    elif operator == "aten::_softmax":
        numel = shape["B"] * shape["H"] * shape["S"]
        flops = 5.0 * numel
        bytes_acc = 3.0 * numel * bytes_per_elem
        compute_us = flops / (float(calib["attention_tflops"]) * 1.0e12) * 1.0e6
        memory_us = bytes_acc / (memory_bandwidth * 0.8) * 1.0e6
        estimate_us = max(compute_us, memory_us) + launch_overhead_us
    elif operator == "aten::mean":
        numel = shape["B"] * shape["S"] * shape["H"]
        flops = float(numel)
        bytes_acc = 2.0 * numel * bytes_per_elem
        compute_us = flops / (float(calib["attention_tflops"]) * 1.0e12) * 1.0e6
        memory_us = bytes_acc / (memory_bandwidth * 0.9) * 1.0e6
        estimate_us = max(compute_us, memory_us) + launch_overhead_us
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    transform = (model.get("operator_transforms") or {}).get(operator)
    if transform:
        return estimate_us * float(transform.get("slope", 1.0)) + float(
            transform.get("intercept_us", 0.0)
        )
    return estimate_us * float((model.get("operator_scales") or {}).get(operator, 1.0))


def base_model_with_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "compute_operator_model",
        "calibration": calibration,
        "operator_scales": {},
        "operator_transforms": {},
        "operator_curves": {},
    }


def shape_work_units(operator: str, shape: dict[str, int]) -> float:
    if operator == "aten::mm":
        return float(shape["M"] * shape["K"] * shape["N"])
    if operator == "aten::matmul":
        return float(shape["B"] * shape["M"] * shape["K"] * shape["N"])
    if operator in {"aten::add", "aten::mul", "aten::pow", "aten::rsqrt", "aten::mean"}:
        return float(shape["B"] * shape["S"] * shape["H"])
    if operator == "aten::_softmax":
        return float(shape["B"] * shape["H"] * shape["S"])
    raise ValueError(f"Unsupported operator: {operator}")


def interpolate_curve_us(points: list[dict[str, float]], work: float) -> float | None:
    if not points:
        return None
    ordered = sorted(points, key=lambda item: float(item["work"]))
    if len(ordered) == 1:
        return float(ordered[0]["real_us"])
    if work <= float(ordered[0]["work"]):
        left = ordered[0]
        right = ordered[1]
    elif work >= float(ordered[-1]["work"]):
        left = ordered[-2]
        right = ordered[-1]
    else:
        left = ordered[0]
        right = ordered[-1]
        for start, end in zip(ordered, ordered[1:]):
            if float(start["work"]) <= work <= float(end["work"]):
                left = start
                right = end
                break
    left_work = float(left["work"])
    right_work = float(right["work"])
    left_us = float(left["real_us"])
    right_us = float(right["real_us"])
    if right_work == left_work:
        return left_us
    ratio = (work - left_work) / (right_work - left_work)
    return left_us + (right_us - left_us) * ratio


def fit_linear_transform(xs: list[float], ys: list[float]) -> dict[str, float]:
    if len(xs) < 2:
        return {"slope": 1.0, "intercept_us": 0.0}
    count = float(len(xs))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(item * item for item in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denominator = count * sum_xx - sum_x * sum_x
    if denominator == 0:
        return {"slope": 1.0, "intercept_us": 0.0}
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return {"slope": slope, "intercept_us": intercept}


def write_csv(
    path_text: str, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    ensure_parent(path_text)
    with open(path_text, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path_text: str) -> list[dict[str, str]]:
    with open(path_text, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def command_benchmark(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    calibration = build_calibration(dtype, device)
    rows = []
    for item in compute_cases():
        measured_us = benchmark_case(item["operator"], item["shape"], dtype, device)
        rows.append(
            {
                "operator": item["operator"],
                "shape_signature": shape_signature(item["shape"]),
                "shape_json": json.dumps(item["shape"], sort_keys=True),
                "real_us": f"{measured_us:.4f}",
                "point_role": item["point_role"],
            }
        )
    write_csv(
        args.output,
        rows,
        ["operator", "shape_signature", "shape_json", "real_us", "point_role"],
    )
    model = {"kind": "compute_operator_model", "calibration": asdict(calibration)}
    ensure_parent(args.calibration_output)
    Path(args.calibration_output).write_text(
        json.dumps(model, indent=2), encoding="utf-8"
    )


def command_build(args: argparse.Namespace) -> None:
    rows = read_csv(args.input)
    if not rows:
        raise ValueError("benchmark input is empty")
    calibration_path = Path(args.calibration_input)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))[
            "calibration"
        ]
    else:
        device = torch.device(args.device)
        dtype = dtype_from_name(args.dtype)
        calibration = asdict(build_calibration(dtype, device))
    model = base_model_with_calibration(calibration)
    model["operators"] = sorted({row["operator"] for row in rows})
    operator_scales: dict[str, float] = {}
    operator_transforms: dict[str, dict[str, float]] = {}
    operator_curves: dict[str, list[dict[str, float]]] = {}
    for operator in model["operators"]:
        calibration_rows = [
            row
            for row in rows
            if row["operator"] == operator and row["point_role"] == "calibration"
        ]
        scale_samples = []
        base_samples = []
        real_samples = []
        for row in calibration_rows:
            shape = json.loads(row["shape_json"])
            base_estimate = predict_us(
                base_model_with_calibration(calibration),
                operator,
                shape,
            )
            if base_estimate > 0:
                scale_samples.append(float(row["real_us"]) / base_estimate)
                base_samples.append(base_estimate)
                real_samples.append(float(row["real_us"]))
            operator_curves.setdefault(operator, []).append(
                {
                    "work": shape_work_units(operator, shape),
                    "real_us": float(row["real_us"]),
                }
            )
        operator_scales[operator] = (
            statistics.median(scale_samples) if scale_samples else 1.0
        )
        operator_transforms[operator] = fit_linear_transform(base_samples, real_samples)
    model["operator_scales"] = operator_scales
    model["operator_transforms"] = operator_transforms
    model["operator_curves"] = operator_curves
    ensure_parent(args.model_output)
    Path(args.model_output).write_text(json.dumps(model, indent=2), encoding="utf-8")


def command_evaluate(args: argparse.Namespace) -> None:
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    input_rows = read_csv(args.input)
    strict_rows = []
    grouped: dict[str, list[float]] = {}
    for row in input_rows:
        operator = row["operator"]
        shape = json.loads(row["shape_json"])
        real_us = float(row["real_us"])
        sim_us = predict_us(model, operator, shape)
        error_pct = relative_error_pct(sim_us, real_us)
        strict_rows.append(
            {
                "operator": operator,
                "shape_signature": row["shape_signature"],
                "shape_json": row["shape_json"],
                "real_us": f"{real_us:.4f}",
                "sim_us": f"{sim_us:.4f}",
                "error_pct": f"{error_pct:.4f}",
                "point_role": row["point_role"],
            }
        )
        if row["point_role"] == "validation":
            grouped.setdefault(operator, []).append(error_pct)
    report_rows = []
    for operator, errors in sorted(grouped.items()):
        report_rows.append(
            {
                "operator": operator,
                "validation_points": len(errors),
                "avg_error_pct": f"{statistics.mean(errors):.4f}",
                "max_error_pct": f"{max(errors):.4f}",
            }
        )
    write_csv(
        args.summary_output,
        strict_rows,
        [
            "operator",
            "shape_signature",
            "shape_json",
            "real_us",
            "sim_us",
            "error_pct",
            "point_role",
        ],
    )
    write_csv(
        args.report_output,
        report_rows,
        ["operator", "validation_points", "avg_error_pct", "max_error_pct"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified compute operator tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark compute operators"
    )
    benchmark_parser.add_argument("--device", default=DEFAULT_DEVICE)
    benchmark_parser.add_argument("--dtype", default="fp16")
    benchmark_parser.add_argument("--output", default="results/raw/compute_bench.csv")
    benchmark_parser.add_argument(
        "--calibration-output", default="results/raw/compute_calibration.json"
    )
    benchmark_parser.set_defaults(func=command_benchmark)

    build_parser_cmd = subparsers.add_parser("build", help="Build compute model")
    build_parser_cmd.add_argument("--input", default="results/raw/compute_bench.csv")
    build_parser_cmd.add_argument(
        "--calibration-input", default="results/raw/compute_calibration.json"
    )
    build_parser_cmd.add_argument("--device", default=DEFAULT_DEVICE)
    build_parser_cmd.add_argument("--dtype", default="fp16")
    build_parser_cmd.add_argument(
        "--model-output", default="results/processed/compute_space_model.json"
    )
    build_parser_cmd.set_defaults(func=command_build)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate compute model")
    eval_parser.add_argument(
        "--model", default="results/processed/compute_space_model.json"
    )
    eval_parser.add_argument("--input", default="results/raw/compute_bench.csv")
    eval_parser.add_argument(
        "--summary-output",
        default="results/processed/compute_model_validation_strict.csv",
    )
    eval_parser.add_argument(
        "--report-output",
        default="results/processed/compute_model_validation_report.csv",
    )
    eval_parser.set_defaults(func=command_evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
