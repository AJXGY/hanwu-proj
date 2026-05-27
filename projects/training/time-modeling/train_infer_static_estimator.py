from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from mvp_train_types import ModelArchitecture, TrainCalibration, TrainConfig
from mvp_train_unified_estimator import estimate_train_step_with_tp as _estimate_step


def _read_scalar(text: str, key: str, default: float) -> float:
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line.startswith(f"{key}:"):
            continue
        value = line.split(":", 1)[1].strip()
        if value in {"", "null", "None"}:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    return default


def load_train_infer_calibration(config_path: str | Path | None) -> TrainCalibration:
    defaults = TrainCalibration(
        device_name="mlu",
        device_index=0,
        gemm_tflops=2.9,
        attention_tflops=2.9,
        memory_bandwidth_gbps=450.0,
        launch_overhead_ms=0.15,
        backward_compute_scale=0.0,
        optimizer_scale_factor=1.4,
        effective_tflops_scale=0.9,
        backward_efficiency_scale=0.07,
        kernel_overhead_factor=0.15,
        forward_parallelism_factor=0.76,
        parallelism_factor=0.25,
        overhead_scale=0.015476190476190477,
        has_nvlink=True,
        overlap_ratio=0.9,
        tp_backward_efficiency=0.2,
        tp_forward_efficiency=0.166,
        gradient_allreduce_tflops=100.0,
        tp_config={
            "communication": {
                "nvlink_bandwidth_gbps": 450.0,
                "nvlink_latency_ms": 0.3,
                "pcie_bandwidth_gbps": 32.0,
                "pcie_latency_ms": 5.0,
                "overlap_ratio": 0.9,
                "optimizer_efficiency": 0.4524052358756343,
            }
        },
    )
    if config_path is None:
        return defaults
    path = Path(config_path)
    if not path.exists():
        return defaults
    text = path.read_text(encoding="utf-8")
    tp_config = {
        "communication": {
            "nvlink_bandwidth_gbps": _read_scalar(text, "local_bandwidth_gbps", 450.0),
            "nvlink_latency_ms": _read_scalar(text, "local_latency_ms", 0.3),
            "pcie_bandwidth_gbps": _read_scalar(text, "pcie_bandwidth_gbps", 32.0),
            "pcie_latency_ms": _read_scalar(text, "pcie_latency_ms", 5.0),
            "overlap_ratio": _read_scalar(text, "overlap_ratio", 0.9),
            "optimizer_efficiency": _read_scalar(
                text, "optimizer_efficiency", 0.4524052358756343
            ),
        }
    }
    calibration = TrainCalibration(
        device_name="mlu",
        device_index=0,
        gemm_tflops=_read_scalar(text, "gemm_tflops", defaults.gemm_tflops),
        attention_tflops=_read_scalar(text, "attention_tflops", defaults.attention_tflops),
        memory_bandwidth_gbps=_read_scalar(
            text, "memory_bandwidth_gbps", defaults.memory_bandwidth_gbps
        ),
        launch_overhead_ms=_read_scalar(
            text, "launch_overhead_ms", defaults.launch_overhead_ms
        ),
        backward_compute_scale=_read_scalar(
            text, "backward_compute_scale", defaults.backward_compute_scale
        ),
        optimizer_scale_factor=_read_scalar(
            text, "optimizer_scale_factor", defaults.optimizer_scale_factor
        ),
        effective_tflops_scale=_read_scalar(
            text, "effective_tflops_scale", defaults.effective_tflops_scale
        ),
        backward_efficiency_scale=_read_scalar(
            text, "backward_efficiency_scale", defaults.backward_efficiency_scale
        ),
        kernel_overhead_factor=_read_scalar(
            text, "kernel_overhead_factor", defaults.kernel_overhead_factor
        ),
        forward_parallelism_factor=_read_scalar(
            text, "forward_parallelism_factor", defaults.forward_parallelism_factor
        ),
        parallelism_factor=_read_scalar(
            text, "backward_parallelism_factor", defaults.parallelism_factor
        ),
        overhead_scale=_read_scalar(text, "overhead_scale", defaults.overhead_scale),
        has_nvlink=True,
        overlap_ratio=tp_config["communication"]["overlap_ratio"],
        tp_backward_efficiency=_read_scalar(
            text, "tp_backward_efficiency", defaults.tp_backward_efficiency
        ),
        tp_forward_efficiency=_read_scalar(
            text, "tp_forward_efficiency", defaults.tp_forward_efficiency
        ),
        gradient_allreduce_tflops=_read_scalar(
            text, "gradient_allreduce_tflops", defaults.gradient_allreduce_tflops
        ),
        tp_config=tp_config,
    )
    calibration.single_backward_compute_scale = _read_scalar(
        text, "single_backward_compute_scale", 14.25
    )
    return calibration


def architecture_from_config(
    model_config: Any,
    optimizer_param_count: int | None = None,
) -> ModelArchitecture:
    return ModelArchitecture(
        num_layers=int(getattr(model_config, "num_hidden_layers", 0)),
        hidden_size=int(getattr(model_config, "hidden_size", 0)),
        num_attention_heads=int(getattr(model_config, "num_attention_heads", 0)),
        vocab_size=int(getattr(model_config, "vocab_size", 0)),
        intermediate_size=int(
            getattr(model_config, "intermediate_size", 0)
            or 4 * int(getattr(model_config, "hidden_size", 0))
        ),
        model_type=str(getattr(model_config, "model_type", "unknown")),
        adapter_param_count=optimizer_param_count,
    )


def estimate_train_step_with_tp(
    batch_size: int,
    seq_len: int,
    arch: ModelArchitecture,
    calibration: TrainCalibration,
    tp_size: int,
    gradient_accumulation_steps: int,
    training_mode: str,
    ddp_enabled: bool = False,
) -> dict[str, Any]:
    config = TrainConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ddp_enabled=ddp_enabled,
        tp_size=tp_size,
    )
    if tp_size <= 1 and hasattr(calibration, "single_backward_compute_scale"):
        calibration = TrainCalibration(
            **{
                **asdict(calibration),
                "backward_compute_scale": getattr(
                    calibration, "single_backward_compute_scale"
                ),
            }
        )
    step = _estimate_step(
        batch_size=batch_size,
        seq_len=seq_len,
        arch=arch,
        calibration=calibration,
        config=config,
    )
    backward_compute_ms = (
        step.backward_summary.compute_time_ms if step.backward_summary else step.backward_time_ms
    )
    backward_comm_ms = step.backward_summary.comm_time_ms if step.backward_summary else 0.0
    backward_non_comm_ms = max(step.backward_time_ms - backward_comm_ms, backward_compute_ms)
    return {
        "source": "train_infer_static",
        "implementation": "mvp_train_unified_estimator",
        "total_time_ms": step.total_time_ms,
        "forward_ms": step.forward_time_ms,
        "backward_compute_ms": backward_non_comm_ms,
        "backward_raw_compute_ms": backward_compute_ms,
        "backward_total_ms": step.backward_time_ms,
        "backward_comm_ms": backward_comm_ms,
        "optimizer_ms": step.optimizer_time_ms,
        "forward": asdict(step.forward_summary) if step.forward_summary else None,
        "backward": asdict(step.backward_summary) if step.backward_summary else None,
        "optimizer": asdict(step.optimizer_summary) if step.optimizer_summary else None,
        "tp_info": {
            "tp_enabled": tp_size > 1,
            "tp_size": tp_size,
            "ddp_enabled": ddp_enabled,
            "overlap_ratio": calibration.overlap_ratio,
        },
        "architecture": asdict(arch),
        "calibration": asdict(calibration),
        "training_mode": training_mode,
    }
