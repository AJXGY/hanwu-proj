"""
Unified Training TP Estimator

This module integrates all four TP mission modules into a unified estimation framework:

- Mission 1: Backward graph extraction (mvp_backward_graph.py, mvp_train_graph.py)
- Mission 2: Backward communication model (mvp_backward_comm.py)
- Mission 3: Forward TP estimation (mvp_train_tp_estimator.py)
- Mission 4: Optimizer TP estimation (mvp_optimizer_tp_estimator.py)

Usage:
    from mvp_train_unified_estimator import estimate_train_step_with_tp

    result = estimate_train_step_with_tp(
        batch_size=1,
        seq_len=512,
        arch=model_arch,
        calibration=calibration,
        config=train_config,
    )
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

# Import mission modules
from mvp_backward_comm import (
    BackwardCommEstimate,
    CommCalibration,
    create_train_calibration_with_comm_params,
    estimate_backward_comm_simple,
    estimate_backward_comm_time,
    estimate_backward_with_comm,
)
from mvp_backward_graph import (
    BackwardGraphInfo,
    GradientInfo,
    extract_backward_graph,
    get_gradient_summary,
)
from mvp_optimizer_tp_estimator import (
    ADAM_PARAM_BYTES,
    compute_effective_tp_scale_for_optimizer,
    estimate_optimizer_time_latency_bandwidth,
    estimate_optimizer_tp_overhead,
    get_optimizer_efficiency,
)
from mvp_train_graph import TrainingGraphs, extract_training_graphs
from mvp_train_tp_calibration import TPTrainCalibration, from_calibration, get_default_calibration
from mvp_train_tp_estimator import (
    analyze_tp_scale_breakdown,
    compute_effective_tp_scale,
    estimate_forward_phase_with_tp,
    estimate_forward_with_tp,
    estimate_forward_with_tp_from_graph,
)

if TYPE_CHECKING:
    from mvp_train_types import ModelArchitecture, TrainCalibration, TrainConfig

import torch

from mvp_train_estimator import (
    estimate_backward_flops,
    estimate_backward_from_graph_nodes,
    estimate_forward_flops,
    estimate_gradient_communication_bytes,
    estimate_model_architecture,
    estimate_optimizer_flops,
    estimate_train_step,
    build_gradient_bytes_mapping,
    build_train_estimate_report,
)
from mvp_train_types import TrainConfig, TrainPhaseSummary, TrainStepEstimate


def estimate_train_step_with_tp(
    batch_size: int,
    seq_len: int,
    arch: "ModelArchitecture",
    calibration: "TrainCalibration",
    config: "TrainConfig",
    backward_graph_info: "BackwardGraphInfo | None" = None,
    forward_graph_nodes: "list[NodeEstimate] | None" = None,
) -> TrainStepEstimate:
    """
    Unified training step estimation with TP support.

    This function combines:
    - Mission 3: Forward TP estimation (精细化 per-op TP 缩放)
    - Mission 2: Backward communication model (延迟-带宽分离)
    - Mission 4: Optimizer TP estimation (延迟-带宽分离)

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        arch: Model architecture
        calibration: Training calibration
        config: Training configuration (includes tp_size, ddp_enabled)
        backward_graph_info: Optional backward graph info from Mission 1
        forward_graph_nodes: Optional list of forward graph NodeEstimates for
                           per-node backward estimation (recommended for consistency
                           with Single/DDP mode). If not provided, falls back to
                           formula-based estimation.

    Returns:
        TrainStepEstimate with detailed time breakdown
    """
    tp_size = config.tp_size
    ddp_enabled = config.ddp_enabled

    # Get TP calibration parameters
    has_nvlink = getattr(calibration, "has_nvlink", True)
    overlap_ratio = getattr(calibration, "overlap_ratio", 0.3)

    # ===== Forward Pass (Mission 3) =====
    if tp_size > 1:
        # Use refined TP scaling from Mission 3
        forward_result = estimate_forward_phase_with_tp(
            batch_size=batch_size,
            seq_len=seq_len,
            arch=arch,
            calibration=calibration,
            tp_size=tp_size,
            use精细化缩放=True,
        )
        # Apply TP forward efficiency calibration (dividing since estimate is too low)
        tp_forward_eff = getattr(calibration, 'tp_forward_efficiency', 0.05)
        forward_time_ms = forward_result["total_time_ms"] / tp_forward_eff
        forward_compute_time_ms = forward_result["compute_time_ms"] / tp_forward_eff
        forward_memory_time_ms = forward_result["memory_time_ms"] / tp_forward_eff
        forward_flops = forward_result["flops"]
        effective_tp_scale = forward_result["effective_tp_scale"]
    else:
        # Use standard estimation
        forward_flops = estimate_forward_flops(batch_size, seq_len, arch)
        effective_forward_tflops = calibration.gemm_tflops * calibration.effective_tflops_scale
        forward_compute_time_ms = (
            forward_flops / (effective_forward_tflops * 1e12) * 1e3
        )
        activation_memory_bytes = (
            batch_size * seq_len * arch.hidden_size * 4 * arch.num_layers * 2
        )
        forward_memory_time_ms = (
            activation_memory_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
        )
        kernel_overhead_ms = arch.num_layers * 0.15
        forward_time_ms = max(forward_compute_time_ms, forward_memory_time_ms) + kernel_overhead_ms
        effective_tp_scale = 1.0

    # ===== Backward Pass (Mission 2) =====
    # Use per-node backward estimation for consistency across all modes.
    # estimate_backward_from_graph_nodes() returns total time that already includes:
    # max(compute, memory) + overhead. So for Single mode it's complete.
    # For TP mode, we scale the compute time and add communication.

    # Build gradient bytes mapping from backward_info if available
    gradient_bytes_by_scope = None
    if backward_graph_info is not None and hasattr(backward_graph_info, 'gradient_infos'):
        gradient_bytes_by_scope = build_gradient_bytes_mapping(backward_graph_info.gradient_infos)

    # Backward communication time (TP AllReduce) - always calculated for TP
    backward_comm_estimate = estimate_backward_comm_simple(
        num_layers=arch.num_layers,
        hidden_size=arch.hidden_size,
        batch_size=batch_size,
        seq_len=seq_len,
        tp_size=tp_size,
        calibration=calibration,
        num_parameters=arch.parameters,
    )
    backward_comm_time_ms = backward_comm_estimate.effective_comm_time_ms

    if forward_graph_nodes is not None and len(forward_graph_nodes) > 0:
        # Use per-node backward estimation (consistent with Single/DDP mode)
        # estimate_backward_from_graph_nodes returns (total_time, total_flops, total_bytes, breakdown)
        # where total_time = sum of per-node times, each already includes max(compute, memory) + overhead
        (
            backward_time_from_graph,
            backward_flops,
            backward_bytes,
            backward_breakdown,
        ) = estimate_backward_from_graph_nodes(forward_graph_nodes, calibration, gradient_bytes_by_scope)

        if tp_size > 1:
            # For TP mode with graph-based estimation: use per-node estimate as-is.
            # The per-node estimation accounts for compute, memory, and overhead.
            # Only add TP communication time on top.
            backward_time_ms = backward_time_from_graph + backward_comm_time_ms
            backward_compute_time_ms = backward_time_from_graph
        else:
            # Single card: use the per-node estimate directly (already complete)
            backward_time_ms = backward_time_from_graph
            backward_compute_time_ms = backward_time_from_graph
    else:
        # Fallback to formula-based estimation (original TP logic)
        backward_scale = calibration.backward_compute_scale
        backward_flops_estimate = estimate_backward_flops(forward_flops, calibration)

        # Memory bandwidth estimate for backward
        backward_activation_bytes = (
            batch_size * seq_len * arch.hidden_size * 4 * arch.num_layers * 2 * 3
        )
        backward_memory_time_ms = (
            backward_activation_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
        )

        # Per-layer overhead (kernel launch, constant cost per layer)
        per_layer_overhead_ms = (
            calibration.launch_overhead_ms * calibration.overhead_scale
        )
        num_ops_per_layer = 7  # q/k/v/o + gate/up/down projections
        total_overhead_ms = arch.num_layers * num_ops_per_layer * per_layer_overhead_ms

        if tp_size > 1:
            # TP-specific backward efficiency
            tp_backward_efficiency = getattr(calibration, 'tp_backward_efficiency', 0.13)
            backward_effective_tflops = calibration.gemm_tflops * tp_backward_efficiency
            total_backward_flops = backward_flops_estimate * tp_size
            backward_compute_time_ms = (
                total_backward_flops / tp_size / (backward_effective_tflops * 1e12) * 1e3
            )
            backward_time_ms = max(backward_compute_time_ms, backward_memory_time_ms) + backward_comm_time_ms + total_overhead_ms
        else:
            # Single card: use the standard formula
            backward_effective_tflops = max(calibration.gemm_tflops * 0.07, 14.5)
            backward_compute_time_ms = (
                backward_flops_estimate / (backward_effective_tflops * 1e12) * 1e3
            )
            backward_time_ms = max(backward_compute_time_ms, backward_memory_time_ms) + total_overhead_ms

        backward_flops = backward_flops_estimate
        backward_bytes = backward_activation_bytes

    backward_time_ms = backward_time_ms * config.gradient_accumulation_steps

    # ===== Optimizer Step (Mission 4) =====
    num_params = arch.optimizer_param_count

    if tp_size > 1:
        # Use refined optimizer estimation from Mission 4
        optimizer_result = estimate_optimizer_time_latency_bandwidth(
            num_parameters=num_params,
            tp_size=tp_size,
            ddp_enabled=ddp_enabled,
            has_nvlink=has_nvlink,
            overlap_ratio=overlap_ratio,
            memory_bandwidth_gbps=calibration.memory_bandwidth_gbps,
            optimizer_efficiency=get_optimizer_efficiency(calibration),
        )
        optimizer_time_ms = optimizer_result["total_time_ms"]
    else:
        # Use standard estimation
        optimizer_flops, optimizer_bytes = estimate_optimizer_flops(
            num_params, batch_size, seq_len, calibration
        )
        optimizer_memory_time_ms = (
            optimizer_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
        )
        optimizer_time_ms = optimizer_memory_time_ms * calibration.optimizer_scale_factor

    # ===== Communication (DDP gradient allreduce) =====
    # Only add DDP gradient allreduce when DDP is enabled
    # For pure TP (ddp_enabled=False), the TP AllReduce is already included in backward_comm_time_ms
    comm_time_ms = 0.0
    if ddp_enabled:
        comm_bytes, comm_latency_ms = estimate_gradient_communication_bytes(
            num_params, tp_size, ddp_enabled, has_nvlink=has_nvlink
        )
        comm_time_ms = comm_latency_ms
        if comm_bytes > 0:
            comm_time_ms += (
                comm_bytes / (calibration.gradient_allreduce_tflops * 1e9) * 1e3
            )

    # ===== Total =====
    total_time_ms = forward_time_ms + backward_time_ms + optimizer_time_ms + comm_time_ms

    # Samples per second
    samples_per_sec = (batch_size * 1000.0) / total_time_ms if total_time_ms > 0 else 0.0
    tokens_per_sec = (
        (batch_size * seq_len * 1000.0) / total_time_ms if total_time_ms > 0 else None
    )

    # Build phase summaries
    forward_summary = TrainPhaseSummary(
        phase="forward",
        estimated_time_ms=forward_time_ms,
        flops=forward_flops,
        bytes_moved=0,
        compute_time_ms=forward_compute_time_ms,
        memory_time_ms=forward_memory_time_ms,
        comm_time_ms=0.0,
        node_count=arch.num_layers * 2 + 1,
        top_ops=[],
        op_family_breakdown_ms={},
    )

    backward_summary = TrainPhaseSummary(
        phase="backward",
        estimated_time_ms=backward_time_ms,
        flops=backward_flops,
        bytes_moved=0,
        compute_time_ms=backward_compute_time_ms,
        memory_time_ms=forward_memory_time_ms * 3.0,
        comm_time_ms=backward_comm_time_ms,
        node_count=arch.num_layers * 2 + 1,
        top_ops=[],
        op_family_breakdown_ms={},
    )

    optimizer_summary = TrainPhaseSummary(
        phase="optimizer",
        estimated_time_ms=optimizer_time_ms,
        flops=0,
        bytes_moved=0,
        compute_time_ms=0.0,
        memory_time_ms=0.0,
        comm_time_ms=0.0,
        node_count=1,
        top_ops=[],
        op_family_breakdown_ms={},
    )

    return TrainStepEstimate(
        forward_time_ms=forward_time_ms,
        backward_time_ms=backward_time_ms,
        optimizer_time_ms=optimizer_time_ms,
        total_time_ms=total_time_ms,
        samples_per_sec=samples_per_sec,
        tokens_per_sec=tokens_per_sec,
        forward_summary=forward_summary,
        backward_summary=backward_summary,
        optimizer_summary=optimizer_summary,
    )


def build_unified_estimate_report(
    arch: "ModelArchitecture",
    config: "TrainConfig",
    step_estimate: TrainStepEstimate,
    calibration: "TrainCalibration",
    num_train_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a complete training estimate report with TP details."""

    report = build_train_estimate_report(arch, config, step_estimate, calibration, num_train_tokens)

    # Add TP-specific information
    tp_info = {
        "tp_enabled": config.tp_size > 1,
        "tp_size": config.tp_size,
        "ddp_enabled": config.ddp_enabled,
        "has_nvlink": getattr(calibration, "has_nvlink", True),
        "overlap_ratio": getattr(calibration, "overlap_ratio", 0.3),
    }

    # Add scale analysis if TP is enabled
    if config.tp_size > 1:
        scale_breakdown = analyze_tp_scale_breakdown(arch, config.tp_size)
        tp_info["scale_analysis"] = scale_breakdown

    report["tp_info"] = tp_info

    return report


# Re-export all mission types and functions for convenience
__all__ = [
    # From Mission 1
    "extract_backward_graph",
    "extract_training_graphs",
    "BackwardGraphInfo",
    "GradientInfo",
    "get_gradient_summary",
    "TrainingGraphs",
    # From Mission 2
    "estimate_backward_comm_simple",
    "estimate_backward_with_comm",
    "BackwardCommEstimate",
    "CommCalibration",
    "create_train_calibration_with_comm_params",
    # From Mission 3
    "estimate_forward_with_tp",
    "estimate_forward_phase_with_tp",
    "estimate_forward_with_tp_from_graph",
    "compute_effective_tp_scale",
    "analyze_tp_scale_breakdown",
    # From Mission 4
    "estimate_optimizer_tp_overhead",
    "estimate_optimizer_time_latency_bandwidth",
    "compute_effective_tp_scale_for_optimizer",
    "TPTrainCalibration",
    "from_calibration",
    "get_default_calibration",
    # Unified
    "estimate_train_step_with_tp",
    "build_unified_estimate_report",
]
