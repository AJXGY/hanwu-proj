from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HardwareCalibration:
    accelerator_kind: str
    device_name: str
    device_index: int
    gemm_tflops: float
    attention_tflops: float
    memory_bandwidth_gbps: float
    launch_overhead_ms: float


@dataclass
class NodeEstimate:
    node_name: str
    target: str
    op_family: str
    phase: str
    region: str
    module_scope: str
    output_shapes: list[list[int]]
    output_dtype: str
    shape_signature: str
    ordinal: int
    flops: float
    bytes_moved: float
    compute_time_ms: float
    memory_time_ms: float
    runtime_overhead_ms: float
    estimated_time_ms: float


@dataclass
class RankPlacement:
    rank: int
    host: str
    node_rank: int
    local_rank: int
    physical_device: int


@dataclass
class ExecutionConfig:
    accelerator_kind: str
    parallel_mode: str
    physical_devices: list[int]
    visible_devices: str
    world_size: int
    tp_size: int
    topology: str
    local_topology: str
    interconnect: str
    nnodes: int
    nproc_per_node: int
    host_name: str
    master_addr: str
    master_port: int
    local_device: int
    placements: list[RankPlacement]
    collective_bandwidth_gbps: float | None
    collective_latency_ms: float | None
    rank: int
    local_rank: int
    node_rank: int


def placement_for_rank(execution: ExecutionConfig, rank: int) -> RankPlacement:
    for placement in execution.placements:
        if placement.rank == rank:
            return placement
    physical_device = (
        execution.physical_devices[0]
        if execution.physical_devices
        else execution.local_device
    )
    return RankPlacement(
        rank=rank,
        host=execution.host_name,
        node_rank=execution.node_rank,
        local_rank=execution.local_rank,
        physical_device=physical_device,
    )


@dataclass
class RuntimeInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    next_token: torch.Tensor
    next_attention_mask: torch.Tensor
    decode_past: Any


@dataclass
class PhaseSummary:
    phase: str
    graph_compute_time_ms: float
    graph_comm_time_ms: float
    phase_adjustment_time_ms: float
    runtime_overhead_time_ms: float
    end_to_end_time_ms: float
    node_count: int
    top_ops: list[dict[str, Any]]
    top_regions: list[dict[str, Any]]
    op_family_breakdown_ms: dict[str, float]


@dataclass
class ModuleProfileRecord:
    module_scope: str
    module_kind: str
    phase: str
    covered_node_ids: list[str]
    covered_op_families: list[str]
    substitution_policy: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]


@dataclass
class PhaseAdjustmentProfile:
    phase: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]
