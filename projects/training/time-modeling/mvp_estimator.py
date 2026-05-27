from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import replace
from typing import Any

import torch

from mvp_graph import *
from mvp_types import (
    ExecutionConfig,
    HardwareCalibration,
    ModuleProfileRecord,
    NodeEstimate,
    PhaseSummary,
    placement_for_rank,
)


def estimate_linear(
    node: torch.fx.Node, outputs: list[Any], calibration: HardwareCalibration
) -> tuple[float, float]:
    input_metas = arg_output_metas(node.args[0])
    weight_metas = arg_output_metas(node.args[1])
    if not input_metas or not weight_metas:
        return 0.0, 0.0
    input_shape = tuple(int(dim) for dim in input_metas[0].shape)
    weight_shape = tuple(int(dim) for dim in weight_metas[0].shape)
    if len(input_shape) < 2 or len(weight_shape) != 2:
        return 0.0, 0.0
    m = product(input_shape[:-1])
    k = input_shape[-1]
    n = weight_shape[0]
    flops = float(2 * m * n * k)
    _, input_bytes, _ = metas_numel_and_bytes(input_metas)
    _, weight_bytes, _ = metas_numel_and_bytes(weight_metas)
    _, output_bytes, _ = metas_numel_and_bytes(outputs)
    bytes_moved = float(input_bytes + weight_bytes + output_bytes)
    return flops, bytes_moved


def estimate_attention(
    node: torch.fx.Node, outputs: list[Any], calibration: HardwareCalibration
) -> tuple[float, float]:
    q_metas = arg_output_metas(node.args[0])
    k_metas = arg_output_metas(node.args[1])
    v_metas = arg_output_metas(node.args[2])
    if not q_metas or not k_metas or not v_metas:
        return 0.0, 0.0
    q_shape = tuple(int(dim) for dim in q_metas[0].shape)
    k_shape = tuple(int(dim) for dim in k_metas[0].shape)
    if len(q_shape) != 4 or len(k_shape) != 4:
        return 0.0, 0.0
    batch, heads, q_len, head_dim = q_shape
    k_len = k_shape[2]
    flops = float(4 * batch * heads * q_len * k_len * head_dim)
    _, q_bytes, _ = metas_numel_and_bytes(q_metas)
    _, k_bytes, _ = metas_numel_and_bytes(k_metas)
    _, v_bytes, _ = metas_numel_and_bytes(v_metas)
    _, out_bytes, _ = metas_numel_and_bytes(outputs)
    bytes_moved = float(q_bytes + k_bytes + v_bytes + out_bytes)
    return flops, bytes_moved


def estimate_embedding(
    node: torch.fx.Node, outputs: list[Any], calibration: HardwareCalibration
) -> tuple[float, float]:
    index_metas = arg_output_metas(node.args[1])
    output_metas = outputs
    _, index_bytes, _ = metas_numel_and_bytes(index_metas)
    _, output_bytes, _ = metas_numel_and_bytes(output_metas)
    return 0.0, float(index_bytes + output_bytes * 2)


def estimate_memory_like(
    node: torch.fx.Node, outputs: list[Any], scale: float = 1.0
) -> tuple[float, float]:
    input_bytes = 0
    for arg in node.args:
        _, arg_bytes, _ = metas_numel_and_bytes(arg_output_metas(arg))
        input_bytes += arg_bytes
    _, output_bytes, _ = metas_numel_and_bytes(outputs)
    return 0.0, float((input_bytes + output_bytes) * scale)


def estimate_node(
    node: torch.fx.Node, phase: str, calibration: HardwareCalibration
) -> NodeEstimate | None:
    if node.op != "call_function":
        return None
    target = canonical_target_name(node.target)
    family = op_family_from_target(target)
    scope = module_scope_from_stack(node.meta.get("nn_module_stack"))
    region = region_from_scope(scope)
    outputs = node_output_metas(node)
    _, output_bytes, output_dtype = metas_numel_and_bytes(outputs)
    output_shapes = tensor_shapes(outputs)
    flops = 0.0
    bytes_moved = float(output_bytes)
    runtime_overhead_ms = calibration.launch_overhead_ms

    if family == "gemm":
        flops, bytes_moved = estimate_linear(node, outputs, calibration)
        compute_time_ms = (
            flops / (calibration.gemm_tflops * 1.0e12) * 1.0e3 if flops else 0.0
        )
        memory_time_ms = (
            bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
            if bytes_moved
            else 0.0
        )
    elif family == "attention":
        flops, bytes_moved = estimate_attention(node, outputs, calibration)
        compute_time_ms = (
            flops / (calibration.attention_tflops * 1.0e12) * 1.0e3 if flops else 0.0
        )
        memory_time_ms = (
            bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
            if bytes_moved
            else 0.0
        )
    elif family == "embedding":
        flops, bytes_moved = estimate_embedding(node, outputs, calibration)
        compute_time_ms = 0.0
        memory_time_ms = (
            bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
            if bytes_moved
            else 0.0
        )
    elif family == "pointwise":
        flops, bytes_moved = estimate_memory_like(node, outputs)
        compute_time_ms = (
            flops / (calibration.attention_tflops * 1.0e12) * 1.0e3 if flops else 0.0
        )
        memory_time_ms = (
            bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
            if bytes_moved
            else 0.0
        )
    elif family == "reduction":
        flops, bytes_moved = estimate_memory_like(node, outputs)
        compute_time_ms = (
            flops / (calibration.attention_tflops * 1.0e12) * 1.0e3 if flops else 0.0
        )
        memory_time_ms = (
            bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
            if bytes_moved
            else 0.0
        )
    elif family == "concat":
        flops, bytes_moved = estimate_memory_like(node, outputs, scale=1.0)
        compute_time_ms = 0.0
        memory_time_ms = (
            bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
            if bytes_moved
            else 0.0
        )
    elif family == "view":
        flops = 0.0
        bytes_moved = 0.0
        compute_time_ms = 0.0
        memory_time_ms = 0.0
    else:
        flops, bytes_moved = estimate_memory_like(node, outputs, scale=1.0)
        compute_time_ms = 0.0
        memory_time_ms = (
            bytes_moved / (calibration.memory_bandwidth_gbps * 1.0e9) * 1.0e3
            if bytes_moved
            else 0.0
        )

    estimated_time_ms = max(compute_time_ms, memory_time_ms) + runtime_overhead_ms
    return NodeEstimate(
        node_name=node.name,
        target=target,
        op_family=family,
        phase=phase,
        region=region,
        module_scope=scope,
        output_shapes=output_shapes,
        output_dtype=output_dtype,
        shape_signature=shape_signature(output_shapes),
        ordinal=0,
        flops=flops,
        bytes_moved=bytes_moved,
        compute_time_ms=compute_time_ms,
        memory_time_ms=memory_time_ms,
        runtime_overhead_ms=runtime_overhead_ms,
        estimated_time_ms=estimated_time_ms,
    )


def summarize_phase(phase: str, estimates: list[NodeEstimate]) -> PhaseSummary:
    op_family_breakdown = defaultdict(float)
    region_breakdown = defaultdict(float)
    graph_compute = 0.0
    runtime_overhead = 0.0
    for estimate in estimates:
        op_family_breakdown[estimate.op_family] += estimate.estimated_time_ms
        region_breakdown[estimate.region] += estimate.estimated_time_ms
        graph_compute += max(estimate.compute_time_ms, estimate.memory_time_ms)
        runtime_overhead += estimate.runtime_overhead_ms
    top_ops = [
        {
            "node_name": item.node_name,
            "target": item.target,
            "region": item.region,
            "module_scope": item.module_scope,
            "estimated_time_ms": round(item.estimated_time_ms, 4),
            "compute_time_ms": round(item.compute_time_ms, 4),
            "memory_time_ms": round(item.memory_time_ms, 4),
        }
        for item in sorted(estimates, key=lambda x: x.estimated_time_ms, reverse=True)[
            :10
        ]
    ]
    top_regions = [
        {"region": key, "estimated_time_ms": round(value, 4)}
        for key, value in sorted(
            region_breakdown.items(), key=lambda kv: kv[1], reverse=True
        )[:10]
    ]
    return PhaseSummary(
        phase=phase,
        graph_compute_time_ms=graph_compute,
        graph_comm_time_ms=0.0,
        runtime_overhead_time_ms=runtime_overhead,
        end_to_end_time_ms=graph_compute + runtime_overhead,
        node_count=len(estimates),
        top_ops=top_ops,
        top_regions=top_regions,
        op_family_breakdown_ms={
            key: round(value, 4)
            for key, value in sorted(
                op_family_breakdown.items(), key=lambda kv: kv[1], reverse=True
            )
        },
    )


def finalize_estimate_ordinals(estimates: list[NodeEstimate]) -> list[NodeEstimate]:
    counters: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for estimate in estimates:
        key = (
            estimate.phase,
            estimate.target,
            estimate.module_scope,
            estimate.shape_signature,
        )
        estimate.ordinal = counters[key]
        counters[key] += 1
    return estimates


def summarize_phase_with_module_substitution(
    phase: str,
    estimates: list[NodeEstimate],
    module_records: list[ModuleProfileRecord],
    graph_comm_time_ms: float = 0.0,
) -> PhaseSummary:
    covered = {
        node_id for record in module_records for node_id in record.covered_node_ids
    }
    residual_estimates = [
        estimate for estimate in estimates if estimate.node_name not in covered
    ]

    op_family_breakdown = defaultdict(float)
    region_breakdown = defaultdict(float)
    graph_compute = 0.0
    runtime_overhead = 0.0

    for estimate in residual_estimates:
        op_family_breakdown[estimate.op_family] += estimate.estimated_time_ms
        region_breakdown[estimate.region] += estimate.estimated_time_ms
        graph_compute += max(estimate.compute_time_ms, estimate.memory_time_ms)
        runtime_overhead += estimate.runtime_overhead_ms

    for record in module_records:
        op_family_breakdown[f"module:{record.module_kind}"] += record.mean_ms
        region_breakdown[record.module_kind] += record.mean_ms
        graph_compute += record.mean_ms

    top_ops = [
        {
            "node_name": item.node_name,
            "target": item.target,
            "region": item.region,
            "module_scope": item.module_scope,
            "estimated_time_ms": round(item.estimated_time_ms, 4),
            "compute_time_ms": round(item.compute_time_ms, 4),
            "memory_time_ms": round(item.memory_time_ms, 4),
        }
        for item in sorted(
            residual_estimates, key=lambda x: x.estimated_time_ms, reverse=True
        )[:10]
    ]
    top_ops = [
        {
            "node_name": record.module_scope,
            "target": record.substitution_policy,
            "region": record.module_kind,
            "module_scope": record.module_scope,
            "estimated_time_ms": round(record.mean_ms, 4),
            "compute_time_ms": round(record.mean_ms, 4),
            "memory_time_ms": 0.0,
        }
        for record in sorted(
            module_records, key=lambda item: item.mean_ms, reverse=True
        )[:10]
    ] + top_ops
    top_ops = sorted(top_ops, key=lambda item: item["estimated_time_ms"], reverse=True)[
        :10
    ]
    top_regions = [
        {"region": key, "estimated_time_ms": round(value, 4)}
        for key, value in sorted(
            region_breakdown.items(), key=lambda kv: kv[1], reverse=True
        )[:10]
    ]
    return PhaseSummary(
        phase=phase,
        graph_compute_time_ms=graph_compute,
        graph_comm_time_ms=graph_comm_time_ms,
        runtime_overhead_time_ms=runtime_overhead,
        end_to_end_time_ms=graph_compute + runtime_overhead + graph_comm_time_ms,
        node_count=len(residual_estimates) + len(module_records),
        top_ops=top_ops,
        top_regions=top_regions,
        op_family_breakdown_ms={
            key: round(value, 4)
            for key, value in sorted(
                op_family_breakdown.items(), key=lambda kv: kv[1], reverse=True
            )
        },
    )


def aggregate_module_profiles(
    gathered_module_profiles: list[dict[str, list[dict[str, Any]]]],
) -> dict[str, list[ModuleProfileRecord]]:
    if len(gathered_module_profiles) == 1:
        payload = gathered_module_profiles[0]
        return {
            phase: [ModuleProfileRecord(**record) for record in payload.get(phase, [])]
            for phase in ("prefill", "decode_step")
        }
    aggregated: dict[str, list[ModuleProfileRecord]] = {
        "prefill": [],
        "decode_step": [],
    }
    for phase in aggregated:
        by_scope: dict[str, list[ModuleProfileRecord]] = defaultdict(list)
        for payload in gathered_module_profiles:
            for record in payload.get(phase, []):
                by_scope[record["module_scope"]].append(ModuleProfileRecord(**record))
        for scope in sorted(by_scope.keys(), key=module_scope_key):
            slowest = max(by_scope[scope], key=lambda item: item.mean_ms)
            aggregated[phase].append(slowest)
    return aggregated


def covered_estimates_for_scope(
    scope: str, phase_estimates: list[NodeEstimate]
) -> list[NodeEstimate]:
    aliases = scope_aliases(scope)
    return [
        estimate
        for estimate in phase_estimates
        if any(estimate.module_scope.startswith(alias) for alias in aliases)
    ]


def sanitize_module_profiles(
    phase_estimates: list[NodeEstimate],
    module_records: list[ModuleProfileRecord],
) -> list[ModuleProfileRecord]:
    if not module_records:
        return []

    peer_medians = {
        module_kind: statistics.median(
            [item.mean_ms for item in module_records if item.module_kind == module_kind]
        )
        for module_kind in {item.module_kind for item in module_records}
    }
    sanitized: list[ModuleProfileRecord] = []
    for record in module_records:
        covered = covered_estimates_for_scope(record.module_scope, phase_estimates)
        if not covered:
            sanitized.append(record)
            continue

        covered_estimate_ms = sum(item.estimated_time_ms for item in covered)
        peer_median_ms = peer_medians.get(record.module_kind, record.mean_ms)
        is_outlier = (
            covered_estimate_ms > 0.0
            and record.mean_ms > peer_median_ms * 4.0
            and record.mean_ms > covered_estimate_ms * 3.0
            and record.mean_ms - covered_estimate_ms > 1.0
        )
        if not is_outlier:
            sanitized.append(record)
            continue

        fallback_ms = covered_estimate_ms
        sanitized.append(
            replace(
                record,
                substitution_policy="module_profile_outlier_fallback_to_estimate",
                mean_ms=fallback_ms,
                median_ms=fallback_ms,
                min_ms=fallback_ms,
                max_ms=fallback_ms,
                samples_ms=[fallback_ms],
            )
        )
    return sanitized


def build_estimate_compare_rows(
    estimates: list[NodeEstimate], execution: ExecutionConfig
) -> list[list[dict[str, Any]]]:
    rows_by_rank: list[list[dict[str, Any]]] = []
    rank_count = execution.world_size if execution.parallel_mode == "tp" else 1
    sharded_estimates = [
        tp_shard_node_estimate(estimate, execution) for estimate in estimates
    ]
    for rank in range(rank_count):
        placement = placement_for_rank(execution, rank)
        rank_rows = []
        for estimate in sharded_estimates:
            rank_rows.append(
                {
                    "phase": estimate.phase,
                    "rank": rank,
                    "host": placement.host,
                    "node_rank": placement.node_rank,
                    "local_rank": placement.local_rank,
                    "device": placement.physical_device,
                    "node_name": estimate.node_name,
                    "target": tp_localized_target(estimate, execution),
                    "op_family": estimate.op_family,
                    "scope": estimate.module_scope,
                    "shape_signature": tp_compare_shape_signature(estimate, execution),
                    "ordinal": 0,
                    "est_ms": estimate.estimated_time_ms,
                }
            )
        assign_ordinals_by_group(
            rank_rows, ("phase", "target", "scope", "shape_signature")
        )
        rows_by_rank.append(rank_rows)
    return rows_by_rank


def estimate_collective_bandwidth_gbps(topology: str) -> float:
    if topology.startswith("NV"):
        return 280.0
    if topology in {"PIX", "PXB", "PHB", "NODE"}:
        return 48.0
    return 24.0


def estimate_collective_latency_ms(topology: str) -> float:
    if topology.startswith("NV"):
        return 0.02
    if topology in {"PIX", "PXB", "PHB", "NODE"}:
        return 0.06
    return 0.1


def collective_bandwidth_gbps(execution: ExecutionConfig) -> float:
    if execution.collective_bandwidth_gbps is not None:
        return execution.collective_bandwidth_gbps
    if execution.nnodes > 1:
        if execution.interconnect == "infiniband":
            return 25.0
        if execution.interconnect == "roce":
            return 18.0
        return 12.5
    return estimate_collective_bandwidth_gbps(execution.topology)


def collective_latency_ms(execution: ExecutionConfig) -> float:
    if execution.collective_latency_ms is not None:
        return execution.collective_latency_ms
    if execution.nnodes > 1:
        if execution.interconnect == "infiniband":
            return 0.08
        if execution.interconnect == "roce":
            return 0.12
        return 0.25
    return estimate_collective_latency_ms(execution.topology)


def build_predicted_comm(
    estimates: list[NodeEstimate],
    execution: ExecutionConfig,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if execution.parallel_mode != "tp" or execution.tp_size <= 1:
        return {
            "collectives": [],
            "predicted_collectives": [],
            "predicted_total_ms": 0.0,
        }
    predicted_collectives = []
    # Use config tp.communication values when available, fall back to execution params
    if config is not None:
        from config.config_loader import get_tp_comm_params

        bandwidth_gbps, base_latency_ms = get_tp_comm_params(
            config, execution.interconnect, execution.nnodes
        )
    else:
        bandwidth_gbps = collective_bandwidth_gbps(execution)
        base_latency_ms = collective_latency_ms(execution)
    for estimate in estimates:
        if estimate.op_family != "gemm" or not is_tp_rowwise_scope(
            estimate.module_scope
        ):
            continue
        output_bytes = estimate_output_bytes(estimate)
        ring_factor = 2.0 * (execution.tp_size - 1) / execution.tp_size
        predicted_ms = (
            base_latency_ms
            + ring_factor * output_bytes / (bandwidth_gbps * 1.0e9) * 1.0e3
        )
        predicted_collectives.append(
            {
                "collective": "all_reduce",
                "scope": estimate.module_scope,
                "count": 1,
                "bytes": output_bytes,
                "predicted_ms": predicted_ms,
                "topology": execution.topology,
            }
        )
    return {
        "collectives": [],
        "predicted_collectives": predicted_collectives,
        "predicted_total_ms": sum(
            item["predicted_ms"] * item["count"] for item in predicted_collectives
        ),
    }
