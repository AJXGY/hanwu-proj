from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from mvp_types import ExecutionConfig, placement_for_rank


def cuda_wall_time_ms(fn, warmup: int, repeat: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        samples.append((end - start) * 1.0e3)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def benchmark_allreduce_ms(
    num_bytes: float,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    if (
        num_bytes <= 0
        or not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() <= 1
    ):
        return aggregate_sample_stats([0.0 for _ in range(max(repeat, 1))])

    dtype = torch.bfloat16
    element_size = torch.tensor([], dtype=dtype).element_size()
    max_chunk_bytes = 256 * 1024 * 1024
    chunk_bytes = min(max(int(num_bytes), 1), max_chunk_bytes)
    chunk_numel = max(1, math.ceil(chunk_bytes / element_size))
    tensor = torch.ones(chunk_numel, device=device, dtype=dtype)
    num_chunks = max(1, math.ceil(num_bytes / chunk_bytes))

    for _ in range(warmup):
        dist.barrier()
        torch.cuda.synchronize()
        for _ in range(num_chunks):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

    gathered_samples: list[list[float]] = []
    for _ in range(repeat):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_chunks):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1.0e3
        gathered: list[float | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, elapsed_ms)
        gathered_samples.append([float(item or 0.0) for item in gathered])

    worst_case_samples = [max(items) for items in gathered_samples]
    return aggregate_sample_stats(worst_case_samples)


def cuda_wall_time_ms_phases(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    """Measure forward, backward, optimizer times separately using CUDA events.

    Returns dict with keys: forward, backward, optimizer, combined

    For DDP/TP modes, backward measurement includes communication time since
    gradient allreduce happens during backward() call.
    """
    model.train()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def one_step():
        """Single train step: forward + backward + optimizer."""
        model.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    # Warmup
    for _ in range(warmup):
        one_step()

    # Measure each phase separately
    forward_times = []
    backward_times = []
    optimizer_times = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    loss = None

    for _ in range(repeat):
        # === Forward phase ===
        model.zero_grad()
        torch.cuda.synchronize()
        start_event.record()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        end_event.record()
        torch.cuda.synchronize()
        forward_ms = start_event.elapsed_time(end_event)
        forward_times.append(forward_ms)

        # === Backward phase ===
        start_event.record()
        loss.backward()
        end_event.record()
        torch.cuda.synchronize()
        backward_ms = start_event.elapsed_time(end_event)
        backward_times.append(backward_ms)

        # === Optimizer phase ===
        start_event.record()
        optimizer.step()
        end_event.record()
        torch.cuda.synchronize()
        optimizer_ms = start_event.elapsed_time(end_event)
        optimizer_times.append(optimizer_ms)

        optimizer.zero_grad()

    return {
        "forward": aggregate_sample_stats(forward_times),
        "backward": aggregate_sample_stats(backward_times),
        "optimizer": aggregate_sample_stats(optimizer_times),
        "combined": aggregate_sample_stats([
            f + b + o for f, b, o in zip(forward_times, backward_times, optimizer_times)
        ]),
    }


def aggregate_sample_stats(samples: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def cuda_wall_time_ms_phases_tp(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    warmup: int,
    repeat: int,
    tp_size: int = 1,
    num_parameters: int = 0,
) -> dict[str, Any]:
    """Measure forward, backward, optimizer times for TP mode.

    For TP mode, backward measurement includes both compute and communication
    since they are fused in the backward pass. We don't attempt to separate
    them here - the separation is done during calibration based on estimates.

    Returns dict with keys: forward, backward, optimizer, combined, backward_total
    """
    # Fall back to standard phases - TP backward includes comm
    result = cuda_wall_time_ms_phases(
        model, input_ids, attention_mask, labels, optimizer, warmup, repeat
    )
    # Add backward_total for compatibility
    result["backward_total"] = result["backward"]
    return result


def _estimate_num_layers(model: nn.Module) -> int:
    """Estimate the number of transformer layers in the model."""
    num_layers = 0
    for name, _ in model.named_modules():
        # Look for patterns like "layers.0" or "h.0" which indicate layer numbering
        if ".layers." in name or ".h." in name:
            for part in name.split("."):
                if part.isdigit():
                    num_layers = max(num_layers, int(part) + 1)
        # Also check for "transformer.h." pattern (GPT-2 style)
        if ".transformer.h." in name:
            for part in name.split("."):
                if part.isdigit():
                    num_layers = max(num_layers, int(part) + 1)
    return max(num_layers, 1)  # At least 1 layer


def distributed_cuda_wall_time_ms(
    fn,
    warmup: int,
    repeat: int,
    execution: ExecutionConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if execution.parallel_mode == "single":
        measured = cuda_wall_time_ms(fn, warmup, repeat)
        return measured, [
            {
                "rank": execution.rank,
                "device": execution.physical_devices[0],
                **measured,
            }
        ]

    for _ in range(warmup):
        dist.barrier()
        fn()
        torch.cuda.synchronize()
    dist.barrier()
    local_samples: list[float] = []
    gathered_samples: list[list[float]] = []
    for _ in range(repeat):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1.0e3
        local_samples.append(elapsed_ms)
        gathered: list[float | None] = [None for _ in range(execution.world_size)]
        dist.all_gather_object(gathered, elapsed_ms)
        gathered_samples.append([float(item or 0.0) for item in gathered])
    aggregate_samples = [max(sample) for sample in gathered_samples]
    per_rank_samples = list(map(list, zip(*gathered_samples)))
    rank_measurements = []
    for rank, samples in enumerate(per_rank_samples):
        placement = placement_for_rank(execution, rank)
        rank_measurements.append(
            {
                "rank": rank,
                "host": placement.host,
                "node_rank": placement.node_rank,
                "local_rank": placement.local_rank,
                "device": placement.physical_device,
                **aggregate_sample_stats(samples),
            }
        )
    return aggregate_sample_stats(aggregate_samples), rank_measurements


def gather_rank_objects(value: Any, execution: ExecutionConfig) -> list[Any]:
    if execution.parallel_mode == "single":
        return [value]
    gathered: list[Any] = [None for _ in range(execution.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def merge_comm_summaries(per_rank_comm: list[dict[str, Any]]) -> dict[str, Any]:
    collectives: dict[str, dict[str, Any]] = {}
    total_ms = 0.0
    for item in per_rank_comm:
        total_ms += float(item.get("total_measured_ms", 0.0))
        for entry in item.get("collectives", []):
            row = collectives.setdefault(
                entry["collective"],
                {
                    "collective": entry["collective"],
                    "count": 0,
                    "total_measured_ms": 0.0,
                    "per_rank": [],
                },
            )
            row["count"] += int(entry["count"])
            row["total_measured_ms"] += float(entry["total_measured_ms"])
            row["per_rank"].append(
                {
                    "rank": entry["rank"],
                    "host": entry.get("host"),
                    "node_rank": entry.get("node_rank"),
                    "local_rank": entry.get("local_rank"),
                    "device": entry["device"],
                    "count": entry["count"],
                    "total_measured_ms": entry["total_measured_ms"],
                }
            )
    rows = sorted(
        collectives.values(), key=lambda item: item["total_measured_ms"], reverse=True
    )
    return {"collectives": rows, "total_measured_ms": total_ms}


def build_execution_report(execution: ExecutionConfig) -> dict[str, Any]:
    return {
        "parallel_mode": execution.parallel_mode,
        "physical_devices": execution.physical_devices,
        "visible_devices": execution.visible_devices,
        "world_size": execution.world_size,
        "tp_size": execution.tp_size,
        "topology": execution.topology,
        "local_topology": execution.local_topology,
        "interconnect": execution.interconnect,
        "nnodes": execution.nnodes,
        "nproc_per_node": execution.nproc_per_node,
        "host_name": execution.host_name,
        "master_addr": execution.master_addr,
        "master_port": execution.master_port,
        "local_device": execution.local_device,
        "placements": [
            {
                "rank": placement.rank,
                "host": placement.host,
                "node_rank": placement.node_rank,
                "local_rank": placement.local_rank,
                "physical_device": placement.physical_device,
            }
            for placement in execution.placements
        ],
        "collective_bandwidth_gbps": execution.collective_bandwidth_gbps,
        "collective_latency_ms": execution.collective_latency_ms,
    }


def is_primary_rank(execution: ExecutionConfig) -> bool:
    return execution.rank == 0


def compare_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_est_ms = sum(float(row.get("est_ms", 0.0)) for row in rows)
    matched_est_ms = sum(
        float(row.get("est_ms", 0.0))
        for row in rows
        if row.get("match_method") != "unmatched"
    )
    matched_rows = sum(1 for row in rows if row.get("match_method") != "unmatched")
    return {
        "matched_rows": matched_rows,
        "coverage_estimate_ms_pct": (matched_est_ms / total_est_ms * 100.0)
        if total_est_ms
        else 0.0,
    }


def build_operator_compare_rows(
    phase: str,
    estimate_rows_by_rank: list[list[dict[str, Any]]],
    measured_by_rank: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank_index, rank_rows in enumerate(measured_by_rank):
        estimate_rows = (
            estimate_rows_by_rank[rank_index]
            if rank_index < len(estimate_rows_by_rank)
            else []
        )
        exact_lookup = {
            (
                row["target"],
                row["module_scope"],
                row["shape_signature"],
                row["ordinal"],
            ): row
            for row in rank_rows
        }
        grouped_lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
            list
        )
        for row in rank_rows:
            grouped_lookup[
                (row["op_family"], row["module_scope"], row["shape_signature"])
            ].append(row)
        matched_exact_keys: set[tuple[str, str, str, int]] = set()

        for estimate in estimate_rows:
            exact_key = (
                estimate["target"],
                estimate["scope"],
                estimate["shape_signature"],
                estimate["ordinal"],
            )
            measured = exact_lookup.get(exact_key)
            if measured is not None:
                matched_exact_keys.add(exact_key)
                rows.append(
                    {
                        "phase": phase,
                        "rank": measured["rank"],
                        "host": measured.get("host"),
                        "node_rank": measured.get("node_rank"),
                        "local_rank": measured.get("local_rank"),
                        "device": measured["device"],
                        "node_name": estimate["node_name"],
                        "target": estimate["target"],
                        "scope": estimate["scope"],
                        "shape_signature": estimate["shape_signature"],
                        "ordinal": estimate["ordinal"],
                        "est_ms": estimate["est_ms"],
                        "measured_ms": measured["measured_ms"],
                        "abs_err_ms": abs(estimate["est_ms"] - measured["measured_ms"]),
                        "rel_err_pct": relative_error_pct(
                            estimate["est_ms"], measured["measured_ms"]
                        ),
                        "calls": 1,
                        "match_method": "exact",
                        "match_confidence": 1.0,
                    }
                )

        grouped_estimates: dict[tuple[str, str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for estimate in estimate_rows:
            exact_key = (
                estimate["target"],
                estimate["scope"],
                estimate["shape_signature"],
                estimate["ordinal"],
            )
            if exact_key in matched_exact_keys:
                continue
            grouped_estimates[
                (
                    estimate["op_family"],
                    estimate["scope"],
                    estimate["shape_signature"],
                )
            ].append(estimate)

        matched_group_keys: set[tuple[str, str, str]] = set()
        for group_key, group_estimates in grouped_estimates.items():
            group_measured = grouped_lookup.get(group_key, [])
            if group_measured:
                matched_group_keys.add(group_key)
                est_ms = sum(item["est_ms"] for item in group_estimates)
                measured_ms = sum(item["measured_ms"] for item in group_measured)
                rows.append(
                    {
                        "phase": phase,
                        "rank": group_measured[0]["rank"],
                        "host": group_measured[0].get("host"),
                        "node_rank": group_measured[0].get("node_rank"),
                        "local_rank": group_measured[0].get("local_rank"),
                        "device": group_measured[0]["device"],
                        "node_name": group_estimates[0]["node_name"],
                        "target": group_estimates[0]["target"],
                        "scope": group_key[1],
                        "shape_signature": group_key[2],
                        "ordinal": None,
                        "est_ms": est_ms,
                        "measured_ms": measured_ms,
                        "abs_err_ms": abs(est_ms - measured_ms),
                        "rel_err_pct": relative_error_pct(est_ms, measured_ms),
                        "calls": len(group_measured),
                        "match_method": "shape_scope_grouped",
                        "match_confidence": 0.65,
                    }
                )
            else:
                for estimate in group_estimates:
                    rows.append(
                        {
                            "phase": phase,
                            "rank": rank_rows[0]["rank"] if rank_rows else 0,
                            "host": rank_rows[0].get("host") if rank_rows else None,
                            "node_rank": (
                                rank_rows[0].get("node_rank") if rank_rows else None
                            ),
                            "local_rank": (
                                rank_rows[0].get("local_rank") if rank_rows else None
                            ),
                            "device": rank_rows[0]["device"] if rank_rows else None,
                            "node_name": estimate["node_name"],
                            "target": estimate["target"],
                            "scope": estimate["scope"],
                            "shape_signature": estimate["shape_signature"],
                            "ordinal": estimate["ordinal"],
                            "est_ms": estimate["est_ms"],
                            "measured_ms": 0.0,
                            "abs_err_ms": abs(estimate["est_ms"]),
                            "rel_err_pct": 100.0,
                            "calls": 0,
                            "match_method": "unmatched",
                            "match_confidence": 0.0,
                        }
                    )

        for row in rank_rows:
            exact_key = (
                row["target"],
                row["module_scope"],
                row["shape_signature"],
                row["ordinal"],
            )
            family_group_key = (
                row["op_family"],
                row["module_scope"],
                row["shape_signature"],
            )
            if (
                exact_key in matched_exact_keys
                or family_group_key in matched_group_keys
            ):
                continue
            rows.append(
                {
                    "phase": phase,
                    "rank": row["rank"],
                    "host": row.get("host"),
                    "node_rank": row.get("node_rank"),
                    "local_rank": row.get("local_rank"),
                    "device": row["device"],
                    "node_name": "",
                    "target": row["target"],
                    "scope": row["module_scope"],
                    "shape_signature": row["shape_signature"],
                    "ordinal": row["ordinal"],
                    "est_ms": 0.0,
                    "measured_ms": row["measured_ms"],
                    "abs_err_ms": abs(row["measured_ms"]),
                    "rel_err_pct": 100.0,
                    "calls": row["calls"],
                    "match_method": "unmatched",
                    "match_confidence": 0.0,
                }
            )
    rows.sort(key=lambda item: item["abs_err_ms"], reverse=True)
    return rows


def round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    return value


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(round_nested(report), indent=2), encoding="utf-8")

    prefill = report["estimate"]["prefill"]
    decode = report["estimate"]["decode_step"]
    measured = report["measured"]
    lines = [
        "# Torch Inference MVP Report",
        "",
        f"- model: `{report['model']['path']}`",
        f"- prompt tokens: {report['model']['prompt_tokens']}",
        f"- max new tokens: {report['model']['max_new_tokens']}",
        f"- device: `{report['calibration']['device_name']}`",
        "",
        "## Estimate",
        "",
        f"- prefill end_to_end_time_ms: {prefill['end_to_end_time_ms']:.4f}",
        f"- decode_step end_to_end_time_ms: {decode['end_to_end_time_ms']:.4f}",
        f"- request end_to_end_time_ms: {report['estimate']['request_end_to_end_time_ms']:.4f}",
        f"- prefill module profiles: {len(report['module_profile']['prefill'])}",
        f"- decode module profiles: {len(report['module_profile']['decode_step'])}",
        "",
        "## Measured",
        "",
        f"- prefill mean_ms: {measured['prefill']['mean_ms']:.4f}",
        f"- decode_step mean_ms: {measured['decode_step']['mean_ms']:.4f}",
        f"- request mean_ms: {measured['request']['mean_ms']:.4f}",
        "",
        "## Error",
        "",
        f"- prefill relative_error_pct: {report['comparison']['prefill_relative_error_pct']:.2f}",
        f"- decode_step relative_error_pct: {report['comparison']['decode_step_relative_error_pct']:.2f}",
        f"- request relative_error_pct: {report['comparison']['request_relative_error_pct']:.2f}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dashboard_status(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "dashboard_status.json"
    status_path.write_text(
        json.dumps(round_nested(payload), indent=2), encoding="utf-8"
    )


def write_train_reports(output_dir: Path, report: dict[str, Any]) -> None:
    """Write training estimate reports to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(round_nested(report), indent=2), encoding="utf-8")

    step = report.get("estimate", {}).get("per_step", {})
    lines = [
        "# Torch Training MVP Report",
        "",
        f"- model: `{report.get('model', {}).get('path', 'unknown')}`",
        f"- prompt tokens: {report.get('model', {}).get('prompt_tokens', 0)}",
        f"- device: `{report.get('calibration', {}).get('device_name', 'unknown')}`",
        "",
        "## Estimate (per step)",
        "",
        f"- forward_time_ms: {step.get('forward_time_ms', 0):.4f}",
        f"- backward_time_ms: {step.get('backward_time_ms', 0):.4f}",
        f"- optimizer_time_ms: {step.get('optimizer_time_ms', 0):.4f}",
        f"- total_time_ms: {step.get('total_time_ms', 0):.4f}",
        f"- samples_per_sec: {step.get('samples_per_sec', 0):.4f}",
        f"- tokens_per_sec: {step.get('tokens_per_sec', 0):.4f}" if step.get('tokens_per_sec') else "",
        "",
        "## Phase Breakdown",
        "",
        f"- forward flops: {report.get('estimate', {}).get('forward', {}).get('flops', 0):.0f}",
        f"- backward flops: {report.get('estimate', {}).get('backward', {}).get('flops', 0):.0f}",
        f"- optimizer flops: {report.get('estimate', {}).get('optimizer', {}).get('flops', 0):.0f}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_error_pct(estimate: float, measured: float) -> float:
    if measured == 0:
        return 0.0
    return abs(estimate - measured) / measured * 100.0
