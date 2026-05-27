from __future__ import annotations

import argparse
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist

from mvp_backend import (
    default_device_string,
    detect_accelerator_kind,
    distributed_backend,
    local_topology,
    normalize_device_string,
    set_device,
    uses_visible_device_remap,
)
from mvp_types import ExecutionConfig, RankPlacement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch-based inference prediction MVP")
    parser.add_argument("--model-path", default="/opt/models/Llama-3.2-1B")
    parser.add_argument(
        "--prompt",
        default="Explain what a torch-based runtime estimator needs to measure.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    parser.add_argument("--parallel-mode", choices=["single", "tp"], default="single")
    parser.add_argument("--physical-devices", default="")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--interconnect",
        choices=["auto", "ethernet", "infiniband", "roce"],
        default="auto",
    )
    parser.add_argument("--collective-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--collective-latency-ms", type=float, default=None)
    parser.add_argument("--dist-timeout-minutes", type=int, default=30)
    parser.add_argument("--device", default=default_device_string())
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--estimate-mode",
        choices=["online", "table", "hybrid"],
        default="online",
        help=(
            "online: always collect module profiles from runtime; "
            "table: only load module profiles from table DB; "
            "hybrid: load from table first and collect only missing scopes"
        ),
    )
    parser.add_argument(
        "--table-db-path",
        default="database/module_profile_table.jsonl",
        help="Path to module profile table database (jsonl)",
    )
    parser.add_argument(
        "--table-writeback",
        action="store_true",
        help="Append online-collected module profiles into table DB",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--benchmark-repeat", type=int, default=5)
    parser.add_argument("--profile-repeat", type=int, default=10)
    parser.add_argument("--output-dir", default="reports/torch_mvp")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def parse_physical_devices(raw_value: str, fallback_device: str) -> list[int]:
    text = (raw_value or "").strip()
    if text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if fallback_device.startswith(("cuda:", "mlu:")):
        return [int(fallback_device.split(":", 1)[1])]
    return [0]


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def resolve_interconnect(raw_value: str, nnodes: int) -> str:
    if nnodes <= 1:
        return "local"
    if raw_value and raw_value != "auto":
        return raw_value
    socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "").lower()
    if socket_ifname.startswith("ib"):
        return "infiniband"
    if "roce" in socket_ifname:
        return "roce"
    if os.environ.get("NCCL_IB_DISABLE") == "1":
        return "ethernet"
    return "ethernet"


def gather_rank_placements(
    rank: int,
    host_name: str,
    node_rank: int,
    local_rank: int,
    physical_device: int,
    world_size: int,
) -> list[RankPlacement]:
    local_payload = {
        "rank": rank,
        "host": host_name,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "physical_device": physical_device,
    }
    if world_size <= 1 or not dist.is_initialized():
        return [RankPlacement(**local_payload)]
    gathered: list[dict[str, int | str] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_payload)
    placements = [RankPlacement(**item) for item in gathered if isinstance(item, dict)]
    placements.sort(key=lambda item: item.rank)
    return placements


def resolve_execution_config(
    args: argparse.Namespace,
) -> tuple[ExecutionConfig, torch.device]:
    accelerator_kind = detect_accelerator_kind(args.device)
    normalized_device = normalize_device_string(args.device, accelerator_kind)
    physical_devices = parse_physical_devices(args.physical_devices, normalized_device)
    visible_devices = ",".join(str(device) for device in physical_devices)
    host_name = socket.gethostname()
    local_topology_name = local_topology(accelerator_kind, physical_devices)
    if args.parallel_mode == "tp":
        if args.world_size != args.tp_size:
            raise ValueError("tp mode currently requires world_size to equal tp_size")
        if not dist.is_initialized():
            dist.init_process_group(
                distributed_backend(accelerator_kind),
                timeout=timedelta(minutes=args.dist_timeout_minutes),
            )
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world_size = env_int("LOCAL_WORLD_SIZE", args.nproc_per_node)
        node_rank = env_int("GROUP_RANK", args.node_rank)
        world_size = dist.get_world_size()
        if world_size != args.world_size:
            raise ValueError(
                f"torchrun world_size={world_size} does not match --world-size={args.world_size}"
            )
        nnodes = max(args.nnodes, world_size // max(local_world_size, 1))
        if len(physical_devices) <= local_rank:
            raise ValueError(
                "the local physical device list must cover every local rank on this host"
            )
        local_device = physical_devices[local_rank]
        runtime_device = (
            local_rank
            if uses_visible_device_remap(accelerator_kind)
            else local_device
        )
        set_device(accelerator_kind, runtime_device)
        device = torch.device(accelerator_kind, runtime_device)
        placements = gather_rank_placements(
            rank=rank,
            host_name=host_name,
            node_rank=node_rank,
            local_rank=local_rank,
            physical_device=local_device,
            world_size=world_size,
        )
        interconnect = resolve_interconnect(args.interconnect, nnodes)
        topology = (
            f"inter_host_{interconnect}" if nnodes > 1 else local_topology_name
        )
        execution = ExecutionConfig(
            accelerator_kind=accelerator_kind,
            parallel_mode="tp",
            physical_devices=physical_devices,
            visible_devices=visible_devices,
            world_size=world_size,
            tp_size=args.tp_size,
            topology=topology,
            local_topology=local_topology_name,
            interconnect=interconnect,
            nnodes=nnodes,
            nproc_per_node=local_world_size,
            host_name=host_name,
            master_addr=os.environ.get("MASTER_ADDR", args.master_addr),
            master_port=env_int("MASTER_PORT", args.master_port),
            local_device=local_device,
            placements=placements,
            collective_bandwidth_gbps=args.collective_bandwidth_gbps,
            collective_latency_ms=args.collective_latency_ms,
            rank=rank,
            local_rank=local_rank,
            node_rank=node_rank,
        )
        return execution, device

    local_device = physical_devices[0]
    if normalized_device.startswith(f"{accelerator_kind}:") and uses_visible_device_remap(
        accelerator_kind
    ):
        runtime_device = int(normalized_device.split(":", 1)[1])
    else:
        runtime_device = local_device
    set_device(accelerator_kind, runtime_device)
    device = torch.device(accelerator_kind, runtime_device)
    placements = [
        RankPlacement(
            rank=0,
            host=host_name,
            node_rank=0,
            local_rank=0,
            physical_device=local_device,
        )
    ]
    execution = ExecutionConfig(
        accelerator_kind=accelerator_kind,
        parallel_mode="single",
        physical_devices=physical_devices,
        visible_devices=visible_devices,
        world_size=1,
        tp_size=1,
        topology=local_topology_name,
        local_topology=local_topology_name,
        interconnect="local",
        nnodes=1,
        nproc_per_node=1,
        host_name=host_name,
        master_addr=args.master_addr,
        master_port=args.master_port,
        local_device=local_device,
        placements=placements,
        collective_bandwidth_gbps=args.collective_bandwidth_gbps,
        collective_latency_ms=args.collective_latency_ms,
        rank=0,
        local_rank=0,
        node_rank=0,
    )
    return execution, device
