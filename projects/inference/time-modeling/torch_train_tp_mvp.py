from __future__ import annotations

import argparse
import gc
import json
import os
import random
import socket
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from mvp_backend import (
    default_device_string,
    detect_accelerator_kind,
    distributed_backend,
    empty_cache,
    local_topology,
    normalize_device_string,
    set_device,
    synchronize,
    uses_visible_device_remap,
)

try:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
except ImportError:  # pragma: no cover
    init_device_mesh = None
    ColwiseParallel = None
    RowwiseParallel = None
    parallelize_module = None


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
    rank: int
    local_rank: int
    node_rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch TP training benchmark MVP")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    parser.add_argument("--device", default=default_device_string())
    parser.add_argument("--parallel-mode", choices=["single", "tp"], default="tp")
    parser.add_argument("--physical-devices", default="")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dist-timeout-minutes", type=int, default=30)
    parser.add_argument("--microbatch-count", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer-type", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument("--optimizer-foreach", action="store_true")
    parser.add_argument("--enable-gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--adapter-num-labels", type=int, default=2)
    parser.add_argument(
        "--estimate-mode",
        choices=["online", "table"],
        default="online",
    )
    parser.add_argument(
        "--profile-db-path",
        default="database/train_component_profile_tp.jsonl",
    )
    parser.add_argument("--write-profile-db", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeat", type=int, default=3)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--output-dir", default="reports/torch_train_tp_mvp")
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


def is_primary_rank(execution: ExecutionConfig) -> bool:
    return execution.rank == 0


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
            local_rank if uses_visible_device_remap(accelerator_kind) else local_device
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
        execution = ExecutionConfig(
            accelerator_kind=accelerator_kind,
            parallel_mode="tp",
            physical_devices=physical_devices,
            visible_devices=visible_devices,
            world_size=world_size,
            tp_size=args.tp_size,
            topology=local_topology_name,
            local_topology=local_topology_name,
            interconnect="local",
            nnodes=nnodes,
            nproc_per_node=local_world_size,
            host_name=host_name,
            master_addr=os.environ.get("MASTER_ADDR", args.master_addr),
            master_port=env_int("MASTER_PORT", args.master_port),
            local_device=local_device,
            placements=placements,
            rank=rank,
            local_rank=local_rank,
            node_rank=node_rank,
        )
        return execution, device

    local_device = physical_devices[0]
    if normalized_device.startswith(
        f"{accelerator_kind}:"
    ) and uses_visible_device_remap(accelerator_kind):
        runtime_device = int(normalized_device.split(":", 1)[1])
    else:
        runtime_device = local_device
    set_device(accelerator_kind, runtime_device)
    device = torch.device(accelerator_kind, runtime_device)
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
        placements=[
            RankPlacement(
                rank=0,
                host=host_name,
                node_rank=0,
                local_rank=0,
                physical_device=local_device,
            )
        ],
        rank=0,
        local_rank=0,
        node_rank=0,
    )
    return execution, device


def summarize_samples(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def distributed_wall_time_ms(
    fn,
    warmup: int,
    repeat: int,
    execution: ExecutionConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if execution.parallel_mode == "single":
        for _ in range(warmup):
            fn()
        synchronize(execution.accelerator_kind)
        samples_ms: list[float] = []
        for _ in range(repeat):
            synchronize(execution.accelerator_kind)
            started = time.perf_counter()
            fn()
            synchronize(execution.accelerator_kind)
            samples_ms.append((time.perf_counter() - started) * 1.0e3)
        stats = summarize_samples(samples_ms)
        return stats, [{"rank": 0, "device": execution.local_device, **stats}]

    for _ in range(warmup):
        dist.barrier()
        fn()
        synchronize(execution.accelerator_kind)
    dist.barrier()
    gathered_samples: list[list[float]] = []
    for _ in range(repeat):
        dist.barrier()
        synchronize(execution.accelerator_kind)
        started = time.perf_counter()
        fn()
        synchronize(execution.accelerator_kind)
        elapsed_ms = (time.perf_counter() - started) * 1.0e3
        gathered: list[float | None] = [None for _ in range(execution.world_size)]
        dist.all_gather_object(gathered, elapsed_ms)
        gathered_samples.append([float(item or 0.0) for item in gathered])
    aggregate_samples = [max(sample) for sample in gathered_samples]
    per_rank_samples = list(map(list, zip(*gathered_samples)))
    rank_measurements: list[dict[str, Any]] = []
    for placement, samples in zip(execution.placements, per_rank_samples):
        rank_measurements.append(
            {
                "rank": placement.rank,
                "host": placement.host,
                "node_rank": placement.node_rank,
                "local_rank": placement.local_rank,
                "device": placement.physical_device,
                **summarize_samples(samples),
            }
        )
    return summarize_samples(aggregate_samples), rank_measurements


def load_profile_records(db_path: str | Path) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        records.append(json.loads(text))
    return records


def append_profile_record(db_path: str | Path, record: dict[str, Any]) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def match_profile_record(
    records: list[dict[str, Any]], component: str, match_fields: dict[str, Any]
) -> dict[str, Any] | None:
    matched = None
    for record in records:
        if record.get("component") != component:
            continue
        if any(record.get(key) != value for key, value in match_fields.items()):
            continue
        matched = record
    return matched


def relative_error_pct(measured_ms: float, estimated_ms: float) -> float:
    if measured_ms == 0:
        return 0.0
    return abs(measured_ms - estimated_ms) / measured_ms * 100.0


def apply_tensor_parallel(
    model: torch.nn.Module, accelerator_kind: str, tp_size: int
) -> Any:
    if parallelize_module is None or init_device_mesh is None:
        raise RuntimeError("Tensor parallel APIs are unavailable in this PyTorch build")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "tp mode currently expects a Llama-style model.model.layers layout"
        )
    mesh = init_device_mesh(accelerator_kind, (tp_size,))
    plan = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in model.model.layers:
        parallelize_module(layer, mesh, plan)
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "num_heads"):
                layer.self_attn.num_heads = max(1, layer.self_attn.num_heads // tp_size)
            if hasattr(layer.self_attn, "num_key_value_heads"):
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // tp_size
                )
    return mesh


def build_optimizer(parameters, args: argparse.Namespace) -> torch.optim.Optimizer:
    parameters = list(parameters)
    if not parameters:
        raise RuntimeError("No trainable parameters available for optimizer")
    if args.optimizer_type == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=args.optimizer_foreach,
        )
    return torch.optim.SGD(
        parameters,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,
        weight_decay=args.weight_decay,
        foreach=args.optimizer_foreach,
    )


def freeze_parameters(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def trainable_parameters(module: torch.nn.Module):
    return (parameter for parameter in module.parameters() if parameter.requires_grad)


class LoraClassifier(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        alpha: float,
        num_labels: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise RuntimeError("lora_rank must be >= 1")
        if num_labels < 2:
            raise RuntimeError("adapter_num_labels must be >= 2")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.down = torch.nn.Linear(hidden_size, rank, bias=False, dtype=dtype)
        self.up = torch.nn.Linear(rank, num_labels, bias=False, dtype=dtype)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.up.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.to(hidden_states.device).unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden_states * mask).sum(dim=1) / denom
        return self.up(self.down(pooled)) * self.scaling


class LlamaLoraStyleTrainModel(torch.nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        rank: int,
        alpha: float,
        num_labels: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.backbone = model.model
        freeze_parameters(self.backbone)
        self.adapter = LoraClassifier(
            hidden_size=int(model.config.hidden_size),
            rank=rank,
            alpha=alpha,
            num_labels=num_labels,
            dtype=dtype,
        )
        self.config = model.config

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.adapter.train(mode)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = self.adapter(outputs.last_hidden_state.detach(), attention_mask)
        return F.cross_entropy(logits.float(), labels)


def build_synthetic_microbatches(
    vocab_size: int,
    microbatch_size: int,
    microbatch_count: int,
    sequence_length: int,
    device: torch.device,
    seed: int,
    adapter_num_labels: int = 2,
) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches: list[dict[str, torch.Tensor]] = []
    for _ in range(microbatch_count):
        input_ids_cpu = torch.randint(
            low=0,
            high=max(vocab_size - 1, 1),
            size=(microbatch_size, sequence_length),
            generator=generator,
            dtype=torch.long,
        )
        attention_mask_cpu = torch.ones_like(input_ids_cpu, dtype=torch.long)
        batches.append(
            {
                "input_ids": input_ids_cpu.to(device),
                "attention_mask": attention_mask_cpu.to(device),
                "labels": input_ids_cpu.to(device),
                "class_labels": (input_ids_cpu[:, 0] % adapter_num_labels).to(device),
            }
        )
    return batches


def zero_grad(optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)


def run_train_microbatch(
    model: LlamaLoraStyleTrainModel,
    batch: dict[str, torch.Tensor],
    microbatch_count: int,
) -> torch.Tensor:
    loss = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["class_labels"],
    )
    loss = loss / max(microbatch_count, 1)
    loss.backward()
    return loss.detach()


def build_profile_match_fields(
    model_config: Any,
    execution: ExecutionConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = {
        "runtime_model": "torch_tp_train_v1",
        "training_mode": "lora_style_adapter",
        "accelerator_kind": execution.accelerator_kind,
        "model_type": getattr(model_config, "model_type", "unknown"),
        "num_hidden_layers": int(getattr(model_config, "num_hidden_layers", 0)),
        "hidden_size": int(getattr(model_config, "hidden_size", 0)),
        "intermediate_size": int(getattr(model_config, "intermediate_size", 0)),
        "num_attention_heads": int(getattr(model_config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(model_config, "vocab_size", 0)),
        "dtype": args.dtype,
        "parallel_mode": execution.parallel_mode,
        "tp_size": execution.tp_size,
        "microbatch_size": args.microbatch_size,
        "sequence_length": args.sequence_length,
        "optimizer": args.optimizer_type,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "adapter_num_labels": args.adapter_num_labels,
    }
    if args.optimizer_foreach:
        fields["optimizer_foreach"] = True
    if args.enable_gradient_checkpointing:
        fields["gradient_checkpointing"] = True
    return fields


def resolve_component_profile(
    component: str,
    measure_fn,
    args: argparse.Namespace,
    match_fields: dict[str, Any],
) -> dict[str, Any]:
    records = load_profile_records(args.profile_db_path)
    matched = match_profile_record(records, component, match_fields)
    if matched is not None:
        return {
            "component": component,
            "source": "table",
            "mean_ms": float(matched["mean_ms"]),
            "record": matched,
        }
    if args.estimate_mode == "table":
        raise RuntimeError(
            f"Missing profile component '{component}' in table {args.profile_db_path}"
        )
    stats = measure_fn()
    record = {
        **match_fields,
        "record_type": "training_component_profile",
        "component": component,
        "mean_ms": stats["mean_ms"],
        "unit": "ms",
    }
    if args.write_profile_db and is_primary_rank(current_execution):
        append_profile_record(args.profile_db_path, record)
    return {
        "component": component,
        "source": "online",
        "mean_ms": stats["mean_ms"],
        "stats": stats,
        "record": record,
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# TP Training MVP Report",
        "",
        f"- model_path: {report['model']['path']}",
        f"- accelerator_kind: {report['execution']['accelerator_kind']}",
        f"- parallel_mode: {report['execution']['parallel_mode']}",
        f"- tp_size: {report['execution']['tp_size']}",
        f"- microbatch_count: {report['execution']['microbatch_count']}",
        f"- microbatch_size: {report['execution']['microbatch_size']}",
        f"- sequence_length: {report['execution']['sequence_length']}",
        f"- optimizer_type: {report['execution']['optimizer_type']}",
        f"- training_mode: {report['execution']['training_mode']}",
        f"- lora_rank: {report['execution']['lora_rank']}",
        f"- lora_alpha: {report['execution']['lora_alpha']}",
        f"- adapter_num_labels: {report['execution']['adapter_num_labels']}",
        f"- measured_train_iteration_ms: {report['measured']['train_iteration_time_ms']:.6f}",
        f"- estimated_train_iteration_ms: {report['estimate']['train_iteration_time_ms']:.6f}",
        f"- error_pct: {report['comparison']['train_iteration_relative_error_pct']:.6f}",
        "",
        "## Estimate Components",
        "",
        f"- microbatch_slot_ms: {report['estimate']['microbatch_slot_ms']:.6f}",
        f"- optimizer_step_ms: {report['estimate']['optimizer_step_ms']:.6f}",
        f"- slot_source: {report['estimate']['slot_source']}",
        f"- optimizer_source: {report['estimate']['optimizer_source']}",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


current_execution: ExecutionConfig


def main() -> None:
    global current_execution

    args = parse_args()
    if args.microbatch_count < 1:
        raise RuntimeError("microbatch_count must be >= 1")
    if args.microbatch_size < 1:
        raise RuntimeError("microbatch_size must be >= 1")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    execution, device = resolve_execution_config(args)
    current_execution = execution
    dtype = dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        model_config = AutoConfig.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if not isinstance(model, LlamaForCausalLM):
            raise RuntimeError(
                "TP training MVP currently supports LlamaForCausalLM only"
            )
        model.config.use_cache = False
        model.eval().to(device)

        if execution.parallel_mode == "tp":
            apply_tensor_parallel(model, execution.accelerator_kind, execution.tp_size)
            dist.barrier()

        train_model = LlamaLoraStyleTrainModel(
            model=model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            num_labels=args.adapter_num_labels,
            dtype=dtype,
        )
        train_model.train().to(device)

        optimizer = build_optimizer(trainable_parameters(train_model), args)
        batches = build_synthetic_microbatches(
            vocab_size=train_model.config.vocab_size,
            microbatch_size=args.microbatch_size,
            microbatch_count=args.microbatch_count,
            sequence_length=args.sequence_length,
            device=device,
            seed=args.seed,
            adapter_num_labels=args.adapter_num_labels,
        )
        sample_batch = batches[0]

        def run_iteration() -> None:
            zero_grad(optimizer)
            for batch in batches:
                run_train_microbatch(train_model, batch, args.microbatch_count)
            optimizer.step()

        def measure_slot_profile() -> dict[str, Any]:
            def slot_once() -> None:
                zero_grad(optimizer)
                run_train_microbatch(train_model, sample_batch, args.microbatch_count)

            stats, _ = distributed_wall_time_ms(
                slot_once,
                warmup=1,
                repeat=args.profile_repeat,
                execution=execution,
            )
            return stats

        def measure_optimizer_profile() -> dict[str, Any]:
            def step_once() -> None:
                zero_grad(optimizer)
                run_train_microbatch(train_model, sample_batch, args.microbatch_count)
                synchronize(execution.accelerator_kind)
                started = time.perf_counter()
                optimizer.step()
                synchronize(execution.accelerator_kind)
                elapsed_ms = (time.perf_counter() - started) * 1.0e3
                gathered: list[float | None]
                if execution.parallel_mode == "single":
                    samples_ms.append(elapsed_ms)
                    return
                gathered = [None for _ in range(execution.world_size)]
                dist.all_gather_object(gathered, elapsed_ms)
                samples_ms.append(max(float(item or 0.0) for item in gathered))

            samples_ms: list[float] = []
            for _ in range(1):
                zero_grad(optimizer)
                run_train_microbatch(train_model, sample_batch, args.microbatch_count)
                synchronize(execution.accelerator_kind)
                optimizer.step()
                synchronize(execution.accelerator_kind)
            for _ in range(args.profile_repeat):
                if execution.parallel_mode != "single":
                    dist.barrier()
                step_once()
            return summarize_samples(samples_ms)

        match_fields = build_profile_match_fields(model_config, execution, args)
        slot_profile = resolve_component_profile(
            "microbatch_slot",
            measure_slot_profile,
            args,
            match_fields,
        )
        optimizer_profile = resolve_component_profile(
            "optimizer_step",
            measure_optimizer_profile,
            args,
            match_fields,
        )
        measured_stats, rank_measurements = distributed_wall_time_ms(
            run_iteration,
            warmup=args.warmup,
            repeat=args.benchmark_repeat,
            execution=execution,
        )
        estimate_ms = args.microbatch_count * float(slot_profile["mean_ms"]) + float(
            optimizer_profile["mean_ms"]
        )

        report = {
            "runtime_model": "torch_tp_train_v1",
            "mode": "training",
            "model": {
                "path": str(Path(args.model_path).expanduser()),
                "model_type": getattr(model_config, "model_type", "unknown"),
                "num_hidden_layers": int(getattr(model_config, "num_hidden_layers", 0)),
                "hidden_size": int(getattr(model_config, "hidden_size", 0)),
                "intermediate_size": int(getattr(model_config, "intermediate_size", 0)),
                "num_attention_heads": int(
                    getattr(model_config, "num_attention_heads", 0)
                ),
                "vocab_size": int(getattr(model_config, "vocab_size", 0)),
            },
            "execution": {
                "accelerator_kind": execution.accelerator_kind,
                "training_mode": "lora_style_adapter",
                "backbone_frozen": True,
                "parallel_mode": execution.parallel_mode,
                "physical_devices": execution.physical_devices,
                "visible_devices": execution.visible_devices,
                "world_size": execution.world_size,
                "tp_size": execution.tp_size,
                "topology": execution.topology,
                "local_topology": execution.local_topology,
                "nnodes": execution.nnodes,
                "nproc_per_node": execution.nproc_per_node,
                "host_name": execution.host_name,
                "master_addr": execution.master_addr,
                "master_port": execution.master_port,
                "placements": [asdict(item) for item in execution.placements],
                "dtype": args.dtype,
                "microbatch_count": args.microbatch_count,
                "microbatch_size": args.microbatch_size,
                "sequence_length": args.sequence_length,
                "estimate_mode": args.estimate_mode,
                "optimizer_type": args.optimizer_type,
                "optimizer_foreach": bool(args.optimizer_foreach),
                "gradient_checkpointing": bool(args.enable_gradient_checkpointing),
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "adapter_num_labels": args.adapter_num_labels,
            },
            "measured": {
                "train_iteration_time_ms": measured_stats["mean_ms"],
                "forward_backward_optimizer": measured_stats,
            },
            "estimate": {
                "train_iteration_time_ms": estimate_ms,
                "microbatch_slot_ms": float(slot_profile["mean_ms"]),
                "optimizer_step_ms": float(optimizer_profile["mean_ms"]),
                "slot_source": slot_profile["source"],
                "optimizer_source": optimizer_profile["source"],
            },
            "comparison": {
                "train_iteration_relative_error_pct": relative_error_pct(
                    measured_stats["mean_ms"], estimate_ms
                )
            },
            "rank_measurements": {"train_iteration": rank_measurements},
            "profile_db_path": str(Path(args.profile_db_path).expanduser()),
            "profile_records": [
                record
                for record in [
                    slot_profile.get("record"),
                    optimizer_profile.get("record"),
                ]
                if record
            ],
        }

        if not is_primary_rank(execution):
            return
        write_report(output_dir, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        empty_cache(execution.accelerator_kind)
        gc.collect()
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
