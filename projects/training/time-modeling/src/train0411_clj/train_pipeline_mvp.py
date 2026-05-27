from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .backend import (
    default_device_string,
    detect_accelerator_kind,
    empty_cache,
    normalize_device_string,
    set_device,
    synchronize,
    uses_visible_device_remap,
)
from .profile_db import append_profile_record, load_profile_records, match_profile_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline-parallel training benchmark MVP"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--physical-devices", default="")
    parser.add_argument("--pp-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--microbatch-count", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
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
        choices=["online", "table", "hybrid"],
        default="online",
    )
    parser.add_argument(
        "--profile-db-path",
        default="database/train_component_profile_table.jsonl",
    )
    parser.add_argument("--write-profile-db", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeat", type=int, default=5)
    parser.add_argument("--profile-repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--output-dir", default="reports/torch_train_pipeline_mvp")
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


def relative_error_pct(measured_ms: float, estimated_ms: float) -> float:
    if measured_ms == 0:
        return 0.0
    return abs(measured_ms - estimated_ms) / measured_ms * 100.0


def summarize_samples(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def synchronize_devices(kind: str, devices: list[torch.device]) -> None:
    seen: set[int] = set()
    for device in devices:
        index = int(device.index or 0)
        if index in seen:
            continue
        seen.add(index)
        set_device(kind, index)
        synchronize(kind)


def empty_devices(kind: str, devices: list[torch.device]) -> None:
    seen: set[int] = set()
    for device in devices:
        index = int(device.index or 0)
        if index in seen:
            continue
        seen.add(index)
        set_device(kind, index)
        empty_cache(kind)


def resolve_runtime_devices(args: argparse.Namespace) -> tuple[str, list[int], list[torch.device]]:
    accelerator_kind = detect_accelerator_kind(args.device)
    normalized = normalize_device_string(args.device, accelerator_kind)
    physical_devices = parse_physical_devices(args.physical_devices, normalized)
    if len(physical_devices) < args.pp_size:
        raise RuntimeError(
            f"pp_size={args.pp_size} requires at least {args.pp_size} physical devices"
        )
    runtime_devices: list[torch.device] = []
    for local_index in range(args.pp_size):
        physical_index = physical_devices[local_index]
        runtime_index = (
            local_index
            if uses_visible_device_remap(accelerator_kind)
            else physical_index
        )
        runtime_devices.append(torch.device(accelerator_kind, runtime_index))
    return accelerator_kind, physical_devices, runtime_devices


def build_synthetic_microbatches(
    vocab_size: int,
    microbatch_size: int,
    microbatch_count: int,
    sequence_length: int,
    stage0_device: torch.device,
    stage1_device: torch.device | None,
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
        batch: dict[str, torch.Tensor] = {
            "input_ids_0": input_ids_cpu.to(stage0_device),
            "attention_mask_0": attention_mask_cpu.to(stage0_device),
        }
        target_device = stage1_device or stage0_device
        batch["labels_last"] = input_ids_cpu.to(target_device)
        batch["attention_mask_last"] = attention_mask_cpu.to(target_device)
        batch["class_labels_last"] = (
            input_ids_cpu[:, 0] % adapter_num_labels
        ).to(target_device)
        batches.append(batch)
    return batches


def build_causal_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length, _ = hidden_states.shape
    min_dtype = torch.finfo(hidden_states.dtype).min
    base_mask = torch.full(
        (sequence_length, sequence_length),
        min_dtype,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    base_mask = torch.triu(base_mask, diagonal=1)
    causal_mask = base_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        padding_mask = attention_mask[:, None, None, :].eq(0)
        causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)
    return causal_mask


def stage_layer_range(num_layers: int, stage_index: int, pp_size: int) -> tuple[int, int]:
    boundaries = [round(num_layers * index / pp_size) for index in range(pp_size + 1)]
    start = int(boundaries[stage_index])
    end = int(boundaries[stage_index + 1])
    if end <= start:
        raise RuntimeError(
            f"Invalid pipeline split for {num_layers} layers across pp_size={pp_size}"
        )
    return start, end


def run_llama_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    causal_mask: torch.Tensor,
    position_ids: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    enable_gradient_checkpointing: bool,
) -> torch.Tensor:
    if enable_gradient_checkpointing:
        def layer_forward(
            hidden_states_: torch.Tensor,
            causal_mask_: torch.Tensor,
            position_ids_: torch.Tensor,
            cos_: torch.Tensor,
            sin_: torch.Tensor,
        ) -> torch.Tensor:
            return layer(
                hidden_states_,
                attention_mask=causal_mask_,
                position_ids=position_ids_,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=(cos_, sin_),
            )[0]

        return checkpoint(
            layer_forward,
            hidden_states,
            causal_mask,
            position_ids,
            position_embeddings[0],
            position_embeddings[1],
            use_reentrant=False,
        )
    return layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=position_embeddings,
    )[0]


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


class LlamaTrainStage0(torch.nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        pp_size: int,
        enable_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        total_layers = len(model.model.layers)
        start, end = stage_layer_range(total_layers, 0, pp_size)
        self.layers = torch.nn.ModuleList(list(model.model.layers[start:end]))
        self.embed_tokens = model.model.embed_tokens
        self.rotary_emb = copy.deepcopy(model.model.rotary_emb)
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        freeze_parameters(self)

    def _position_ids(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        return torch.arange(
            sequence_length, device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            hidden_states = self.embed_tokens(input_ids)
            position_ids = self._position_ids(hidden_states)
            causal_mask = build_causal_mask(hidden_states, attention_mask)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer in self.layers:
                hidden_states = run_llama_layer(
                    layer=layer,
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    enable_gradient_checkpointing=False,
                )
        return hidden_states.detach(), attention_mask


class LlamaTrainStage1(torch.nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        pp_size: int,
        lora_rank: int,
        lora_alpha: float,
        adapter_num_labels: int,
        enable_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        total_layers = len(model.model.layers)
        start, end = stage_layer_range(total_layers, pp_size - 1, pp_size)
        self.layers = torch.nn.ModuleList(list(model.model.layers[start:end]))
        self.rotary_emb = copy.deepcopy(model.model.rotary_emb)
        self.norm = model.model.norm
        self.adapter = LoraClassifier(
            hidden_size=int(model.config.hidden_size),
            rank=lora_rank,
            alpha=lora_alpha,
            num_labels=adapter_num_labels,
            dtype=next(model.parameters()).dtype,
        )
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        freeze_parameters(self.layers)
        freeze_parameters(self.rotary_emb)
        freeze_parameters(self.norm)

    def _position_ids(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        return torch.arange(
            sequence_length, device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            position_ids = self._position_ids(hidden_states)
            causal_mask = build_causal_mask(hidden_states, attention_mask)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer in self.layers:
                hidden_states = run_llama_layer(
                    layer=layer,
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    enable_gradient_checkpointing=False,
                )
            hidden_states = self.norm(hidden_states)
        logits = self.adapter(hidden_states.detach(), attention_mask)
        return F.cross_entropy(logits.float(), labels)


def build_single_model(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    enable_gradient_checkpointing: bool,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    adapter_num_labels: int = 2,
) -> LlamaLoraStyleTrainModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if not isinstance(model, LlamaForCausalLM):
        raise RuntimeError("Training MVP currently supports LlamaForCausalLM only")
    model.config.use_cache = False
    train_model = LlamaLoraStyleTrainModel(
        model=model,
        rank=lora_rank,
        alpha=lora_alpha,
        num_labels=adapter_num_labels,
        dtype=dtype,
    )
    train_model.train().to(device)
    return train_model


def build_pipeline_stages(
    model_path: str,
    dtype: torch.dtype,
    devices: list[torch.device],
    enable_gradient_checkpointing: bool,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    adapter_num_labels: int = 2,
) -> tuple[LlamaTrainStage0, LlamaTrainStage1]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if not isinstance(model, LlamaForCausalLM):
        raise RuntimeError("Training MVP currently supports LlamaForCausalLM only")
    model.config.use_cache = False
    stage0 = LlamaTrainStage0(
        model,
        pp_size=2,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    ).train().to(devices[0])
    stage1 = LlamaTrainStage1(
        model,
        pp_size=2,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        adapter_num_labels=adapter_num_labels,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    ).train().to(devices[1])
    del model
    gc.collect()
    return stage0, stage1


def zero_grad_optimizers(optimizers: list[torch.optim.Optimizer | None]) -> None:
    for optimizer in optimizers:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def step_optimizers(optimizers: list[torch.optim.Optimizer | None]) -> None:
    for optimizer in optimizers:
        if optimizer is not None:
            optimizer.step()


def build_optimizer(
    parameters,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
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
    if args.optimizer_type == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
            foreach=args.optimizer_foreach,
        )
    raise RuntimeError(f"Unsupported optimizer_type={args.optimizer_type}")


def build_trainable_optimizer(
    module: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer | None:
    parameters = list(trainable_parameters(module))
    if not parameters:
        return None
    return build_optimizer(parameters, args)


def measure_generic(
    fn,
    warmup: int,
    repeat: int,
    accelerator_kind: str,
    devices: list[torch.device],
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    synchronize_devices(accelerator_kind, devices)
    samples_ms: list[float] = []
    for _ in range(repeat):
        synchronize_devices(accelerator_kind, devices)
        started = time.perf_counter()
        fn()
        synchronize_devices(accelerator_kind, devices)
        samples_ms.append((time.perf_counter() - started) * 1.0e3)
    return summarize_samples(samples_ms)


def measure_optimizer_step(
    prepare_grads_fn,
    optimizer_step_fn,
    warmup: int,
    repeat: int,
    accelerator_kind: str,
    devices: list[torch.device],
) -> dict[str, Any]:
    for _ in range(warmup):
        prepare_grads_fn()
        synchronize_devices(accelerator_kind, devices)
        optimizer_step_fn()
        synchronize_devices(accelerator_kind, devices)
    samples_ms: list[float] = []
    for _ in range(repeat):
        prepare_grads_fn()
        synchronize_devices(accelerator_kind, devices)
        started = time.perf_counter()
        optimizer_step_fn()
        synchronize_devices(accelerator_kind, devices)
        samples_ms.append((time.perf_counter() - started) * 1.0e3)
    return summarize_samples(samples_ms)


def build_profile_match_fields(
    model_config: Any,
    accelerator_kind: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    match_fields = {
        "runtime_model": "torch_pipeline_train_v1",
        "training_mode": "lora_style_adapter",
        "accelerator_kind": accelerator_kind,
        "model_type": getattr(model_config, "model_type", "unknown"),
        "num_hidden_layers": int(getattr(model_config, "num_hidden_layers", 0)),
        "hidden_size": int(getattr(model_config, "hidden_size", 0)),
        "intermediate_size": int(getattr(model_config, "intermediate_size", 0)),
        "num_attention_heads": int(getattr(model_config, "num_attention_heads", 0)),
        "vocab_size": int(getattr(model_config, "vocab_size", 0)),
        "dtype": args.dtype,
        "pp_size": args.pp_size,
        "microbatch_size": args.microbatch_size,
        "sequence_length": args.sequence_length,
        "optimizer": args.optimizer_type,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "adapter_num_labels": args.adapter_num_labels,
    }
    if args.optimizer_foreach:
        match_fields["optimizer_foreach"] = True
    if args.enable_gradient_checkpointing:
        match_fields["gradient_checkpointing"] = True
    return match_fields

def make_profile_record(
    component: str,
    mean_ms: float,
    match_fields: dict[str, Any],
) -> dict[str, Any]:
    record = dict(match_fields)
    record.update(
        {
            "record_type": "training_component_profile",
            "component": component,
            "mean_ms": mean_ms,
            "unit": "ms",
        }
    )
    return record


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
    record = make_profile_record(component, stats["mean_ms"], match_fields)
    if args.write_profile_db:
        append_profile_record(args.profile_db_path, record)
    return {
        "component": component,
        "source": "online",
        "mean_ms": stats["mean_ms"],
        "stats": stats,
        "record": record,
    }


def load_existing_component_profile(
    component: str,
    args: argparse.Namespace,
    match_fields: dict[str, Any],
) -> dict[str, Any] | None:
    records = load_profile_records(args.profile_db_path)
    matched = match_profile_record(records, component, match_fields)
    if matched is None:
        return None
    return {
        "component": component,
        "source": "table",
        "mean_ms": float(matched["mean_ms"]),
        "record": matched,
    }


def persist_derived_component_profile(
    component: str,
    mean_ms: float,
    args: argparse.Namespace,
    match_fields: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = make_profile_record(component, mean_ms, match_fields)
    if extra_fields:
        record.update(extra_fields)
    if args.write_profile_db:
        append_profile_record(args.profile_db_path, record)
    return record


def single_microbatch_slot(
    model: LlamaLoraStyleTrainModel,
    batch: dict[str, torch.Tensor],
    microbatch_count: int,
) -> torch.Tensor:
    loss = model(
        input_ids=batch["input_ids_0"],
        attention_mask=batch["attention_mask_0"],
        labels=batch["class_labels_last"],
    )
    loss = loss / max(microbatch_count, 1)
    loss.backward()
    return loss.detach()


def pipeline_microbatch_slot(
    stage0: LlamaTrainStage0,
    stage1: LlamaTrainStage1,
    batch: dict[str, torch.Tensor],
    device1: torch.device,
    microbatch_count: int,
) -> torch.Tensor:
    hidden_states, attention_mask = stage0(
        batch["input_ids_0"],
        batch["attention_mask_0"],
    )
    hidden_states = hidden_states.to(device1)
    attention_mask = attention_mask.to(device1)
    loss = stage1(hidden_states, attention_mask, batch["class_labels_last"])
    loss = loss / max(microbatch_count, 1)
    loss.backward()
    return loss.detach()


def run_single_mode(
    args: argparse.Namespace,
    model_config: Any,
    dtype: torch.dtype,
    accelerator_kind: str,
    devices: list[torch.device],
) -> dict[str, Any]:
    device = devices[0]
    model = build_single_model(
        args.model_path,
        dtype=dtype,
        device=device,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapter_num_labels=args.adapter_num_labels,
    )
    optimizer = build_trainable_optimizer(model, args)
    batches = build_synthetic_microbatches(
        vocab_size=model.config.vocab_size,
        microbatch_size=args.microbatch_size,
        microbatch_count=args.microbatch_count,
        sequence_length=args.sequence_length,
        stage0_device=device,
        stage1_device=None,
        seed=args.seed,
        adapter_num_labels=args.adapter_num_labels,
    )

    def run_iteration() -> None:
        zero_grad_optimizers([optimizer])
        for batch in batches:
            single_microbatch_slot(model, batch, args.microbatch_count)
        step_optimizers([optimizer])

    sample_batch = batches[0]
    match_fields = build_profile_match_fields(model_config, accelerator_kind, args)

    def measure_slot_profile() -> dict[str, Any]:
        def slot_once() -> None:
            zero_grad_optimizers([optimizer])
            single_microbatch_slot(model, sample_batch, args.microbatch_count)

        return measure_generic(
            slot_once,
            warmup=1,
            repeat=args.profile_repeat,
            accelerator_kind=accelerator_kind,
            devices=devices,
        )

    def measure_optimizer_profile() -> dict[str, Any]:
        def prepare_grads() -> None:
            zero_grad_optimizers([optimizer])
            single_microbatch_slot(model, sample_batch, args.microbatch_count)

        def step_once() -> None:
            step_optimizers([optimizer])

        return measure_optimizer_step(
            prepare_grads,
            step_once,
            warmup=1,
            repeat=args.profile_repeat,
            accelerator_kind=accelerator_kind,
            devices=devices,
        )

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
    estimate_ms = (
        args.microbatch_count * float(slot_profile["mean_ms"])
        + float(optimizer_profile["mean_ms"])
    )
    measured_stats = measure_generic(
        run_iteration,
        warmup=args.warmup,
        repeat=args.benchmark_repeat,
        accelerator_kind=accelerator_kind,
        devices=devices,
    )
    return {
        "measured": measured_stats,
        "estimate": {
            "train_iteration_time_ms": estimate_ms,
            "microbatch_slot_ms": float(slot_profile["mean_ms"]),
            "microbatch_steady_slot_ms": float(slot_profile["mean_ms"]),
            "optimizer_step_ms": float(optimizer_profile["mean_ms"]),
            "slot_source": slot_profile["source"],
            "steady_slot_source": slot_profile["source"],
            "optimizer_source": optimizer_profile["source"],
        },
        "profile_records": [slot_profile.get("record"), optimizer_profile.get("record")],
    }


def run_pipeline_mode(
    args: argparse.Namespace,
    model_config: Any,
    dtype: torch.dtype,
    accelerator_kind: str,
    devices: list[torch.device],
) -> dict[str, Any]:
    stage0, stage1 = build_pipeline_stages(
        args.model_path,
        dtype=dtype,
        devices=devices,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapter_num_labels=args.adapter_num_labels,
    )
    optimizer0 = build_trainable_optimizer(stage0, args)
    optimizer1 = build_trainable_optimizer(stage1, args)
    optimizers = [optimizer0, optimizer1]
    batches = build_synthetic_microbatches(
        vocab_size=model_config.vocab_size,
        microbatch_size=args.microbatch_size,
        microbatch_count=args.microbatch_count,
        sequence_length=args.sequence_length,
        stage0_device=devices[0],
        stage1_device=devices[1],
        seed=args.seed,
        adapter_num_labels=args.adapter_num_labels,
    )

    def run_iteration() -> None:
        zero_grad_optimizers(optimizers)
        for batch in batches:
            pipeline_microbatch_slot(
                stage0,
                stage1,
                batch,
                device1=devices[1],
                microbatch_count=args.microbatch_count,
            )
        step_optimizers(optimizers)

    sample_batch = batches[0]
    match_fields = build_profile_match_fields(model_config, accelerator_kind, args)

    def measure_slot_profile() -> dict[str, Any]:
        def slot_once() -> None:
            zero_grad_optimizers(optimizers)
            pipeline_microbatch_slot(
                stage0,
                stage1,
                sample_batch,
                device1=devices[1],
                microbatch_count=args.microbatch_count,
            )

        return measure_generic(
            slot_once,
            warmup=1,
            repeat=args.profile_repeat,
            accelerator_kind=accelerator_kind,
            devices=devices,
        )

    def measure_optimizer_profile() -> dict[str, Any]:
        def prepare_grads() -> None:
            zero_grad_optimizers(optimizers)
            pipeline_microbatch_slot(
                stage0,
                stage1,
                sample_batch,
                device1=devices[1],
                microbatch_count=args.microbatch_count,
            )

        def step_once() -> None:
            step_optimizers(optimizers)

        return measure_optimizer_step(
            prepare_grads,
            step_once,
            warmup=1,
            repeat=args.profile_repeat,
            accelerator_kind=accelerator_kind,
            devices=devices,
        )

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
    measured_stats = measure_generic(
        run_iteration,
        warmup=args.warmup,
        repeat=args.benchmark_repeat,
        accelerator_kind=accelerator_kind,
        devices=devices,
    )
    first_slot_ms = float(slot_profile["mean_ms"])
    optimizer_step_ms = float(optimizer_profile["mean_ms"])
    steady_slot_profile = None
    steady_slot_ms = first_slot_ms
    steady_slot_source = slot_profile["source"]
    if args.microbatch_count > 1:
        steady_slot_profile = load_existing_component_profile(
            "microbatch_steady_slot",
            args,
            match_fields,
        )
        if steady_slot_profile is not None:
            steady_slot_ms = float(steady_slot_profile["mean_ms"])
            steady_slot_source = steady_slot_profile["source"]
        elif args.estimate_mode != "table" and args.microbatch_count == 2:
            derived_steady_slot_ms = max(
                measured_stats["mean_ms"] - first_slot_ms - optimizer_step_ms,
                0.0,
            )
            steady_record = persist_derived_component_profile(
                "microbatch_steady_slot",
                derived_steady_slot_ms,
                args,
                match_fields,
                extra_fields={"derived_from_microbatch_count": 2},
            )
            steady_slot_profile = {
                "component": "microbatch_steady_slot",
                "source": "online_derived",
                "mean_ms": derived_steady_slot_ms,
                "record": steady_record,
            }
            steady_slot_ms = derived_steady_slot_ms
            steady_slot_source = "online_derived"
    estimate_ms = (
        first_slot_ms
        + max(args.microbatch_count - 1, 0) * steady_slot_ms
        + optimizer_step_ms
    )
    return {
        "measured": measured_stats,
        "estimate": {
            "train_iteration_time_ms": estimate_ms,
            "microbatch_slot_ms": first_slot_ms,
            "microbatch_steady_slot_ms": steady_slot_ms,
            "optimizer_step_ms": optimizer_step_ms,
            "slot_source": slot_profile["source"],
            "steady_slot_source": steady_slot_source,
            "optimizer_source": optimizer_profile["source"],
        },
        "profile_records": [
            slot_profile.get("record"),
            steady_slot_profile.get("record") if steady_slot_profile else None,
            optimizer_profile.get("record"),
        ],
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Training MVP Report",
        "",
        f"- model_path: {report['model']['path']}",
        f"- pp_size: {report['execution']['pp_size']}",
        f"- microbatch_count: {report['execution']['microbatch_count']}",
        f"- microbatch_size: {report['execution']['microbatch_size']}",
        f"- sequence_length: {report['execution']['sequence_length']}",
        f"- optimizer_type: {report['execution']['optimizer_type']}",
        f"- optimizer_foreach: {report['execution']['optimizer_foreach']}",
        f"- gradient_checkpointing: {report['execution']['gradient_checkpointing']}",
        f"- training_mode: {report['execution']['training_mode']}",
        f"- lora_rank: {report['execution']['lora_rank']}",
        f"- lora_alpha: {report['execution']['lora_alpha']}",
        f"- adapter_num_labels: {report['execution']['adapter_num_labels']}",
        f"- measured_train_iteration_ms: {report['measured']['train_iteration_time_ms']:.6f}",
        f"- estimated_train_iteration_ms: {report['estimate']['train_iteration_time_ms']:.6f}",
        f"- error_pct: {report['error_pct']:.6f}",
        "",
        "## Estimate Components",
        "",
        f"- microbatch_slot_ms: {report['estimate']['microbatch_slot_ms']:.6f}",
        f"- microbatch_steady_slot_ms: {report['estimate']['microbatch_steady_slot_ms']:.6f}",
        f"- optimizer_step_ms: {report['estimate']['optimizer_step_ms']:.6f}",
        f"- slot_source: {report['estimate']['slot_source']}",
        f"- steady_slot_source: {report['estimate']['steady_slot_source']}",
        f"- optimizer_source: {report['estimate']['optimizer_source']}",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.microbatch_count < 1:
        raise RuntimeError("microbatch_count must be >= 1")
    if args.microbatch_size < 1:
        raise RuntimeError("microbatch_size must be >= 1")
    accelerator_kind, physical_devices, devices = resolve_runtime_devices(args)
    dtype = dtype_from_name(args.dtype)
    model_config = AutoConfig.from_pretrained(args.model_path)
    run_name = f"pp{args.pp_size}_mb{args.microbatch_count}"
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.pp_size == 1:
        result = run_single_mode(args, model_config, dtype, accelerator_kind, [devices[0]])
    else:
        result = run_pipeline_mode(args, model_config, dtype, accelerator_kind, devices[:2])

    measured_ms = float(result["measured"]["mean_ms"])
    estimated_ms = float(result["estimate"]["train_iteration_time_ms"])
    report = {
        "runtime_model": "torch_pipeline_train_v1",
        "model": {
            "path": str(Path(args.model_path).expanduser()),
            "model_type": getattr(model_config, "model_type", "unknown"),
            "num_hidden_layers": int(getattr(model_config, "num_hidden_layers", 0)),
            "hidden_size": int(getattr(model_config, "hidden_size", 0)),
            "intermediate_size": int(getattr(model_config, "intermediate_size", 0)),
            "num_attention_heads": int(getattr(model_config, "num_attention_heads", 0)),
            "vocab_size": int(getattr(model_config, "vocab_size", 0)),
        },
        "execution": {
            "accelerator_kind": accelerator_kind,
            "training_mode": "lora_style_adapter",
            "dtype": args.dtype,
            "pp_size": args.pp_size,
            "microbatch_count": args.microbatch_count,
            "microbatch_size": args.microbatch_size,
            "sequence_length": args.sequence_length,
            "physical_devices": physical_devices[: args.pp_size],
            "estimate_mode": args.estimate_mode,
            "optimizer_type": args.optimizer_type,
            "optimizer_foreach": bool(args.optimizer_foreach),
            "gradient_checkpointing": bool(args.enable_gradient_checkpointing),
            "backbone_frozen": True,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "adapter_num_labels": args.adapter_num_labels,
        },
        "measured": {
            "train_iteration_time_ms": measured_ms,
            **result["measured"],
        },
        "estimate": result["estimate"],
        "error_pct": relative_error_pct(measured_ms, estimated_ms),
        "profile_db_path": str(Path(args.profile_db_path).expanduser()),
        "profile_records": [record for record in result["profile_records"] if record],
    }
    write_report(output_dir / run_name, report)
    empty_devices(accelerator_kind, devices[: args.pp_size])
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
