from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from mvp_backend import (
    default_device_string,
    detect_accelerator_kind,
    distributed_backend,
    empty_cache,
    normalize_device_string,
    set_device,
    synchronize,
    uses_visible_device_remap,
)
from mvp_measurement import relative_error_pct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline-parallel inference benchmark MVP"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompt",
        default="Explain what a torch-based runtime estimator needs to measure.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    parser.add_argument("--device", default=default_device_string())
    parser.add_argument("--physical-devices", default="")
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeat", type=int, default=3)
    parser.add_argument("--output-dir", default="reports/torch_pipeline_mvp")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dist-timeout-minutes", type=int, default=30)
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


def duplicate_prompt_inputs(
    tokenizer: AutoTokenizer,
    prompt: str,
    microbatch_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer must define eos_token to derive pad_token")
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer([prompt] * microbatch_count, return_tensors="pt", padding=True)
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


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
    if pp_size <= 0:
        raise ValueError("pp_size must be positive")
    boundaries = [round(num_layers * index / pp_size) for index in range(pp_size + 1)]
    start = int(boundaries[stage_index])
    end = int(boundaries[stage_index + 1])
    if end <= start:
        raise ValueError(
            f"Invalid pipeline split for {num_layers} layers across pp_size={pp_size}"
        )
    return start, end


class LlamaPipelineStage(torch.nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        stage_index: int,
        pp_size: int,
    ) -> None:
        super().__init__()
        total_layers = len(model.model.layers)
        start, end = stage_layer_range(total_layers, stage_index, pp_size)
        self.stage_index = stage_index
        self.pp_size = pp_size
        self.is_first = stage_index == 0
        self.is_last = stage_index == pp_size - 1
        self.layers = torch.nn.ModuleList(list(model.model.layers[start:end]))
        self.rotary_emb = model.model.rotary_emb
        self.embed_tokens = model.model.embed_tokens if self.is_first else None
        self.norm = model.model.norm if self.is_last else None
        self.lm_head = model.lm_head if self.is_last else None

    def _position_ids(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        return torch.arange(
            sequence_length, device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = self._position_ids(hidden_states)
        causal_mask = build_causal_mask(hidden_states, attention_mask)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )[0]
        return hidden_states

    def forward(self, *args):
        if self.is_first:
            input_ids, attention_mask = args
            if self.embed_tokens is None:
                raise RuntimeError("First pipeline stage is missing embeddings")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states, attention_mask = args
        hidden_states = self._run_layers(hidden_states, attention_mask)
        if not self.is_last:
            return hidden_states, attention_mask
        if self.norm is None or self.lm_head is None:
            raise RuntimeError("Last pipeline stage is missing output modules")
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def build_stage_module(
    model_path: str,
    dtype: torch.dtype,
    stage_index: int,
    pp_size: int,
    device: torch.device,
) -> LlamaPipelineStage:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if not isinstance(model, LlamaForCausalLM):
        raise RuntimeError("Pipeline MVP currently supports LlamaForCausalLM only")
    stage_module = LlamaPipelineStage(model, stage_index=stage_index, pp_size=pp_size)
    del model
    gc.collect()
    empty_cache(device.type)
    stage_module.eval().to(device)
    return stage_module


def stage0_sample_output(
    stage_module: LlamaPipelineStage,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        hidden_states, next_attention_mask = stage_module(
            input_ids[:1],
            attention_mask[:1],
        )
    return hidden_states, next_attention_mask


def send_tensor_pair(hidden_states: torch.Tensor, attention_mask: torch.Tensor, dst: int) -> None:
    dist.send(hidden_states.contiguous(), dst=dst)
    dist.send(attention_mask.contiguous(), dst=dst)


def recv_tensor_pair(
    device: torch.device,
    hidden_shape: tuple[int, ...],
    hidden_dtype: torch.dtype,
    mask_shape: tuple[int, ...],
    mask_dtype: torch.dtype,
    src: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.empty(hidden_shape, dtype=hidden_dtype, device=device)
    attention_mask = torch.empty(mask_shape, dtype=mask_dtype, device=device)
    dist.recv(hidden_states, src=src)
    dist.recv(attention_mask, src=src)
    return hidden_states, attention_mask


def stage_timing_ms(fn, warmup: int, repeat: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(repeat):
        synchronize()
        started = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - started) * 1.0e3)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def pipeline_request_timing_ms(
    schedule: ScheduleGPipe,
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    warmup: int,
    repeat: int,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        dist.barrier()
        if rank == 0:
            schedule.step(input_ids, attention_mask)
        else:
            schedule.step()
        dist.barrier()
        synchronize()
    samples = []
    per_rank_samples = [[] for _ in range(world_size)]
    for _ in range(repeat):
        dist.barrier()
        synchronize()
        started = time.perf_counter()
        if rank == 0:
            schedule.step(input_ids, attention_mask)
        else:
            schedule.step()
        dist.barrier()
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1.0e3
        gathered: list[float | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, elapsed_ms)
        all_samples = [float(item or 0.0) for item in gathered]
        samples.append(max(all_samples))
        for sample_index, value in enumerate(all_samples):
            per_rank_samples[sample_index].append(value)
    return {
        "aggregate": {
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
        },
        "per_rank": [
            {
                "rank": rank_index,
                "mean_ms": statistics.mean(rank_samples),
                "median_ms": statistics.median(rank_samples),
                "min_ms": min(rank_samples),
                "max_ms": max(rank_samples),
                "samples_ms": rank_samples,
            }
            for rank_index, rank_samples in enumerate(per_rank_samples)
        ],
    }


def manual_two_stage_request_timing_ms(
    stage_module: LlamaPipelineStage,
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    sample_hidden_states: torch.Tensor,
    sample_attention_mask: torch.Tensor,
    warmup: int,
    repeat: int,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    ack_tensor = torch.ones((1,), dtype=torch.int32, device=sample_hidden_states.device)

    for _ in range(warmup):
        dist.barrier()
        if rank == 0:
            with torch.no_grad():
                hidden_states, next_attention_mask = stage_module(
                    input_ids[:1], attention_mask[:1]
                )
            send_tensor_pair(hidden_states, next_attention_mask, dst=1)
            dist.recv(ack_tensor, src=1)
        else:
            hidden_states, next_attention_mask = recv_tensor_pair(
                device=sample_hidden_states.device,
                hidden_shape=tuple(sample_hidden_states.shape),
                hidden_dtype=sample_hidden_states.dtype,
                mask_shape=tuple(sample_attention_mask.shape),
                mask_dtype=sample_attention_mask.dtype,
                src=0,
            )
            with torch.no_grad():
                stage_module(hidden_states, next_attention_mask)
            dist.send(ack_tensor, dst=0)
        dist.barrier()
        synchronize()
    samples = []
    per_rank_samples = [[] for _ in range(world_size)]
    for _ in range(repeat):
        dist.barrier()
        synchronize()
        started = time.perf_counter()
        if rank == 0:
            with torch.no_grad():
                hidden_states, next_attention_mask = stage_module(
                    input_ids[:1], attention_mask[:1]
                )
            send_tensor_pair(hidden_states, next_attention_mask, dst=1)
            dist.recv(ack_tensor, src=1)
        else:
            hidden_states, next_attention_mask = recv_tensor_pair(
                device=sample_hidden_states.device,
                hidden_shape=tuple(sample_hidden_states.shape),
                hidden_dtype=sample_hidden_states.dtype,
                mask_shape=tuple(sample_attention_mask.shape),
                mask_dtype=sample_attention_mask.dtype,
                src=0,
            )
            with torch.no_grad():
                stage_module(hidden_states, next_attention_mask)
            dist.send(ack_tensor, dst=0)
        dist.barrier()
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1.0e3
        gathered: list[float | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, elapsed_ms)
        all_samples = [float(item or 0.0) for item in gathered]
        samples.append(max(all_samples))
        for sample_index, value in enumerate(all_samples):
            per_rank_samples[sample_index].append(value)
    return {
        "aggregate": {
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
        },
        "per_rank": [
            {
                "rank": rank_index,
                "mean_ms": statistics.mean(rank_samples),
                "median_ms": statistics.median(rank_samples),
                "min_ms": min(rank_samples),
                "max_ms": max(rank_samples),
                "samples_ms": rank_samples,
            }
            for rank_index, rank_samples in enumerate(per_rank_samples)
        ],
    }


def pipeline_estimate_from_stage_times(
    stage_times_ms: list[float],
    microbatch_count: int,
) -> dict[str, float]:
    fill_drain_ms = sum(stage_times_ms) + max(microbatch_count - 1, 0) * max(
        stage_times_ms
    )
    return {
        "request_makespan_ms": fill_drain_ms,
        "per_request_latency_ms": fill_drain_ms / max(microbatch_count, 1),
    }


def pipeline_estimate_from_slot_time(
    slot_time_ms: float,
    microbatch_count: int,
) -> dict[str, float]:
    makespan_ms = slot_time_ms * max(microbatch_count, 1)
    return {
        "request_makespan_ms": makespan_ms,
        "per_request_latency_ms": makespan_ms / max(microbatch_count, 1),
    }


def pipeline_estimate_from_first_and_steady_slot(
    first_slot_ms: float,
    steady_slot_ms: float,
    microbatch_count: int,
) -> dict[str, float]:
    makespan_ms = first_slot_ms + max(microbatch_count - 1, 0) * steady_slot_ms
    return {
        "request_makespan_ms": makespan_ms,
        "per_request_latency_ms": makespan_ms / max(microbatch_count, 1),
    }


def single_device_request_timing_ms(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    microbatch_count: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    micro_input_ids = list(torch.tensor_split(input_ids, microbatch_count, dim=0))
    micro_attention_masks = list(
        torch.tensor_split(attention_mask, microbatch_count, dim=0)
    )

    def run_once() -> None:
        with torch.no_grad():
            for micro_input_ids_item, micro_attention_mask_item in zip(
                micro_input_ids, micro_attention_masks
            ):
                model(
                    input_ids=micro_input_ids_item,
                    attention_mask=micro_attention_mask_item,
                    use_cache=False,
                )

    return stage_timing_ms(run_once, warmup=warmup, repeat=repeat)


def single_device_stage_time_ms(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    def run_once() -> None:
        with torch.no_grad():
            model(
                input_ids=input_ids[:1],
                attention_mask=attention_mask[:1],
                use_cache=False,
            )

    return stage_timing_ms(run_once, warmup=warmup, repeat=repeat)


def write_pipeline_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Pipeline Inference MVP Report",
        "",
        f"- model: `{report['model']['path']}`",
        f"- prompt tokens: {report['model']['prompt_tokens']}",
        f"- microbatch_count: {report['execution']['microbatch_count']}",
        f"- pp_size: {report['execution']['pp_size']}",
        "",
        "## Estimate",
        "",
        f"- request_makespan_ms: {report['estimate']['request_makespan_ms']:.4f}",
        f"- per_request_latency_ms: {report['estimate']['per_request_latency_ms']:.4f}",
        "",
        "## Measured",
        "",
        f"- request_makespan_ms: {report['measured']['request']['mean_ms']:.4f}",
        f"- per_request_latency_ms: {report['measured']['per_request_latency_ms']:.4f}",
        "",
        "## Error",
        "",
        f"- request_relative_error_pct: {report['comparison']['request_relative_error_pct']:.4f}",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pp_mode(args: argparse.Namespace) -> None:
    accelerator_kind = detect_accelerator_kind(args.device)
    normalized_device = normalize_device_string(args.device, accelerator_kind)
    physical_devices = parse_physical_devices(args.physical_devices, normalized_device)
    if args.pp_size != 2:
        raise RuntimeError("Current pipeline MVP supports pp_size=2 only")
    if len(physical_devices) != args.pp_size:
        raise RuntimeError("physical_devices must contain exactly pp_size devices")
    if args.microbatch_count < 1:
        raise RuntimeError("microbatch_count must be >= 1")

    if not dist.is_initialized():
        dist.init_process_group(
            distributed_backend(accelerator_kind),
            timeout=timedelta(minutes=args.dist_timeout_minutes),
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.pp_size:
        raise RuntimeError(
            f"torchrun world_size={world_size} does not match pp_size={args.pp_size}"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    local_device = physical_devices[local_rank]
    runtime_device = local_rank if uses_visible_device_remap(accelerator_kind) else local_device
    set_device(accelerator_kind, runtime_device)
    device = torch.device(accelerator_kind, runtime_device)
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    host_batch_input_ids = None
    host_batch_attention_mask = None
    if rank == 0:
        host_batch_input_ids, host_batch_attention_mask = duplicate_prompt_inputs(
            tokenizer,
            args.prompt,
            args.microbatch_count,
            device,
        )
    prompt_tokens = len(tokenizer(args.prompt, return_tensors="pt")["input_ids"][0])

    stage_module = build_stage_module(
        args.model_path,
        dtype=dtype,
        stage_index=rank,
        pp_size=args.pp_size,
        device=device,
    )

    sample_hidden_states = None
    sample_attention_mask = None
    if rank == 0:
        sample_hidden_states, sample_attention_mask = stage0_sample_output(
            stage_module,
            host_batch_input_ids,
            host_batch_attention_mask,
        )
        shape_payload = {
            "hidden_shape": tuple(sample_hidden_states.shape),
            "hidden_dtype": sample_hidden_states.dtype,
            "mask_shape": tuple(sample_attention_mask.shape),
            "mask_dtype": sample_attention_mask.dtype,
        }
    else:
        shape_payload = None
    payload_list = [shape_payload]
    dist.broadcast_object_list(payload_list, src=0)
    shape_payload = payload_list[0] or {}
    if rank == 0:
        send_tensor_pair(sample_hidden_states, sample_attention_mask, dst=1)
    else:
        sample_hidden_states, sample_attention_mask = recv_tensor_pair(
            device=device,
            hidden_shape=tuple(shape_payload["hidden_shape"]),
            hidden_dtype=shape_payload["hidden_dtype"],
            mask_shape=tuple(shape_payload["mask_shape"]),
            mask_dtype=shape_payload["mask_dtype"],
            src=0,
        )

    if rank == 0:
        local_stage_stats = stage_timing_ms(
            lambda: stage_module(
                host_batch_input_ids[:1], host_batch_attention_mask[:1]
            ),
            warmup=args.warmup,
            repeat=args.benchmark_repeat,
        )
    else:
        local_stage_stats = stage_timing_ms(
            lambda: stage_module(sample_hidden_states, sample_attention_mask),
            warmup=args.warmup,
            repeat=args.benchmark_repeat,
        )
    gathered_stage_stats: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_stage_stats, local_stage_stats)

    slot_calibration = manual_two_stage_request_timing_ms(
        stage_module=stage_module,
        input_ids=host_batch_input_ids if rank == 0 else None,
        attention_mask=host_batch_attention_mask if rank == 0 else None,
        sample_hidden_states=sample_hidden_states,
        sample_attention_mask=sample_attention_mask,
        warmup=args.warmup,
        repeat=args.benchmark_repeat,
        rank=rank,
        world_size=world_size,
    )

    if args.microbatch_count == 1:
        measured = slot_calibration
    else:
        stage_input_args = (
            (host_batch_input_ids[:1], host_batch_attention_mask[:1])
            if rank == 0
            else (sample_hidden_states, sample_attention_mask)
        )
        stage = PipelineStage(
            stage_module,
            stage_index=rank,
            num_stages=args.pp_size,
            device=device,
            input_args=stage_input_args,
        )
        schedule = ScheduleGPipe(stage, n_microbatches=args.microbatch_count)
        measured = pipeline_request_timing_ms(
            schedule=schedule,
            input_ids=host_batch_input_ids if rank == 0 else None,
            attention_mask=host_batch_attention_mask if rank == 0 else None,
            warmup=args.warmup,
            repeat=args.benchmark_repeat,
            rank=rank,
            world_size=world_size,
        )

    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return

    stage_times_ms = [float(item["mean_ms"]) for item in gathered_stage_stats if item]
    steady_slot_ms = max(stage_times_ms)
    stage_fill_drain_estimate = pipeline_estimate_from_stage_times(
        stage_times_ms=stage_times_ms,
        microbatch_count=args.microbatch_count,
    )
    slot_based_estimate = pipeline_estimate_from_slot_time(
        slot_time_ms=float(slot_calibration["aggregate"]["mean_ms"]),
        microbatch_count=args.microbatch_count,
    )
    estimate = pipeline_estimate_from_first_and_steady_slot(
        first_slot_ms=float(slot_calibration["aggregate"]["mean_ms"]),
        steady_slot_ms=float(steady_slot_ms),
        microbatch_count=args.microbatch_count,
    )
    report = {
        "runtime_model": "torch_pipeline_prefill_v1",
        "mode": "inference",
        "model": {
            "path": args.model_path,
            "prompt": args.prompt,
            "prompt_tokens": int(prompt_tokens),
            "dtype": args.dtype,
        },
        "execution": {
            "parallel_mode": "pp",
            "accelerator_kind": accelerator_kind,
            "pp_size": args.pp_size,
            "microbatch_count": args.microbatch_count,
            "world_size": world_size,
            "physical_devices": physical_devices,
            "host_name": socket.gethostname(),
        },
        "stage_profile": {
            "per_stage": [
                {"stage_index": index, **(item or {})}
                for index, item in enumerate(gathered_stage_stats)
            ]
        },
        "estimate": {
            **estimate,
            "source": "pp_first_slot_plus_steady_stage",
            "first_slot_calibration_ms": float(slot_calibration["aggregate"]["mean_ms"]),
            "steady_slot_ms": float(steady_slot_ms),
            "slot_calibration_ms": float(slot_calibration["aggregate"]["mean_ms"]),
            "slot_based_makespan_ms": float(slot_based_estimate["request_makespan_ms"]),
            "stage_fill_drain_makespan_ms": float(
                stage_fill_drain_estimate["request_makespan_ms"]
            ),
        },
        "measured": {
            "request": measured["aggregate"],
            "per_request_latency_ms": measured["aggregate"]["mean_ms"]
            / args.microbatch_count,
            "rank_measurements": measured["per_rank"],
        },
        "comparison": {
            "request_relative_error_pct": relative_error_pct(
                estimate["per_request_latency_ms"],
                measured["aggregate"]["mean_ms"] / args.microbatch_count,
            )
        },
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    write_pipeline_report(output_dir, report)
    print(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()


def run_single_mode(args: argparse.Namespace) -> None:
    accelerator_kind = detect_accelerator_kind(args.device)
    normalized_device = normalize_device_string(args.device, accelerator_kind)
    physical_devices = parse_physical_devices(args.physical_devices, normalized_device)
    if not physical_devices:
        raise RuntimeError("single-device pipeline benchmark needs one device")
    device_index = physical_devices[0]
    runtime_device = 0 if uses_visible_device_remap(accelerator_kind) else device_index
    set_device(accelerator_kind, runtime_device)
    device = torch.device(accelerator_kind, runtime_device)
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    input_ids, attention_mask = duplicate_prompt_inputs(
        tokenizer,
        args.prompt,
        args.microbatch_count,
        device,
    )
    prompt_tokens = len(tokenizer(args.prompt, return_tensors="pt")["input_ids"][0])
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    stage_stats = single_device_stage_time_ms(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        warmup=args.warmup,
        repeat=args.benchmark_repeat,
    )
    request_stats = single_device_request_timing_ms(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        microbatch_count=args.microbatch_count,
        warmup=args.warmup,
        repeat=args.benchmark_repeat,
    )
    estimate = pipeline_estimate_from_stage_times(
        stage_times_ms=[stage_stats["mean_ms"]],
        microbatch_count=args.microbatch_count,
    )
    report = {
        "runtime_model": "torch_pipeline_prefill_v1",
        "mode": "inference",
        "model": {
            "path": args.model_path,
            "prompt": args.prompt,
            "prompt_tokens": int(prompt_tokens),
            "dtype": args.dtype,
        },
        "execution": {
            "parallel_mode": "pp",
            "accelerator_kind": accelerator_kind,
            "pp_size": 1,
            "microbatch_count": args.microbatch_count,
            "world_size": 1,
            "physical_devices": [device_index],
            "host_name": socket.gethostname(),
        },
        "stage_profile": {"per_stage": [{"stage_index": 0, **stage_stats}]},
        "estimate": estimate,
        "measured": {
            "request": request_stats,
            "per_request_latency_ms": request_stats["mean_ms"] / args.microbatch_count,
            "rank_measurements": [{"rank": 0, **request_stats}],
        },
        "comparison": {
            "request_relative_error_pct": relative_error_pct(
                estimate["per_request_latency_ms"],
                request_stats["mean_ms"] / args.microbatch_count,
            )
        },
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    write_pipeline_report(output_dir, report)
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    if args.pp_size == 1:
        run_single_mode(args)
        return
    run_pp_mode(args)


if __name__ == "__main__":
    main()
