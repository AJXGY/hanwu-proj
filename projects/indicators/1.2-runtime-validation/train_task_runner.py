from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cambricon training adaptation run wrapper"
    )
    parser.add_argument("--train-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="mlu:0")
    parser.add_argument("--physical-devices", default="0")
    parser.add_argument("--pp-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer-type", choices=["adamw", "sgd"], default="sgd")
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument("--optimizer-foreach", action="store_true")
    parser.add_argument("--enable-gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--adapter-num-labels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sparkline_svg(values: list[float], title: str) -> str:
    width = 960
    height = 360
    margin = 48
    if not values:
        values = [0.0]
    min_v = min(values)
    max_v = max(values)
    if math.isclose(min_v, max_v):
        min_v -= 1.0
        max_v += 1.0
    step_x = (width - 2 * margin) / max(len(values) - 1, 1)

    def x_pos(index: int) -> float:
        return margin + index * step_x

    def y_pos(value: float) -> float:
        scale = (value - min_v) / (max_v - min_v)
        return height - margin - scale * (height - 2 * margin)

    polyline = " ".join(
        f"{x_pos(idx):.2f},{y_pos(value):.2f}" for idx, value in enumerate(values)
    )
    dots = "\n".join(
        f'<circle cx="{x_pos(idx):.2f}" cy="{y_pos(value):.2f}" r="5" fill="#0f766e" />'
        for idx, value in enumerate(values)
    )
    labels = "\n".join(
        f'<text x="{x_pos(idx):.2f}" y="{height - margin + 24}" text-anchor="middle" font-size="12" fill="#6b7280">iter {idx + 1}</text>'
        for idx, _ in enumerate(values)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{margin}" y="28" font-size="24" font-family="sans-serif" fill="#111827">{title}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#9ca3af" stroke-width="2"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#9ca3af" stroke-width="2"/>
<polyline fill="none" stroke="#1d4ed8" stroke-width="4" points="{polyline}"/>
{dots}
{labels}
<text x="{margin - 12}" y="{margin}" text-anchor="end" font-size="12" fill="#6b7280">{max_v:.4f}</text>
<text x="{margin - 12}" y="{height - margin}" text-anchor="end" font-size="12" fill="#6b7280">{min_v:.4f}</text>
</svg>
"""


def main() -> None:
    args = parse_args()
    train_repo = Path(args.train_repo).expanduser().resolve()
    sys.path.insert(0, str(train_repo / "src"))

    from transformers import AutoConfig

    from train0411_clj.backend import (
        detect_accelerator_kind,
        empty_cache,
        normalize_device_string,
        set_device,
        synchronize,
        uses_visible_device_remap,
    )
    from train0411_clj.train_pipeline_mvp import (
        build_trainable_optimizer,
        build_pipeline_stages,
        build_single_model,
        build_synthetic_microbatches,
        dtype_from_name,
        parse_physical_devices,
        pipeline_microbatch_slot,
        single_microbatch_slot,
        step_optimizers,
        zero_grad_optimizers,
    )

    def resolve_runtime_devices() -> tuple[str, list[int], list[torch.device]]:
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
                local_index if uses_visible_device_remap(accelerator_kind) else physical_index
            )
            runtime_devices.append(torch.device(accelerator_kind, runtime_index))
        return accelerator_kind, physical_devices, runtime_devices

    def sync_devices(kind: str, devices: list[torch.device]) -> None:
        seen: set[int] = set()
        for device in devices:
            index = int(device.index or 0)
            if index in seen:
                continue
            seen.add(index)
            set_device(kind, index)
            synchronize(kind)

    def clean_devices(kind: str, devices: list[torch.device]) -> None:
        seen: set[int] = set()
        for device in devices:
            index = int(device.index or 0)
            if index in seen:
                continue
            seen.add(index)
            set_device(kind, index)
            empty_cache(kind)

    def parameter_digest(module: torch.nn.Module) -> dict[str, list[float]]:
        digest: dict[str, list[float]] = {}
        for name, tensor in module.state_dict().items():
            flat = tensor.detach().float().reshape(-1).cpu()
            digest[name] = flat[: min(8, flat.numel())].tolist()
            if len(digest) >= 4:
                break
        return digest

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_config = AutoConfig.from_pretrained(args.model_path)
    dtype = dtype_from_name(args.dtype)
    accelerator_kind, physical_devices, devices = resolve_runtime_devices()
    started = time.time()

    losses: list[float] = []
    iteration_times_ms: list[float] = []
    trainable_parameter_count = 0
    checkpoint_artifacts: list[str] = []

    if args.pp_size == 1:
        model = build_single_model(
            args.model_path,
            dtype=dtype,
            device=devices[0],
            enable_gradient_checkpointing=args.enable_gradient_checkpointing,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            adapter_num_labels=args.adapter_num_labels,
        )
        optimizer = build_trainable_optimizer(model, args)
        trainable_parameter_count = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        batches = build_synthetic_microbatches(
            vocab_size=model.config.vocab_size,
            microbatch_size=args.microbatch_size,
            microbatch_count=args.microbatch_count,
            sequence_length=args.sequence_length,
            stage0_device=devices[0],
            stage1_device=None,
            seed=args.seed,
            adapter_num_labels=args.adapter_num_labels,
        )
        for iteration_idx in range(args.iterations):
            zero_grad_optimizers([optimizer])
            sync_devices(accelerator_kind, [devices[0]])
            iter_started = time.perf_counter()
            microbatch_losses: list[float] = []
            for batch in batches:
                loss = single_microbatch_slot(model, batch, args.microbatch_count)
                microbatch_losses.append(float(loss.item()))
            step_optimizers([optimizer])
            sync_devices(accelerator_kind, [devices[0]])
            iteration_times_ms.append((time.perf_counter() - iter_started) * 1.0e3)
            losses.append(statistics.mean(microbatch_losses))
        torch.save(
            {
                "kind": "single_device_lightweight_checkpoint",
                "training_mode": "lora_style_adapter",
                "model_path": args.model_path,
                "pp_size": 1,
                "iterations": args.iterations,
                "loss_history": losses,
                "adapter_state_dict": model.adapter.state_dict(),
                "adapter_parameter_digest": parameter_digest(model.adapter),
            },
            checkpoint_dir / "lora_style_adapter_checkpoint.pt",
        )
        checkpoint_artifacts.append(str(checkpoint_dir / "lora_style_adapter_checkpoint.pt"))
    else:
        stage0, stage1 = build_pipeline_stages(
            args.model_path,
            dtype=dtype,
            devices=devices[:2],
            enable_gradient_checkpointing=args.enable_gradient_checkpointing,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            adapter_num_labels=args.adapter_num_labels,
        )
        optimizer0 = build_trainable_optimizer(stage0, args)
        optimizer1 = build_trainable_optimizer(stage1, args)
        optimizers = [optimizer0, optimizer1]
        trainable_parameter_count = sum(
            param.numel() for module in [stage0, stage1] for param in module.parameters() if param.requires_grad
        )
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
        for iteration_idx in range(args.iterations):
            zero_grad_optimizers(optimizers)
            sync_devices(accelerator_kind, devices[:2])
            iter_started = time.perf_counter()
            microbatch_losses: list[float] = []
            for batch in batches:
                loss = pipeline_microbatch_slot(
                    stage0,
                    stage1,
                    batch,
                    device1=devices[1],
                    microbatch_count=args.microbatch_count,
                )
                microbatch_losses.append(float(loss.item()))
            step_optimizers(optimizers)
            sync_devices(accelerator_kind, devices[:2])
            iteration_times_ms.append((time.perf_counter() - iter_started) * 1.0e3)
            losses.append(statistics.mean(microbatch_losses))
        torch.save(
            {
                "kind": "pipeline_stage0_lightweight_checkpoint",
                "training_mode": "lora_style_adapter",
                "model_path": args.model_path,
                "pp_size": 2,
                "iterations": args.iterations,
                "loss_history": losses,
                "parameter_digest": parameter_digest(stage0),
            },
            checkpoint_dir / "stage0_checkpoint.pt",
        )
        torch.save(
            {
                "kind": "pipeline_stage1_adapter_checkpoint",
                "training_mode": "lora_style_adapter",
                "model_path": args.model_path,
                "pp_size": 2,
                "iterations": args.iterations,
                "loss_history": losses,
                "adapter_state_dict": stage1.adapter.state_dict(),
                "adapter_parameter_digest": parameter_digest(stage1.adapter),
            },
            checkpoint_dir / "stage1_lora_style_adapter_checkpoint.pt",
        )
        checkpoint_artifacts.extend(
            [
                str(checkpoint_dir / "stage0_checkpoint.pt"),
                str(checkpoint_dir / "stage1_lora_style_adapter_checkpoint.pt"),
            ]
        )

    finished = time.time()
    write_json(output_dir / "loss_history.json", {"loss_history": losses, "iteration_times_ms": iteration_times_ms})
    (output_dir / "loss_curve.svg").write_text(
        sparkline_svg(losses, "Training Loss Curve"),
        encoding="utf-8",
    )
    summary = {
        "task": "training",
        "success": len(losses) == args.iterations and len(checkpoint_artifacts) > 0,
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": finished - started,
        "model_path": args.model_path,
        "pp_size": args.pp_size,
        "physical_devices": physical_devices[: args.pp_size],
        "microbatch_count": args.microbatch_count,
        "microbatch_size": args.microbatch_size,
        "sequence_length": args.sequence_length,
        "iterations": args.iterations,
        "optimizer_type": args.optimizer_type,
        "gradient_checkpointing": bool(args.enable_gradient_checkpointing),
        "training_mode": "lora_style_adapter",
        "backbone_frozen": True,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "adapter_num_labels": args.adapter_num_labels,
        "loss_history": losses,
        "iteration_times_ms": iteration_times_ms,
        "final_loss": losses[-1] if losses else None,
        "mean_iteration_time_ms": statistics.mean(iteration_times_ms) if iteration_times_ms else None,
        "trainable_parameter_count": trainable_parameter_count,
        "checkpoint_artifacts": checkpoint_artifacts,
        "loss_curve_path": str(output_dir / "loss_curve.svg"),
        "note": "Frozen Llama backbone forward with LoRA-style low-rank adapter update.",
    }
    write_json(output_dir / "summary.json", summary)
    clean_devices(accelerator_kind, devices[: args.pp_size])
    gc.collect()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
