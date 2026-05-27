from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mvp_train_estimator import estimate_model_architecture


class TrainWrapper(nn.Module):
    """Wrapper for training mode that returns loss."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits


def prepare_train_inputs(
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare inputs for training."""
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Labels are the same as input_ids for causal LM
    labels = input_ids.clone()

    return input_ids, attention_mask, labels


def extract_train_graph(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Extract training computation graph using torch.export."""
    model.eval()

    # Create wrapper that returns loss
    train_wrapper = TrainWrapper(model)

    # Export training graph
    train_export = torch.export.export(
        train_wrapper,
        (input_ids, attention_mask, labels),
    )

    return {
        "train_export": train_export,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collect_train_module_info(
    model: nn.Module,
) -> dict[str, Any]:
    """Collect training-related module information from the model."""
    module_info = {
        "num_layers": 0,
        "layer_names": [],
        "has_gradient_checkpointing": False,
        "optimizer_type": "adam",
    }

    # Count transformer layers
    for name, _ in model.named_modules():
        if "layer" in name.lower() and "h." in name:
            if name not in module_info["layer_names"]:
                module_info["layer_names"].append(name)
        # Check for gradient checkpointing
        if "gradient_checkpointing" in name:
            module_info["has_gradient_checkpointing"] = True

    # Try to determine number of layers from config
    if hasattr(model, 'config'):
        config = model.config
        module_info["num_layers"] = getattr(config, 'num_hidden_layers', 0)
        module_info["hidden_size"] = getattr(config, 'hidden_size', 0)
        module_info["num_heads"] = getattr(config, 'num_attention_heads', 0)
        module_info["vocab_size"] = getattr(config, 'vocab_size', 0)

    return module_info


def benchmark_forward_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    warmup: int = 2,
    repeat: int = 5,
) -> dict[str, float]:
    """Measure forward + backward pass time."""
    model.train()

    times = []

    for _ in range(warmup):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss.backward()

    model.zero_grad()

    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss.backward()

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

        model.zero_grad()

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "samples_ms": times,
    }


def benchmark_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup: int = 2,
    repeat: int = 5,
) -> dict[str, float]:
    """Measure optimizer step time."""
    times = []

    # Pre-setup: create gradients
    for _ in range(warmup):
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.randn_like(p)
        optimizer.step()

    model.zero_grad()

    for _ in range(repeat):
        # Create fresh gradients
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.randn_like(p)

        torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.step()

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

        optimizer.zero_grad()

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "samples_ms": times,
    }


def run_train_step_measurement(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    warmup: int = 2,
    repeat: int = 5,
) -> dict[str, Any]:
    """Run complete training step measurement."""
    results = {}

    # Forward + backward
    fb_results = benchmark_forward_backward(
        model, input_ids, attention_mask, labels, warmup, repeat
    )
    results["forward_backward"] = fb_results

    # Optimizer step
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    opt_results = benchmark_optimizer_step(model, optimizer, warmup, repeat)
    results["optimizer"] = opt_results

    # Combined
    results["combined_mean_ms"] = (
        fb_results["mean_ms"] + opt_results["mean_ms"]
    )
    results["combined_median_ms"] = (
        fb_results["median_ms"] + opt_results["median_ms"]
    )

    return results
