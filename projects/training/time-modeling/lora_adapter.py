"""
LoRA-style Adapter Module for LLM Fine-tuning

Provides a simple LoRA-style low-rank vocab adapter head on top of a
backbone LLM. The optimizer updates only adapter parameters, but autograd
still traverses the full backbone graph so the backward workload matches
the train-infer-estimation training path.

Usage:
    from lora_adapter import LlamaLoRAModel, create_lora_model

    base_model = AutoModelForCausalLM.from_pretrained(...)
    lora_model = LlamaLoRAModel(base_model, adapter_rank=16)
    # optimizer should receive lora_model.trainable_parameters()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAHead(nn.Module):
    """Low-rank adapter head: hidden_size -> rank -> vocab_size."""

    def __init__(self, hidden_size: int, vocab_size: int, rank: int = 16):
        super().__init__()
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, vocab_size, bias=False)
        self._rank = rank
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(hidden_states))

    @property
    def adapter_rank(self) -> int:
        return self._rank


class LlamaLoRAModel(nn.Module):
    """Llama backbone with LoRA adapter head.

    The forward pass runs the backbone with autograd enabled, then applies
    a token-level vocab adapter head. Output is compatible with existing
    code expecting .logits and .loss attributes.

    Args:
        base_model: HuggingFace LlamaForCausalLM (or compatible)
        adapter_rank: Rank of the LoRA bottleneck (default 16)
    """

    def __init__(self, base_model: nn.Module, adapter_rank: int = 16):
        super().__init__()
        self.base_model = base_model
        # NOTE: Do NOT set requires_grad=False on backbone parameters here.
        # The backward pass needs to compute gradients through the backbone
        # (autograd traverses the full graph). Only adapter params are passed
        # to the optimizer via trainable_parameters().

        config = base_model.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        # Determine dtype from base model
        base_dtype = next(base_model.parameters()).dtype

        self.lora_head = LoRAHead(hidden_size, vocab_size, adapter_rank)
        self.lora_head = self.lora_head.to(dtype=base_dtype)
        self._adapter_rank = adapter_rank

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("use_cache", False)
        backbone_out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = backbone_out.hidden_states[-1]
        logits = self.lora_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return _ModelOutput(backbone_out=backbone_out, logits=logits, loss=loss)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return only trainable parameters (adapter head)."""
        return list(self.lora_head.parameters())

    @property
    def adapter_param_count(self) -> int:
        """Number of trainable adapter parameters."""
        return sum(p.numel() for p in self.lora_head.parameters())

    @property
    def backbone_param_count(self) -> int:
        """Number of backbone parameters."""
        return sum(p.numel() for p in self.base_model.parameters())

    @property
    def config(self):
        return self.base_model.config

    @property
    def adapter_rank(self) -> int:
        return self._adapter_rank


class _ModelOutput:
    """Output wrapper with .logits attribute for compatibility, passing through backbone attrs."""

    def __init__(self, backbone_out, logits: torch.Tensor, loss: torch.Tensor | None = None):
        self.logits = logits
        self.loss = loss
        self.hidden_states = getattr(backbone_out, "hidden_states", None)
        self.past_key_values = getattr(backbone_out, "past_key_values", None)
        self.attentions = getattr(backbone_out, "attentions", None)


def create_lora_model(
    base_model: nn.Module,
    adapter_rank: int = 16,
) -> LlamaLoRAModel:
    """Create a LoRA-wrapped model from a base model.

    Args:
        base_model: HuggingFace LlamaForCausalLM (or compatible)
        adapter_rank: Rank of the LoRA bottleneck

    Returns:
        LlamaLoRAModel with a trainable vocab adapter head
    """
    return LlamaLoRAModel(base_model, adapter_rank)
