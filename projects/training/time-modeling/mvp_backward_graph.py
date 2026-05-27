"""
Backward Graph Extraction for Training TP Prediction

This module provides backward graph extraction capabilities to support
Task 2 (分层Backward通信模型) and Task 4 (Optimizer延迟-带宽模型).

Since torch.export.export() does not support DTensor (used by TP models),
we use NON-TP models for graph extraction and capture gradient information
for communication modeling.

================================================================================
API ASSUMPTIONS (from other tasks):
================================================================================
- Task 2 (mvp_train_estimator.py) will use:
  - BackwardGraphInfo produced by this module
  - estimate_backward_comm_time() signature from TP.md

- Task 4 (mvp_train_estimator.py) will use:
  - BackwardGraphInfo.gradient_info for optimizer communication modeling
  - estimate_optimizer_tp_overhead() signature from TP.md

- Task 3 (mvp_graph.py reuse):
  - tp_shard_node_estimate() and tp_parallel_time_scale() are already available
  - These will be used for forward pass TP scaling
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class GradientInfo:
    """Information about a gradient tensor for TP communication modeling.

    Attributes:
        name: Identifier for the gradient source (e.g., layer name, parameter name)
        shape: Shape of the gradient tensor
        dtype: Data type of the gradient tensor
        numel: Number of elements in the gradient
        bytes: Size in bytes of the gradient tensor
        requires_tp_allreduce: Whether this gradient requires TP AllReduce
                                (True for parameters in TP parallel scopes)
    """
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    bytes: int
    requires_tp_allreduce: bool = False


@dataclass
class BackwardNodeInfo:
    """Backward graph node information capturing computation and communication.

    Attributes:
        module_scope: Module scope (e.g., "model.layers.0.self_attn")
        op_family: Operation family ("gemm", "attention", "pointwise", etc.)
        phase: Always "backward" for backward nodes
        forward_node_name: Name of corresponding forward node (if identifiable)
        output_gradient_info: Gradient of this node's output (for communication)
        param_gradient_infos: List of parameter gradients produced by this backward node
        input_gradient_infos: List of input gradients consumed by this backward node
        is_tp_parallel: Whether this node operates in TP parallel scope
    """
    module_scope: str
    op_family: str
    phase: str = "backward"
    forward_node_name: str = ""
    output_gradient_info: GradientInfo | None = None
    param_gradient_infos: list[GradientInfo] = field(default_factory=list)
    input_gradient_infos: list[GradientInfo] = field(default_factory=list)
    is_tp_parallel: bool = False


@dataclass
class BackwardGraphInfo:
    """Complete backward graph information for TP communication modeling.

    This is the main output of extract_backward_graph() and provides
    all gradient information needed for Task 2 (backward communication model)
    and Task 4 (optimizer communication model).

    Attributes:
        model_name: Name/identifier of the model
        num_layers: Number of transformer layers
        hidden_size: Hidden size dimension
        gradient_infos: List of all gradient tensors and their metadata
        node_infos: List of backward node information (optional, for detailed analysis)
        total_gradient_bytes: Total bytes of all gradients
        tp_gradient_bytes: Total bytes of gradients requiring TP AllReduce
        embedding_gradient_bytes: Gradient bytes for embedding/LM head (requires AllReduce)
        kernel_launch_count: Estimated number of backward kernel launches
    """
    model_name: str
    num_layers: int
    hidden_size: int
    gradient_infos: list[GradientInfo] = field(default_factory=list)
    node_infos: list[BackwardNodeInfo] = field(default_factory=list)
    total_gradient_bytes: int = 0
    tp_gradient_bytes: int = 0
    embedding_gradient_bytes: int = 0
    kernel_launch_count: int = 0


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to string representation."""
    return str(dtype).replace("torch.", "")


def _compute_gradient_bytes(tensor: torch.Tensor) -> int:
    """Compute bytes size of a gradient tensor."""
    return tensor.numel() * tensor.element_size()


def _is_tp_parallel_scope(scope: str) -> bool:
    """Check if a module scope is in a TP parallel region.

    TP parallel scopes include self_attn and mlp modules.

    NOTE: This is a simplified version. The actual is_tp_parallel_scope()
    from mvp_graph.py will be used when integrating with Task 2/3.
    """
    return ".self_attn" in scope or ".mlp" in scope


def _extract_gradients_by_hook(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Extract gradients by registering hooks on the model.

    This approach registers forward and backward hooks to capture
    intermediate gradients during the backward pass.

    Returns:
        Tuple of (forward_activations, output_gradients) where:
        - forward_activations: Dict mapping scope -> activation tensor
        - output_gradients: Dict mapping scope -> gradient tensor
    """
    gradients: dict[str, torch.Tensor] = {}
    activations: dict[str, torch.Tensor] = {}

    def forward_hook(scope: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[scope] = output.detach()
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        activations[f"{scope}.{i}"] = o.detach()
        return hook

    def backward_hook(scope: str):
        def hook(module, grad_input, grad_output):
            if grad_output is not None and len(grad_output) > 0:
                if isinstance(grad_output[0], torch.Tensor):
                    gradients[scope] = grad_output[0].detach()
                elif isinstance(grad_output, torch.Tensor):
                    gradients[scope] = grad_output.detach()
        return hook

    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            handles.append(module.register_forward_hook(forward_hook(name)))
            handles.append(module.register_full_backward_hook(backward_hook(name)))

    # Forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]

    # Backward pass (create a dummy loss for gradient capture)
    # Sum of logits for a specific token position as loss
    dummy_loss = logits.sum()

    # Clear previous gradients
    model.zero_grad()

    # Backward pass to trigger hooks
    dummy_loss.backward()

    # Remove hooks
    for handle in handles:
        handle.remove()

    return activations, gradients


def _analyze_gradient_flow(
    gradients: dict[str, torch.Tensor],
    activations: dict[str, torch.Tensor],
    model: nn.Module,
) -> list[GradientInfo]:
    """Analyze gradient flow to determine which gradients require TP AllReduce.

    In TP (Tensor Parallelism):
    - Parameters in column-wise parallel scopes (q_proj, k_proj, v_proj, gate_proj, up_proj)
      produce gradients that need to be AllReduced across TP ranks
    - Parameters in row-wise parallel scopes (o_proj, down_proj) have local gradients
    - Embedding and LayerNorm gradients also typically need AllReduce

    Returns:
        List of GradientInfo for each significant gradient tensor
    """
    gradient_infos = []

    # TP parallel scopes that require AllReduce
    tp_allreduce_scopes = {
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
        ".mlp.gate_proj",
        ".mlp.up_proj",
        ".mlp.down_proj",
        "embed_tokens",
        "lm_head",
    }

    def requires_allreduce(scope: str) -> bool:
        for tp_scope in tp_allreduce_scopes:
            if scope.endswith(tp_scope):
                return True
        return False

    # Process gradients from hooks
    for scope, grad in gradients.items():
        if grad is None or not isinstance(grad, torch.Tensor):
            continue
        if grad.numel() == 0:
            continue

        # Skip tiny gradients (numerical artifacts)
        if grad.numel() < 4:
            continue

        gradient_infos.append(GradientInfo(
            name=scope,
            shape=tuple(grad.shape),
            dtype=_dtype_to_str(grad.dtype),
            numel=grad.numel(),
            bytes=_compute_gradient_bytes(grad),
            requires_tp_allreduce=requires_allreduce(scope),
        ))

    return gradient_infos


def extract_backward_graph(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    model_name: str = "unknown",
) -> BackwardGraphInfo:
    """Extract backward graph information for training TP prediction.

    This function extracts gradient information needed for Task 2
    (分层Backward通信模型) and Task 4 (Optimizer延迟-带宽模型).

    Since torch.export.export() does not support DTensor, we use
    NON-TP models and capture gradient information via hooks.

    The key insight is that for TP communication modeling, we need:
    1. Per-layer gradient tensor sizes (for AllReduce bandwidth estimation)
    2. Which gradients require TP AllReduce (for total communication volume)
    3. Kernel launch count (for computational overhead)

    Args:
        model: The NON-TP model (e.g., LlamaForCausalLM without TP wrapping)
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        model_name: Optional name identifier for the model

    Returns:
        BackwardGraphInfo containing:
        - gradient_infos: List of all gradient GradientInfo objects
        - total_gradient_bytes: Sum of all gradient sizes
        - tp_gradient_bytes: Sum of gradients requiring TP AllReduce
        - embedding_gradient_bytes: Gradient bytes for embedding/LM head
        - kernel_launch_count: Estimated backward kernel launches

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from mvp_backward_graph import extract_backward_graph
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        >>> input_ids = torch.randint(0, 32000, (1, 128))
        >>> attn_mask = torch.ones(1, 128)
        >>> bg_info = extract_backward_graph(model, input_ids, attn_mask)
        >>> print(f"Total gradient bytes: {bg_info.total_gradient_bytes}")
        >>> print(f"TP gradient bytes: {bg_info.tp_gradient_bytes}")
    """
    # Extract gradients via hooks
    activations, gradients = _extract_gradients_by_hook(
        model, input_ids, attention_mask
    )

    # Analyze gradient flow
    gradient_infos = _analyze_gradient_flow(gradients, activations, model)

    # Calculate totals
    total_gradient_bytes = sum(gi.bytes for gi in gradient_infos)
    tp_gradient_bytes = sum(gi.bytes for gi in gradient_infos if gi.requires_tp_allreduce)
    embedding_gradient_bytes = sum(
        gi.bytes for gi in gradient_infos
        if "embed" in gi.name.lower() or "lm_head" in gi.name.lower()
    )

    # Estimate kernel launch count (based on gradient count and model size)
    # Each gradient tensor typically requires 1-3 kernel launches
    kernel_launch_count = len(gradient_infos) * 2

    # Extract layer count and hidden size from model
    num_layers = 0
    hidden_size = 0

    # Try to infer from model architecture
    if hasattr(model, "config"):
        config = model.config
        num_layers = getattr(config, "num_hidden_layers", 0)
        hidden_size = getattr(config, "hidden_size", 0)

    # Fallback: count transformer layers from model
    if num_layers == 0:
        for name, _ in model.named_modules():
            if ".layers." in name or ".h." in name:
                # Extract layer number
                for part in name.split("."):
                    if part.isdigit():
                        num_layers = max(num_layers, int(part) + 1)

    return BackwardGraphInfo(
        model_name=model_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        gradient_infos=gradient_infos,
        total_gradient_bytes=total_gradient_bytes,
        tp_gradient_bytes=tp_gradient_bytes,
        embedding_gradient_bytes=embedding_gradient_bytes,
        kernel_launch_count=kernel_launch_count,
    )


def get_gradient_summary(bg_info: BackwardGraphInfo) -> dict[str, Any]:
    """Get a summary dict of gradient information for logging/debugging.

    Args:
        bg_info: BackwardGraphInfo from extract_backward_graph()

    Returns:
        Dictionary with summary information
    """
    tp_gradients = [gi for gi in bg_info.gradient_infos if gi.requires_tp_allreduce]
    non_tp_gradients = [gi for gi in bg_info.gradient_infos if not gi.requires_tp_allreduce]

    return {
        "model_name": bg_info.model_name,
        "num_layers": bg_info.num_layers,
        "hidden_size": bg_info.hidden_size,
        "total_gradient_count": len(bg_info.gradient_infos),
        "tp_gradient_count": len(tp_gradients),
        "non_tp_gradient_count": len(non_tp_gradients),
        "total_gradient_bytes": bg_info.total_gradient_bytes,
        "tp_gradient_bytes": bg_info.tp_gradient_bytes,
        "embedding_gradient_bytes": bg_info.embedding_gradient_bytes,
        "kernel_launch_count": bg_info.kernel_launch_count,
        "top_tp_gradients": sorted(
            [(gi.name, gi.bytes) for gi in tp_gradients],
            key=lambda x: x[1],
            reverse=True
        )[:10],  # Top 10 TP gradients by size
    }
