"""
Training Graph Extraction Integration

This module provides integration between inference graph extraction (mvp_runtime.py)
and backward graph extraction (mvp_backward_graph.py) for training TP prediction.

It extends the existing extract_inference_graphs() to also support backward graph
extraction, fulfilling Task 1 requirements.

================================================================================
TASK 1 DEPENDENCIES:
================================================================================
- Uses extract_inference_graphs() from mvp_runtime.py (already exists)
- Uses extract_backward_graph() from mvp_backward_graph.py (new, Task 1)

================================================================================
TASK 2+ DEPENDENCIES (what this provides to other tasks):
================================================================================
- BackwardGraphInfo: Input for estimate_backward_comm_time() in Task 2
- GradientInfo: Input for estimate_optimizer_tp_overhead() in Task 4
- get_gradient_summary(): Utility for logging/debugging

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from mvp_runtime import extract_inference_graphs, prepare_runtime_inputs
from mvp_backward_graph import (
    BackwardGraphInfo,
    GradientInfo,
    extract_backward_graph,
    get_gradient_summary,
)


@dataclass
class TrainingGraphs:
    """Container for all graphs extracted for training.

    Attributes:
        prefill_export: Forward prefill graph (from mvp_runtime.py)
        decode_export: Forward decode graph (from mvp_runtime.py)
        backward_info: Backward graph information (from mvp_backward_graph.py)
        runtime_inputs: Runtime inputs for the model (from mvp_runtime.py)
    """
    prefill_export: Any = None
    decode_export: Any = None
    backward_info: BackwardGraphInfo | None = None
    runtime_inputs: Any = None


def extract_training_graphs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    include_backward: bool = True,
    model_name: str = "unknown",
) -> TrainingGraphs:
    """Extract all graphs needed for training time prediction.

    This function combines:
    1. Forward graphs (prefill, decode) from mvp_runtime.py
    2. Backward graph from mvp_backward_graph.py (if include_backward=True)

    Args:
        model: The NON-TP model for graph extraction
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        include_backward: Whether to extract backward graph (default True)
        model_name: Optional name for the model

    Returns:
        TrainingGraphs containing:
        - prefill_export: Forward prefill export graph
        - decode_export: Forward decode export graph
        - backward_info: Backward graph info (if include_backward=True)
        - runtime_inputs: Runtime inputs used

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from mvp_train_graph import extract_training_graphs
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        >>> input_ids = torch.randint(0, 32000, (1, 128))
        >>> attn_mask = torch.ones(1, 128)
        >>> graphs = extract_training_graphs(model, input_ids, attn_mask)
        >>> print(f"Backward gradient bytes: {graphs.backward_info.total_gradient_bytes}")
    """
    # Extract forward graphs (existing functionality)
    inference_graphs = extract_inference_graphs(model, input_ids, attention_mask)

    # Prepare result
    result = TrainingGraphs(
        prefill_export=inference_graphs.get("prefill_export"),
        decode_export=inference_graphs.get("decode_export"),
        runtime_inputs=inference_graphs,
        backward_info=None,
    )

    # Extract backward graph if requested
    if include_backward:
        result.backward_info = extract_backward_graph(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            model_name=model_name,
        )

    return result


# Re-export for convenience
__all__ = [
    "TrainingGraphs",
    "extract_training_graphs",
    "BackwardGraphInfo",
    "GradientInfo",
    "get_gradient_summary",
]
