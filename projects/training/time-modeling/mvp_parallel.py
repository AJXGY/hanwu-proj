from __future__ import annotations

from typing import Any

import torch

from mvp_types import ExecutionConfig

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


def apply_tensor_parallel(model: torch.nn.Module, execution: ExecutionConfig) -> Any:
    if execution.parallel_mode != "tp":
        return None
    if parallelize_module is None or init_device_mesh is None:
        raise RuntimeError("Tensor parallel APIs are unavailable in this PyTorch build")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "tp mode currently expects a Llama-style model.model.layers layout"
        )
    mesh = init_device_mesh("cuda", (execution.tp_size,))
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
                layer.self_attn.num_heads = max(
                    1, layer.self_attn.num_heads // execution.tp_size
                )
            if hasattr(layer.self_attn, "num_key_value_heads"):
                layer.self_attn.num_key_value_heads = max(
                    1,
                    layer.self_attn.num_key_value_heads // execution.tp_size,
                )
    return mesh
