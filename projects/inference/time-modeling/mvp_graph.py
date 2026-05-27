from __future__ import annotations

import json
from dataclasses import replace
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch

from mvp_types import ExecutionConfig, NodeEstimate


def product(shape: Sequence[int]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return value


def dtype_num_bytes(dtype: Any) -> int:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.split(".")[-1], None)
    if dtype in (torch.float16, torch.bfloat16, torch.int16, torch.uint16):
        return 2
    if dtype in (torch.float32, torch.int32, torch.uint32):
        return 4
    if dtype in (torch.float64, torch.int64, torch.uint64):
        return 8
    if dtype in (torch.int8, torch.uint8, torch.bool):
        return 1
    return 4


def safe_tensor_meta(value: Any) -> list[Any]:
    metas: list[Any] = []
    if value is None:
        return metas
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        metas.append(value)
        return metas
    if isinstance(value, (list, tuple)):
        for item in value:
            metas.extend(safe_tensor_meta(item))
    return metas


def node_output_metas(node: torch.fx.Node) -> list[Any]:
    return safe_tensor_meta(node.meta.get("tensor_meta") or node.meta.get("val"))


def arg_output_metas(arg: Any) -> list[Any]:
    if isinstance(arg, torch.fx.Node):
        return node_output_metas(arg)
    if isinstance(arg, (list, tuple)):
        metas: list[Any] = []
        for item in arg:
            metas.extend(arg_output_metas(item))
        return metas
    return []


def metas_numel_and_bytes(metas: Iterable[Any]) -> tuple[int, int, str]:
    total_numel = 0
    total_bytes = 0
    dtype_name = "unknown"
    for meta in metas:
        shape = tuple(int(dim) for dim in meta.shape)
        numel = product(shape)
        total_numel += numel
        total_bytes += numel * dtype_num_bytes(meta.dtype)
        dtype_name = str(meta.dtype)
    return total_numel, total_bytes, dtype_name


def module_scope_from_stack(stack: Any) -> str:
    if not stack:
        return "global"
    scopes = [entry[0] for entry in stack.values() if entry and entry[0]]
    return normalize_module_scope_name(scopes[-1] if scopes else "global")


def normalize_module_scope_name(scope: str) -> str:
    while scope.startswith("model.model."):
        scope = "model." + scope[len("model.model.") :]
    return scope


def region_from_scope(scope: str) -> str:
    if ".self_attn" in scope:
        return "attention"
    if ".mlp" in scope:
        return "mlp"
    if "embed_tokens" in scope:
        return "embedding"
    if scope.endswith("norm") or ".norm" in scope or "layernorm" in scope:
        return "norm"
    if "lm_head" in scope:
        return "lm_head"
    return "misc"


def op_family_from_target(target: str) -> str:
    if "scaled_dot_product_attention" in target:
        return "attention"
    if any(name in target for name in ("linear", "mm", "bmm", "matmul", "addmm")):
        return "gemm"
    if "embedding" in target:
        return "embedding"
    if any(
        name in target
        for name in ("silu", "gelu", "relu", "mul", "add", "sub", "neg", "pow")
    ):
        return "pointwise"
    if any(name in target for name in ("mean", "sum", "rsqrt", "softmax")):
        return "reduction"
    if any(
        name in target
        for name in (
            "slice",
            "view",
            "reshape",
            "transpose",
            "unsqueeze",
            "expand",
            "getitem",
            "contiguous",
        )
    ):
        return "view"
    if "cat" in target:
        return "concat"
    return "misc"


def tensor_shapes(metas: Iterable[Any]) -> list[list[int]]:
    return [list(int(dim) for dim in meta.shape) for meta in metas]


def shape_numel(shape: Sequence[int]) -> int:
    return product([int(dim) for dim in shape]) if shape else 0


def tensor_shapes_from_value(value: Any) -> list[list[int]]:
    shapes: list[list[int]] = []
    if isinstance(value, torch.Tensor):
        shapes.append([int(dim) for dim in value.shape])
    elif isinstance(value, (list, tuple)):
        for item in value:
            shapes.extend(tensor_shapes_from_value(item))
    elif isinstance(value, dict):
        for item in value.values():
            shapes.extend(tensor_shapes_from_value(item))
    return shapes


def shape_signature(shapes: Sequence[Sequence[int]]) -> str:
    return json.dumps([list(int(dim) for dim in shape) for shape in shapes])


def canonical_target_name(target: Any) -> str:
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    text = str(target).replace("torch.ops.", "")
    if text.startswith("<built-in function ") and text.endswith(">"):
        return text[len("<built-in function ") : -1]
    if "::" in text:
        return text.split(".default", 1)[0]
    parts = text.split(".")
    if len(parts) >= 2 and parts[0] in {
        "aten",
        "prims",
        "c10d_functional",
        "_c10d_functional",
    }:
        return f"{parts[0]}::{parts[1]}"
    return text


def assign_ordinals_by_group(
    items: list[dict[str, Any]], key_fields: Sequence[str]
) -> None:
    counters: dict[tuple[Any, ...], int] = defaultdict(int)
    for item in items:
        key = tuple(item[field] for field in key_fields)
        item["ordinal"] = counters[key]
        counters[key] += 1


def estimate_output_bytes_from_shapes(
    shapes: Sequence[Sequence[int]], dtype_name: str
) -> float:
    dtype = getattr(torch, dtype_name.split(".")[-1], None) if dtype_name else None
    total_numel = sum(shape_numel(shape) for shape in shapes)
    return float(total_numel * dtype_num_bytes(dtype or dtype_name))


def estimate_output_bytes(estimate: NodeEstimate) -> float:
    return estimate_output_bytes_from_shapes(
        estimate.output_shapes, estimate.output_dtype
    )


def flatten_last_dim_shapes(shapes: Sequence[Sequence[int]]) -> list[list[int]]:
    flattened = []
    for shape in shapes:
        dims = [int(dim) for dim in shape]
        if len(dims) >= 2:
            flattened.append([product(dims[:-1]), dims[-1]])
        elif dims:
            flattened.append(dims)
    return flattened


def is_tp_colwise_scope(scope: str) -> bool:
    return any(
        scope.endswith(suffix)
        for suffix in (
            ".self_attn.q_proj",
            ".self_attn.k_proj",
            ".self_attn.v_proj",
            ".mlp.gate_proj",
            ".mlp.up_proj",
        )
    )


def is_tp_rowwise_scope(scope: str) -> bool:
    return any(
        scope.endswith(suffix) for suffix in (".self_attn.o_proj", ".mlp.down_proj")
    )


def is_tp_parallel_scope(scope: str) -> bool:
    return ".self_attn" in scope or ".mlp" in scope


def tp_localized_target(estimate: NodeEstimate, execution: ExecutionConfig) -> str:
    if execution.parallel_mode != "tp":
        return estimate.target
    if estimate.op_family == "gemm" and (
        is_tp_colwise_scope(estimate.module_scope)
        or is_tp_rowwise_scope(estimate.module_scope)
    ):
        return "aten::mm"
    return estimate.target


def tp_compare_shape_signature(
    estimate: NodeEstimate, execution: ExecutionConfig
) -> str:
    if execution.parallel_mode != "tp":
        return estimate.shape_signature
    if estimate.op_family == "gemm" and (
        is_tp_colwise_scope(estimate.module_scope)
        or is_tp_rowwise_scope(estimate.module_scope)
    ):
        return shape_signature(flatten_last_dim_shapes(estimate.output_shapes))
    return estimate.shape_signature


def tp_parallel_time_scale(estimate: NodeEstimate, execution: ExecutionConfig) -> float:
    if execution.parallel_mode != "tp" or execution.tp_size <= 1:
        return 1.0
    if estimate.op_family == "attention" and ".self_attn" in estimate.module_scope:
        return 1.0 / execution.tp_size
    if estimate.op_family in {
        "gemm",
        "pointwise",
        "reduction",
        "concat",
        "view",
        "misc",
    } and is_tp_parallel_scope(estimate.module_scope):
        return 1.0 / execution.tp_size
    return 1.0


def tp_shard_node_estimate(
    estimate: NodeEstimate, execution: ExecutionConfig
) -> NodeEstimate:
    if execution.parallel_mode != "tp":
        return estimate
    scale = tp_parallel_time_scale(estimate, execution)
    if scale == 1.0:
        return estimate
    output_bytes = estimate_output_bytes(estimate)
    if is_tp_rowwise_scope(estimate.module_scope) and estimate.op_family == "gemm":
        local_bytes = (
            output_bytes + max(estimate.bytes_moved - output_bytes, 0.0) * scale
        )
    else:
        local_bytes = estimate.bytes_moved * scale
    compute_time_ms = estimate.compute_time_ms * scale
    memory_time_ms = estimate.memory_time_ms * scale
    return replace(
        estimate,
        flops=estimate.flops * scale,
        bytes_moved=local_bytes,
        compute_time_ms=compute_time_ms,
        memory_time_ms=memory_time_ms,
        estimated_time_ms=max(compute_time_ms, memory_time_ms)
        + estimate.runtime_overhead_ms,
    )


def flatten_past_key_values(
    past_key_values: Any,
) -> list[torch.Tensor]:
    flat: list[torch.Tensor] = []
    if hasattr(past_key_values, "layers"):
        pairs = [
            (layer.keys, layer.values)
            for layer in past_key_values.layers
            if hasattr(layer, "keys") and hasattr(layer, "values")
        ]
    else:
        pairs = list(past_key_values)
    for key, value in pairs:
        flat.extend([key, value])
    return flat


def module_scope_key(path: str) -> tuple[int, str, int, str]:
    parts = path.split(".")
    layer_id = -1
    module_kind = parts[-1]
    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts):
            try:
                layer_id = int(parts[idx + 1])
            except ValueError:
                layer_id = -1
    priority = 0 if module_kind == "self_attn" else 1 if module_kind == "mlp" else 2
    return layer_id, module_kind, priority, path


def scope_aliases(scope: str) -> list[str]:
    aliases = {scope}
    aliases.add(f"model.{scope}")
    if scope.startswith("model."):
        aliases.add(scope[len("model.") :])
        aliases.add(f"model.{scope[len('model.') :]}")
    return sorted(aliases, key=len, reverse=True)
