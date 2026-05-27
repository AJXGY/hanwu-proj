from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import torch

from mvp_types import HardwareCalibration, NodeEstimate
from mvp_train_types import (
    ModelArchitecture,
    TrainCalibration,
    TrainConfig,
    TrainPhaseSummary,
    TrainStepEstimate,
)


def estimate_model_architecture(model: torch.nn.Module) -> ModelArchitecture:
    """Extract model architecture information from a HuggingFace model."""
    # Try to get architecture from config
    if hasattr(model, 'config'):
        config = model.config
        num_layers = getattr(config, 'num_hidden_layers', 0)
        hidden_size = getattr(config, 'hidden_size', 0)
        num_heads = getattr(config, 'num_attention_heads', 0)
        vocab_size = getattr(config, 'vocab_size', 0)
        intermediate_size = getattr(config, 'intermediate_size', None)

        return ModelArchitecture(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            model_type=getattr(config, 'model_type', 'unknown'),
        )

    # Fallback: count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Rough estimation based on common LLM architectures
    hidden_size = 4096
    num_layers = int(total_params / (hidden_size * hidden_size * 12))
    num_heads = 32
    vocab_size = 32000

    return ModelArchitecture(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        vocab_size=vocab_size,
    )


def estimate_attention_flops(batch_size: int, seq_len: int, hidden_size: int,
                              num_heads: int, head_dim: int) -> float:
    """Estimate FLOPs for one attention layer (forward pass)."""
    # Q, K, V projections: 3 * (hidden_size * hidden_size * 3) but we count as 3 separate ops
    qkv_flops = 3 * batch_size * seq_len * hidden_size * hidden_size
    # Attention computation: Q * K^T + softmax + softmax * V
    attn_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    # Output projection
    out_flops = batch_size * seq_len * hidden_size * hidden_size
    return float(qkv_flops + attn_flops + out_flops)


def estimate_mlp_flops(batch_size: int, seq_len: int, hidden_size: int,
                        intermediate_size: int) -> float:
    """Estimate FLOPs for one MLP layer (forward pass)."""
    # Gate projection: hidden -> intermediate
    gate_flops = batch_size * seq_len * hidden_size * intermediate_size
    # Up projection: hidden -> intermediate
    up_flops = batch_size * seq_len * hidden_size * intermediate_size
    # Down projection: intermediate -> hidden
    down_flops = batch_size * seq_len * intermediate_size * hidden_size
    # Activation (element-wise, minimal FLOPs)
    return float(gate_flops + up_flops + down_flops)


def estimate_layer_norm_flops(batch_size: int, seq_len: int, hidden_size: int) -> float:
    """Estimate FLOPs for layer norm."""
    # Mean, variance, normalize, scale, shift
    return float(batch_size * seq_len * hidden_size * 6)


def estimate_embedding_flops(batch_size: int, seq_len: int, vocab_size: int,
                              hidden_size: int) -> float:
    """Estimate FLOPs for embedding lookup (approximate as gather + linear)."""
    # Embedding lookup is memory-bound, minimal compute
    return float(batch_size * seq_len * hidden_size * 2)


def estimate_rms_norm_flops(batch_size: int, seq_len: int, hidden_size: int) -> float:
    """Estimate FLOPs for RMS norm."""
    return float(batch_size * seq_len * hidden_size * 3)


def estimate_forward_flops(batch_size: int, seq_len: int,
                            arch: ModelArchitecture) -> float:
    """Estimate total forward pass FLOPs for the entire model."""
    flops = 0.0
    hidden_size = arch.hidden_size
    num_heads = arch.num_attention_heads
    head_dim = arch.head_dim
    intermediate_size = arch.intermediate_size or 4 * hidden_size

    # Input embedding
    flops += estimate_embedding_flops(batch_size, seq_len, arch.vocab_size, hidden_size)

    # Each layer
    for _ in range(arch.num_layers):
        # Self-attention
        flops += estimate_attention_flops(batch_size, seq_len, hidden_size, num_heads, head_dim)
        # MLP
        flops += estimate_mlp_flops(batch_size, seq_len, hidden_size, intermediate_size)
        # Post-attention norm
        flops += estimate_rms_norm_flops(batch_size, seq_len, hidden_size)

    # Final norm
    flops += estimate_rms_norm_flops(batch_size, seq_len, hidden_size)

    # LM head (only logits computation, not embedding tie)
    flops += batch_size * seq_len * hidden_size * arch.vocab_size

    return flops


def estimate_backward_flops(forward_flops: float, calibration: TrainCalibration) -> float:
    """Estimate backward pass FLOPs based on forward FLOPs."""
    # Backward pass typically requires 2-3x the FLOPs of forward pass
    # because we need to:
    # 1. Compute gradients of activations (similar to forward)
    # 2. Compute gradients of weights (requires transposed operations)
    return forward_flops * calibration.backward_compute_scale


def estimate_optimizer_flops(num_parameters: int, batch_size: int, seq_len: int,
                              calibration: TrainCalibration) -> tuple[float, float]:
    """Estimate optimizer step FLOPs and memory bytes.

    For Adam optimizer (per parameter):
    - First moment: m = beta1 * m + (1-beta1) * g  → 3 FLOPs (mul, mul, add)
    - Second moment: v = beta2 * v + (1-beta2) * g^2  → 4 FLOPs (mul, mul, add, mul)
    - Update: m_hat = m / (1-beta1^t)  → 2 FLOPs (div, mul)
    - Update: v_hat = v / (1-beta2^t)  → 2 FLOPs (div, mul)
    - Param update: param = param - lr * m_hat / (sqrt(v_hat) + eps)  → 4 FLOPs (sqrt, div, mul, sub)
    Total: ~15 FLOPs per parameter

    Memory movement (Adam with fp32 params, fp32 moments):
    - Read: gradients + moment1 + moment2 + params = 4 * 4 bytes
    - Write: updated params + updated moment1 + updated moment2 = 3 * 4 bytes
    Total: 7 * num_parameters * 4 bytes

    Returns (flops, bytes_moved)
    """
    # Adam optimizer: ~15 FLOPs per parameter (more realistic than 7)
    adam_flops = num_parameters * 15.0

    # Memory movement: read 4 tensors + write 3 tensors
    # Each tensor is num_parameters * 4 bytes (fp32)
    bytes_moved = num_parameters * (4 + 3) * 4  # 28 * num_parameters bytes

    return float(adam_flops), float(bytes_moved)


def estimate_gradient_communication_bytes(
    num_parameters: int,
    tp_size: int,
    ddp_enabled: bool,
    config: dict | None = None,
    interconnect: str = "local",
    nnodes: int = 1,
) -> tuple[float, float]:
    """Estimate gradient allreduce communication.

    Returns (bytes_per_step, latency_ms_estimate)

    For DDP with N GPUs:
    - Gradient allreduce requires all ranks to synchronize
    - For ring allreduce with 2 GPUs: 2 * (N-1) / N * param_bytes

    For TP: allreduce after embedding and output projection

    Args:
        num_parameters: Number of model parameters
        tp_size: Tensor parallel size
        ddp_enabled: Whether DDP is enabled
        config: Configuration dictionary (loaded from config/train_config.yaml)
        interconnect: Interconnect type ("local", "NV", "infiniband", "roce", etc.)
        nnodes: Number of nodes
    """
    if tp_size <= 1 and not ddp_enabled:
        return 0.0, 0.0

    # For DDP: allreduce gradients across ranks
    # For TP: allreduce after embedding and output projection
    param_bytes = num_parameters * 4  # fp32

    if ddp_enabled and tp_size > 1:
        # Allreduce across all ranks - most expensive
        # 2x for bidirectional allreduce
        return float(param_bytes * 2), 1.0  # increased latency
    elif ddp_enabled:
        # DDP allreduce - use config for bandwidth if available
        if config is not None:
            from config.config_loader import get_ddp_comm_params

            bandwidth = get_ddp_comm_params(config)
        else:
            bandwidth = 100.0  # default fallback

        if interconnect.startswith("NV"):
            # NVLink: use config values
            if config is not None:
                tp_comm = config.get("tp", {}).get("communication", {})
                bandwidth = tp_comm.get("nvlink_bandwidth_gbps", 450.0)
                latency = tp_comm.get("nvlink_latency_ms", 0.3)
            else:
                bandwidth = 450.0
                latency = 0.3
        else:
            # PCIe or network
            if config is not None:
                tp_comm = config.get("tp", {}).get("communication", {})
                bandwidth = tp_comm.get("pcie_bandwidth_gbps", 32.0)
                latency = tp_comm.get("pcie_latency_ms", 5.0)
            else:
                bandwidth = 32.0
                latency = 5.0

        # Ring allreduce factor: 2 * (N-1) / N ≈ 2 for large N
        ring_factor = 2.0 * (nnodes - 1) / nnodes if nnodes > 1 else 1.0
        return float(param_bytes * ring_factor), float(latency)
    elif tp_size > 1:
        # TP allreduce - use config for bandwidth if available
        if config is not None:
            from config.config_loader import get_tp_comm_params

            bandwidth, latency = get_tp_comm_params(config, interconnect, nnodes)
        else:
            if interconnect.startswith("NV"):
                bandwidth, latency = 450.0, 0.3
            else:
                bandwidth, latency = 32.0, 5.0

        ring_factor = 2.0 * (tp_size - 1) / tp_size
        comm_bytes = param_bytes * ring_factor
        return float(comm_bytes), float(latency)
    return 0.0, 0.0


# Backward FLOPs multipliers by op family
# These represent how backward FLOPs compare to forward FLOPs for each op type
# Based on actual physics: GEMM backward = 2x (input grad + weight grad),
# attention similar, view=0 (no compute, just shape change), etc.
BACKWARD_FLOPS_SCALE_BY_FAMILY = {
    "gemm": 2.0,
    "attention": 2.0,
    "embedding": 1.0,
    "pointwise": 1.0,
    "reduction": 1.0,
    "concat": 1.0,
    "view": 0.0,
}

# Backward memory multipliers by op family
# These represent how backward memory bytes compare to forward bytes for each op type
# Based on actual physics: GEMM backward needs ~2.5x memory (read input+weight, write grad_input+grad_weight)
BACKWARD_MEMORY_MULTIPLIER = {
    "gemm": 2.5,
    "attention": 2.5,
    "pointwise": 1.5,
    "reduction": 1.5,
    "embedding": 1.5,
    "concat": 1.5,
    "view": 0.5,
    "misc": 1.5,
}


def estimate_backward_node_from_forward(
    forward_node: NodeEstimate,
    calibration: TrainCalibration,
    gradient_bytes: float = 0.0,
) -> NodeEstimate:
    """Estimate backward pass time for a single node based on its forward estimate.

    Uses the forward node estimate to compute corresponding backward FLOPs and bytes,
    then calculates backward compute and memory time.

    Args:
        forward_node: The forward pass NodeEstimate for this node
        calibration: Training calibration with hardware parameters
        gradient_bytes: Total bytes of gradients associated with this node
                       (for memory-bound operations like weight gradients)

    Returns:
        NodeEstimate with backward time estimation
    """
    # Get backward FLOPs scale for this op family
    backward_flops_scale = BACKWARD_FLOPS_SCALE_BY_FAMILY.get(
        forward_node.op_family, 1.0
    )

    # Backward FLOPs = forward FLOPs × family-specific scale
    backward_flops = forward_node.flops * backward_flops_scale

    # Backward bytes: for accurate estimation, need to account for:
    # 1. Reading input activations (same as forward)
    # 2. Reading weights (same as forward)
    # 3. Writing activation gradients (same as forward output)
    # 4. Writing weight gradients (same as forward weight)
    #
    # Total backward memory ≈ 2x forward memory for most ops
    # For GEMM: 2x (input + weight + output read/write)
    # For attention: similar, ~2x forward
    # Using family-specific multipliers for better accuracy
    if gradient_bytes > 0:
        # gradient_bytes contains weight gradients; add activation gradient access
        # Activation gradients are similar in size to forward output
        backward_bytes = gradient_bytes + forward_node.bytes_moved
    else:
        # Fallback: estimate based on forward bytes with multiplier
        # backward ~= 2x forward for memory traffic (read input+weight, write grad_input+grad_weight)
        multiplier = BACKWARD_MEMORY_MULTIPLIER.get(forward_node.op_family, 1.5)
        backward_bytes = forward_node.bytes_moved * multiplier

    # Compute time based on effective backward TFLOPs
    # 实际算力 = 峰值 * 效率系数
    effective_backward_tflops = (
        calibration.gemm_tflops * calibration.backward_efficiency_scale
    )
    backward_compute_time_ms = (
        backward_flops / (effective_backward_tflops * 1e12) * 1e3
        if backward_flops > 0
        else 0.0
    )

    # Memory time for backward: gradient access dominates
    backward_memory_time_ms = (
        backward_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
        if backward_bytes > 0
        else 0.0
    )

    # Kernel launch overhead for backward
    runtime_overhead_ms = calibration.launch_overhead_ms * calibration.overhead_scale

    # Backward time: max of compute and memory, plus overhead
    # Apply parallelism factor since backward ops have dependencies
    adjusted_compute_time = (
        backward_compute_time_ms * calibration.parallelism_factor
    )
    estimated_time_ms = (
        max(adjusted_compute_time, backward_memory_time_ms) + runtime_overhead_ms
    )

    return NodeEstimate(
        node_name=forward_node.node_name + "_backward",
        target=forward_node.target,
        op_family=forward_node.op_family,
        phase="backward",
        region=forward_node.region,
        module_scope=forward_node.module_scope,
        output_shapes=forward_node.output_shapes,
        output_dtype=forward_node.output_dtype,
        shape_signature=forward_node.shape_signature,
        ordinal=forward_node.ordinal,
        flops=backward_flops,
        bytes_moved=backward_bytes,
        compute_time_ms=backward_compute_time_ms,
        memory_time_ms=backward_memory_time_ms,
        runtime_overhead_ms=runtime_overhead_ms,
        estimated_time_ms=estimated_time_ms,
    )


def estimate_backward_from_graph_nodes(
    forward_graph_nodes: list[NodeEstimate],
    calibration: TrainCalibration,
    gradient_bytes_by_scope: dict[str, float] | None = None,
) -> tuple[float, float, float, dict[str, float]]:
    """Estimate backward pass time from forward graph nodes using per-node estimation.

    Uses per-family backward FLOPs scales for accurate estimation:
    - GEMM backward = 2x forward (input grad + weight grad)
    - Attention backward = 2x forward (similar structure)
    - View backward = 0 (no compute)
    - Other ops = 1x forward

    Args:
        forward_graph_nodes: List of forward NodeEstimates
        calibration: Training calibration with hardware parameters
        gradient_bytes_by_scope: Optional dict mapping module_scope to gradient bytes
                                (e.g., from BackwardGraphInfo.gradient_infos)

    Returns:
        (backward_time_ms, total_backward_flops, total_backward_bytes, op_family_breakdown)
    """
    total_backward_time = 0.0
    total_backward_flops = 0.0
    total_backward_bytes = 0.0
    op_family_breakdown = defaultdict(float)

    for node in forward_graph_nodes:
        # Get gradient bytes for this node by module_scope
        grad_bytes = 0.0
        if gradient_bytes_by_scope is not None:
            grad_bytes = gradient_bytes_by_scope.get(node.module_scope, 0.0)

        # Estimate backward time for this node
        backward_estimate = estimate_backward_node_from_forward(
            node, calibration, grad_bytes
        )

        total_backward_time += backward_estimate.estimated_time_ms
        total_backward_flops += backward_estimate.flops
        total_backward_bytes += backward_estimate.bytes_moved
        op_family_breakdown[backward_estimate.op_family] += backward_estimate.estimated_time_ms

    return (
        total_backward_time,
        total_backward_flops,
        total_backward_bytes,
        dict(op_family_breakdown),
    )


def build_gradient_bytes_mapping(
    gradient_infos: list,
) -> dict[str, float]:
    """Build a mapping from module_scope to gradient bytes.

    Args:
        gradient_infos: List of GradientInfo objects from BackwardGraphInfo

    Returns:
        Dict mapping module_scope (e.g., "model.layers.0.self_attn.q_proj") to bytes
    """
    mapping = {}
    for gi in gradient_infos:
        # Gradient name is the module scope
        scope = gi.name
        mapping[scope] = gi.bytes
    return mapping


def estimate_train_step_from_graph(
    forward_graph_nodes: list[NodeEstimate],
    num_parameters: int,
    calibration: TrainCalibration,
    config: TrainConfig,
) -> TrainStepEstimate:
    """Estimate training step time using graph-based approach.

    Reuses forward graph analysis from estimate_node() and estimates
    backward costs analytically from the forward graph structure.
    """
    # Forward: sum of all forward node estimates
    forward_time_ms = sum(n.estimated_time_ms for n in forward_graph_nodes)
    forward_flops = sum(n.flops for n in forward_graph_nodes)
    forward_bytes = sum(n.bytes_moved for n in forward_graph_nodes)

    # Backward: estimate per-node backward cost using new graph-based method
    (
        backward_time_ms,
        backward_flops,
        backward_bytes,
        backward_breakdown,
    ) = estimate_backward_from_graph_nodes(forward_graph_nodes, calibration)

    # Optimizer: formula-based (memory-bound)
    optimizer_flops, optimizer_bytes = estimate_optimizer_flops(
        num_parameters, config.batch_size, config.seq_len, calibration
    )
    optimizer_memory_time_ms = (
        optimizer_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
    )
    optimizer_time_ms = optimizer_memory_time_ms * calibration.optimizer_scale_factor

    # Apply gradient accumulation scaling
    backward_time_ms = backward_time_ms * config.gradient_accumulation_steps

    # Total time
    total_time_ms = forward_time_ms + backward_time_ms + optimizer_time_ms

    # Samples per second
    samples_per_sec = (config.batch_size * 1000.0) / total_time_ms if total_time_ms > 0 else 0.0
    tokens_per_sec = (
        (config.batch_size * config.seq_len * 1000.0) / total_time_ms
        if total_time_ms > 0
        else None
    )

    # Build phase summaries
    forward_summary = TrainPhaseSummary(
        phase="forward_step",
        estimated_time_ms=forward_time_ms,
        flops=forward_flops,
        bytes_moved=forward_bytes,
        compute_time_ms=sum(n.compute_time_ms for n in forward_graph_nodes),
        memory_time_ms=sum(n.memory_time_ms for n in forward_graph_nodes),
        comm_time_ms=0.0,
        node_count=len(forward_graph_nodes),
        top_ops=[],
        op_family_breakdown_ms={},
    )

    backward_summary = TrainPhaseSummary(
        phase="backward_step",
        estimated_time_ms=backward_time_ms,
        flops=backward_flops,
        bytes_moved=backward_bytes,
        compute_time_ms=backward_flops / (calibration.gemm_tflops * 1e12) * 1e3,
        memory_time_ms=backward_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3,
        comm_time_ms=0.0,
        node_count=len(forward_graph_nodes),
        top_ops=[],
        op_family_breakdown_ms=backward_breakdown,
    )

    optimizer_summary = TrainPhaseSummary(
        phase="optimizer_step",
        estimated_time_ms=optimizer_time_ms,
        flops=optimizer_flops,
        bytes_moved=optimizer_bytes,
        compute_time_ms=0.0,  # Optimizer is memory-bound
        memory_time_ms=optimizer_memory_time_ms,
        comm_time_ms=0.0,
        node_count=1,
        top_ops=[],
        op_family_breakdown_ms={},
    )

    return TrainStepEstimate(
        forward_time_ms=forward_time_ms,
        backward_time_ms=backward_time_ms,
        optimizer_time_ms=optimizer_time_ms,
        total_time_ms=total_time_ms,
        samples_per_sec=samples_per_sec,
        tokens_per_sec=tokens_per_sec,
        forward_summary=forward_summary,
        backward_summary=backward_summary,
        optimizer_summary=optimizer_summary,
    )


def estimate_memory_bytes_forward(batch_size: int, seq_len: int,
                                    arch: ModelArchitecture) -> float:
    """Estimate activation memory in bytes for one forward pass."""
    # Hidden states for all layers + embeddings
    hidden_bytes = batch_size * seq_len * arch.hidden_size * 4 * (arch.num_layers + 1)

    # Attention KV cache (for training with reuse, minimal)
    # In training, we typically don't cache, but we store activations for backward

    # Gradients (same size as activations during backward)
    grad_bytes = hidden_bytes

    # Model parameters
    param_bytes = arch.parameters * 4

    # Moment estimates (for Adam, 2x parameters)
    moment_bytes = arch.parameters * 4 * 2

    return float(hidden_bytes + grad_bytes + param_bytes + moment_bytes)


def build_synthetic_nodes_for_backward(
    batch_size: int,
    seq_len: int,
    arch: ModelArchitecture,
) -> list[NodeEstimate]:
    """Build synthetic graph nodes for backward pass estimation.

    Creates minimal NodeEstimate objects representing the major model components
    so that per-family backward scaling can be applied.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        arch: Model architecture

    Returns:
        List of synthetic NodeEstimate objects for backward estimation
    """
    nodes = []
    node_idx = 0
    hidden_size = arch.hidden_size
    num_heads = arch.num_attention_heads
    head_dim = arch.head_dim
    intermediate_size = arch.intermediate_size or 4 * hidden_size

    # Input embedding
    embed_flops = estimate_embedding_flops(batch_size, seq_len, arch.vocab_size, hidden_size)
    nodes.append(NodeEstimate(
        node_name="model.embed",
        target="synthetic",
        op_family="embedding",
        phase="forward",
        region="synthetic",
        module_scope="model.embed",
        output_shapes=[[batch_size, seq_len, hidden_size]],
        output_dtype="float32",
        shape_signature="",
        ordinal=node_idx,
        flops=embed_flops,
        bytes_moved=0,
        compute_time_ms=0,
        memory_time_ms=0,
        runtime_overhead_ms=0,
        estimated_time_ms=0,
    ))
    node_idx += 1

    # Each transformer layer: attention + MLP + norm
    for layer_idx in range(arch.num_layers):
        # Self-attention
        attn_flops = estimate_attention_flops(batch_size, seq_len, hidden_size, num_heads, head_dim)
        nodes.append(NodeEstimate(
            node_name=f"model.layers.{layer_idx}.self_attn",
            target="synthetic",
            op_family="attention",
            phase="forward",
            region="synthetic",
            module_scope=f"model.layers.{layer_idx}.self_attn",
            output_shapes=[[batch_size, seq_len, hidden_size]],
            output_dtype="float32",
            shape_signature="",
            ordinal=node_idx,
            flops=attn_flops,
            bytes_moved=0,
            compute_time_ms=0,
            memory_time_ms=0,
            runtime_overhead_ms=0,
            estimated_time_ms=0,
        ))
        node_idx += 1

        # MLP
        mlp_flops = estimate_mlp_flops(batch_size, seq_len, hidden_size, intermediate_size)
        nodes.append(NodeEstimate(
            node_name=f"model.layers.{layer_idx}.mlp",
            target="synthetic",
            op_family="gemm",
            phase="forward",
            region="synthetic",
            module_scope=f"model.layers.{layer_idx}.mlp",
            output_shapes=[[batch_size, seq_len, hidden_size]],
            output_dtype="float32",
            shape_signature="",
            ordinal=node_idx,
            flops=mlp_flops,
            bytes_moved=0,
            compute_time_ms=0,
            memory_time_ms=0,
            runtime_overhead_ms=0,
            estimated_time_ms=0,
        ))
        node_idx += 1

        # Post-attention norm (part of the layer, counted once per layer)
        norm_flops = estimate_rms_norm_flops(batch_size, seq_len, hidden_size)
        nodes.append(NodeEstimate(
            node_name=f"model.layers.{layer_idx}.norm",
            target="synthetic",
            op_family="pointwise",
            phase="forward",
            region="synthetic",
            module_scope=f"model.layers.{layer_idx}.norm",
            output_shapes=[[batch_size, seq_len, hidden_size]],
            output_dtype="float32",
            shape_signature="",
            ordinal=node_idx,
            flops=norm_flops,
            bytes_moved=0,
            compute_time_ms=0,
            memory_time_ms=0,
            runtime_overhead_ms=0,
            estimated_time_ms=0,
        ))
        node_idx += 1

    # Final norm
    final_norm_flops = estimate_rms_norm_flops(batch_size, seq_len, hidden_size)
    nodes.append(NodeEstimate(
        node_name="model.final_norm",
        target="synthetic",
        op_family="pointwise",
        phase="forward",
        region="synthetic",
        module_scope="model.final_norm",
        output_shapes=[[batch_size, seq_len, hidden_size]],
        output_dtype="float32",
        shape_signature="",
        ordinal=node_idx,
        flops=final_norm_flops,
        bytes_moved=0,
        compute_time_ms=0,
        memory_time_ms=0,
        runtime_overhead_ms=0,
        estimated_time_ms=0,
    ))

    return nodes


def estimate_train_step(
    batch_size: int,
    seq_len: int,
    arch: ModelArchitecture,
    calibration: TrainCalibration,
    config: TrainConfig,
) -> TrainStepEstimate:
    """Estimate complete training step time.

    =============================================================================
    关键调参说明 (KEY TUNING PARAMETERS):
    =============================================================================

    1. effective_forward_tflops = gemm_tflops * 0.9
       - 原因: 训练时计算效率通常比峰值略低
       - 影响: ~5-10% 误差

    2. kernel_overhead_ms = num_layers * 0.15
       - 原因: 每个transformer层有多个小kernel launch开销
       - 影响: ~10-15% 误差 (容易被忽略!)

    3. backward_compute_scale (默认3.5)
       - 原因: 反向传播计算量约为前向的3-4倍
       - 包括: 激活梯度 + 权重梯度 + 额外归约操作
       - 影响: ~20-30% 误差

    4. optimizer_memory_time_ms * 1.4 (关键!)
       - 原因: Adam优化器实际吞吐约为理论内存带宽的60-70%
       - 因素: 随机内存访问、小tensor、同步开销
       - 影响: ~40-60% 误差 (如果不校准)

    5. activation_memory_bytes * 2 (save + recompute)
       - 原因: 训练需要保存激活值用于反向传播
       - 影响: ~10-20% 误差

    =============================================================================
    """
    # ----- Forward Pass -----
    # [KEY PARAMETER] 训练时计算效率折扣
    effective_forward_tflops = calibration.gemm_tflops * calibration.effective_tflops_scale

    forward_flops = estimate_forward_flops(batch_size, seq_len, arch)
    forward_compute_time_ms = (
        forward_flops / (effective_forward_tflops * 1e12) * 1e3
    )

    # [KEY PARAMETER] 激活内存: 保存用于反向 (2x for save + recompute)
    activation_memory_bytes = (
        batch_size * seq_len * arch.hidden_size * 4 * arch.num_layers * 2
    )
    forward_memory_time_ms = (
        activation_memory_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
    )

    # [KEY PARAMETER] Kernel launch开销: 每层约0.15ms
    # 训练有大量小操作，kernel launch开销不可忽略
    kernel_overhead_ms = arch.num_layers * 0.15

    forward_time_ms = max(forward_compute_time_ms, forward_memory_time_ms) + kernel_overhead_ms

    # ----- Backward Pass (per-family scaling) -----
    # Use per-family backward scaling for consistency with estimate_train_step_from_graph
    # Build synthetic nodes for per-family backward estimation
    synthetic_nodes = build_synthetic_nodes_for_backward(batch_size, seq_len, arch)

    # Use estimate_backward_from_graph_nodes for per-family scaling
    # This applies BACKWARD_FLOPS_SCALE_BY_FAMILY per op family (gemm=2x, attention=2x, etc.)
    (
        backward_time_ms,
        backward_flops,
        backward_bytes,
        backward_breakdown,
    ) = estimate_backward_from_graph_nodes(synthetic_nodes, calibration)

    # Adjust backward time for gradient accumulation
    backward_time_ms = backward_time_ms * config.gradient_accumulation_steps

    # ----- Optimizer Step -----
    num_params = arch.parameters
    optimizer_flops, optimizer_bytes = estimate_optimizer_flops(
        num_params, batch_size, seq_len, calibration
    )

    # [KEY PARAMETER] Optimizer内存时间计算
    # Adam优化器特点:
    #   - 逐参数操作，随机内存访问
    #   - 低计算强度 (compute-inefficient)
    #   - 实际吞吐约为理论内存带宽的60-70%
    optimizer_memory_time_ms = (
        optimizer_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
    )

    # [KEY PARAMETER] Optimizer时间校准系数
    # 经验值: 实际时间约为理论内存带宽时间的optimizer_scale_factor倍
    # 如果误差大，可微调此系数
    optimizer_time_ms = optimizer_memory_time_ms * calibration.optimizer_scale_factor

    # Communication estimation
    comm_bytes, comm_latency_ms = estimate_gradient_communication_bytes(
        num_params, config.tp_size, config.ddp_enabled
    )
    comm_time_ms = comm_latency_ms
    if comm_bytes > 0:
        comm_time_ms += (
            comm_bytes / (calibration.gradient_allreduce_tflops * 1e9) * 1e3
        )

    # Total time
    total_time_ms = forward_time_ms + backward_time_ms + optimizer_time_ms + comm_time_ms

    # Samples per second
    samples_per_sec = (batch_size * 1000.0) / total_time_ms if total_time_ms > 0 else 0.0

    # Build phase summaries
    forward_summary = TrainPhaseSummary(
        phase="forward",
        estimated_time_ms=forward_time_ms,
        flops=forward_flops,
        bytes_moved=activation_memory_bytes,
        compute_time_ms=forward_compute_time_ms,
        memory_time_ms=forward_memory_time_ms,
        comm_time_ms=0.0,
        node_count=arch.num_layers * 2 + 1,
        top_ops=[],
        op_family_breakdown_ms={},
    )

    backward_summary = TrainPhaseSummary(
        phase="backward",
        estimated_time_ms=backward_time_ms,
        flops=backward_flops,
        bytes_moved=backward_bytes,
        compute_time_ms=backward_flops / (effective_forward_tflops * 1e12) * 1e3,
        memory_time_ms=backward_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3 if backward_bytes > 0 else 0.0,
        comm_time_ms=comm_time_ms,
        node_count=len(synthetic_nodes),
        top_ops=[],
        op_family_breakdown_ms=backward_breakdown,
    )

    optimizer_summary = TrainPhaseSummary(
        phase="optimizer",
        estimated_time_ms=optimizer_time_ms,
        flops=optimizer_flops,
        bytes_moved=optimizer_bytes,
        compute_time_ms=0.0,  # Optimizer is memory-bound
        memory_time_ms=optimizer_memory_time_ms,
        comm_time_ms=0.0,
        node_count=1,
        top_ops=[],
        op_family_breakdown_ms={},
    )

    return TrainStepEstimate(
        forward_time_ms=forward_time_ms,
        backward_time_ms=backward_time_ms,
        optimizer_time_ms=optimizer_time_ms,
        total_time_ms=total_time_ms,
        samples_per_sec=samples_per_sec,
        tokens_per_sec=(batch_size * seq_len * 1000.0) / total_time_ms if total_time_ms > 0 else None,
        forward_summary=forward_summary,
        backward_summary=backward_summary,
        optimizer_summary=optimizer_summary,
    )


def estimate_epoch_time(
    train_step_estimate: TrainStepEstimate,
    num_train_tokens: int,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    """Estimate total epoch training time."""
    tokens_per_step = batch_size * seq_len
    num_steps = num_train_tokens // tokens_per_step if tokens_per_step > 0 else 0

    total_time_s = (train_step_estimate.total_time_ms / 1000.0) * num_steps

    return {
        "num_steps": num_steps,
        "total_time_s": total_time_s,
        "total_time_min": total_time_s / 60.0,
        "forward_time_s": (train_step_estimate.forward_time_ms / 1000.0) * num_steps,
        "backward_time_s": (train_step_estimate.backward_time_ms / 1000.0) * num_steps,
        "optimizer_time_s": (train_step_estimate.optimizer_time_ms / 1000.0) * num_steps,
    }


def build_train_estimate_report(
    arch: ModelArchitecture,
    config: TrainConfig,
    step_estimate: TrainStepEstimate,
    calibration: TrainCalibration,
    num_train_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a complete training estimate report."""

    report = {
        "mode": "training",
        "model": {
            "architecture": {
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "num_attention_heads": arch.num_attention_heads,
                "head_dim": arch.head_dim,
                "vocab_size": arch.vocab_size,
                "intermediate_size": arch.intermediate_size,
                "model_type": arch.model_type,
                "parameters": arch.parameters,
            },
            "training_config": {
                "batch_size": config.batch_size,
                "seq_len": config.seq_len,
                "global_batch_size": config.global_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "ddp_enabled": config.ddp_enabled,
                "tp_size": config.tp_size,
            },
        },
        "calibration": {
            "device_name": calibration.device_name,
            "gemm_tflops": calibration.gemm_tflops,
            "attention_tflops": calibration.attention_tflops,
            "memory_bandwidth_gbps": calibration.memory_bandwidth_gbps,
            "backward_compute_scale": calibration.backward_compute_scale,
        },
        "estimate": {
            "per_step": {
                "forward_time_ms": step_estimate.forward_time_ms,
                "backward_time_ms": step_estimate.backward_time_ms,
                "optimizer_time_ms": step_estimate.optimizer_time_ms,
                "total_time_ms": step_estimate.total_time_ms,
                "samples_per_sec": step_estimate.samples_per_sec,
                "tokens_per_sec": step_estimate.tokens_per_sec,
            },
            "forward": {
                "flops": step_estimate.forward_summary.flops if step_estimate.forward_summary else 0,
                "compute_time_ms": step_estimate.forward_summary.compute_time_ms if step_estimate.forward_summary else 0,
                "memory_time_ms": step_estimate.forward_summary.memory_time_ms if step_estimate.forward_summary else 0,
            },
            "backward": {
                "flops": step_estimate.backward_summary.flops if step_estimate.backward_summary else 0,
                "compute_time_ms": step_estimate.backward_summary.compute_time_ms if step_estimate.backward_summary else 0,
                "memory_time_ms": step_estimate.backward_summary.memory_time_ms if step_estimate.backward_summary else 0,
            },
            "optimizer": {
                "flops": step_estimate.optimizer_summary.flops if step_estimate.optimizer_summary else 0,
                "compute_time_ms": step_estimate.optimizer_summary.compute_time_ms if step_estimate.optimizer_summary else 0,
                "memory_time_ms": step_estimate.optimizer_summary.memory_time_ms if step_estimate.optimizer_summary else 0,
            },
        },
    }

    if num_train_tokens is not None and num_train_tokens > 0:
        epoch_estimate = estimate_epoch_time(
            step_estimate, num_train_tokens, config.batch_size, config.seq_len
        )
        report["epoch_estimate"] = epoch_estimate

    return report
