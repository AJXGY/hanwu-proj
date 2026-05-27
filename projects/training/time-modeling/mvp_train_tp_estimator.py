"""
Training TP (Tensor Parallelism) Forward Estimator

任务三：整合 Forward TP 估算（复用推理逻辑）

本模块实现训练中 Forward Pass 的精细化 TP 缩放估算，复用 mvp_graph.py 中的
tp_shard_node_estimate 和 tp_parallel_time_scale 函数。

依赖说明：
- 任务一（extract_backward_graphs）：未实现，假定 API 如下：
    def extract_backward_graphs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        '''
        返回：
        - backward_export: 反向传播图
        - gradient_sizes: 每层输出梯度大小 (bytes)
        '''

- 任务二（estimate_backward_comm_time）：未实现，假定 API 如下：
    def estimate_backward_comm_time(
        backward_graph: Any,
        tp_size: int,
        calibration: TrainCalibration,
    ) -> float:
        '''返回 backward 通信时间 (ms)'''

- 任务四（estimate_optimizer_tp_overhead）：未实现，假定 API 如下：
    def estimate_optimizer_tp_overhead(
        num_parameters: int,
        tp_size: int,
        calibration: TrainCalibration,
    ) -> float:
        '''返回 optimizer TP 开销时间 (ms)'''
"""

from __future__ import annotations

from typing import Any

import torch

from mvp_graph import tp_parallel_time_scale, tp_shard_node_estimate
from mvp_train_types import (
    ModelArchitecture,
    TrainCalibration,
    TrainConfig,
)
from mvp_types import ExecutionConfig


# =============================================================================
# 辅助函数：计算有效 TP 缩放比例
# =============================================================================

def compute_effective_tp_scale(
    arch: ModelArchitecture,
    tp_size: int,
) -> float:
    """
    计算整体 TP 缩放比例（考虑精细化缩放 vs 简单缩放）

    与简单 1/tp_size 缩放不同，本函数考虑：
    - attention 操作：按 1/tp_size 缩放
    - gemm 操作：按 1/tp_size 缩放
    - layernorm, embedding 等：不缩放

    有效缩放比例 = (缩放 ops 的 FLOPs 占比 / tp_size) + (不缩放 ops 的 FLOPs 占比 * 1.0)

    参数：
        arch: 模型架构信息
        tp_size: Tensor Parallelism 大小

    返回：
        整体 TP 缩放比例 (float)
    """
    if tp_size <= 1:
        return 1.0

    # 计算各层 FLOPs 占比
    hidden_size = arch.hidden_size
    num_heads = arch.num_attention_heads
    head_dim = arch.head_dim
    intermediate_size = arch.intermediate_size or 4 * hidden_size

    # 估算单层 FLOPs
    # Attention FLOPs
    qkv_flops = 3 * hidden_size * hidden_size
    attn_flops = 2 * num_heads * head_dim * hidden_size * 2  # QK^T + softmax*V (seq_len factor omitted for ratio)
    out_proj_flops = hidden_size * hidden_size
    attention_flops = qkv_flops + attn_flops + out_proj_flops

    # MLP FLOPs (gate + up + down)
    mlp_flops = (
        hidden_size * intermediate_size +  # gate
        hidden_size * intermediate_size +  # up
        intermediate_size * hidden_size    # down
    )

    # LayerNorm FLOPs (simplified)
    layernorm_flops = hidden_size * 6

    # 可缩放 ops (attention + gemm in MLP)
    scalable_flops = attention_flops + hidden_size * intermediate_size * 2  # gate + down GEMMs
    # 不可缩放 ops (embedding, layernorm, output projection might scale differently)
    non_scalable_flops = layernorm_flops + out_proj_flops  # O projection usually scales

    total_layer_flops = scalable_flops + non_scalable_flops

    # 考虑层数权重
    num_layers = arch.num_layers

    # 估算可缩放和不可缩放 ops 在总 FLOPs 中的比例
    # 这里使用简化模型：假设每层结构类似
    per_layer_scalable_ratio = scalable_flops / total_layer_flops
    per_layer_non_scalable_ratio = non_scalable_flops / total_layer_flops

    # Embedding 和 final norm 不随层数变化
    embedding_flops = arch.vocab_size * hidden_size * 2
    final_norm_flops = hidden_size * 6

    # 计算整体缩放比例
    # 简化：scalable_ops / tp_size + non_scalable_ops * 1.0
    # 这里基于经验比例，attention+gemm 约占总 FLOPs 的 60-70%
    # 实际比例与模型架构、batch size、seq_len 相关

    # 使用更精确的 FLOPs 估算
    # Scalable: attention (QKV + O projections) + MLP gate/down projections
    scalable_per_layer = attention_flops + hidden_size * intermediate_size * 2
    non_scalable_per_layer = layernorm_flops + hidden_size * intermediate_size  # up projection

    total_per_layer = scalable_per_layer + non_scalable_per_layer

    scalable_ratio = scalable_per_layer / total_per_layer
    non_scalable_ratio = non_scalable_per_layer / total_per_layer

    # 整体缩放 = scalable_ops / tp_size + non_scalable_ops * 1.0
    effective_scale = (scalable_ratio / tp_size) + non_scalable_ratio

    return effective_scale


def compute_tp_flops_scale(
    arch: ModelArchitecture,
    tp_size: int,
) -> float:
    """
    计算 TP 模式下 FLOPs 缩放比例（用于计算时间）

    与 compute_effective_tp_scale 不同，本函数直接基于 FLOPs 比例计算。
    """
    if tp_size <= 1:
        return 1.0

    # Attention 在总 FLOPs 中占较大比例，约为 30-40%
    # MLP  GEMM 占 40-50%
    # 其他 (layernorm, embedding) 占 10-20%

    # 使用经验值
    attention_ratio = 0.35
    mlp_gemm_ratio = 0.45
    other_ratio = 0.20

    # Attention 和 GEMM 按 tp_size 缩放，other 不缩放
    scalable_ratio = attention_ratio + mlp_gemm_ratio  # 0.80

    # 整体缩放
    return (scalable_ratio / tp_size) + other_ratio


# =============================================================================
# Forward Pass TP 估算（复用推理逻辑）
# =============================================================================

def estimate_forward_with_tp(
    batch_size: int,
    seq_len: int,
    arch: ModelArchitecture,
    calibration: TrainCalibration,
    tp_size: int,
) -> float:
    """
    使用与推理相同的 per-op TP 缩放，估算 forward pass 时间

    参数：
        batch_size: 批次大小
        seq_len: 序列长度
        arch: 模型架构
        calibration: 训练校准数据
        tp_size: Tensor Parallelism 大小

    返回：
        Forward pass 时间 (ms)
    """
    from mvp_train_estimator import estimate_forward_flops

    if tp_size <= 1:
        # 无 TP，直接使用标准估算
        forward_flops = estimate_forward_flops(batch_size, seq_len, arch)
        effective_tflops = calibration.gemm_tflops * calibration.effective_tflops_scale
        compute_time_ms = forward_flops / (effective_tflops * 1e12) * 1e3

        # Memory time
        activation_memory_bytes = (
            batch_size * seq_len * arch.hidden_size * 4 * arch.num_layers * 2
        )
        memory_time_ms = (
            activation_memory_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
        )

        kernel_overhead_ms = arch.num_layers * calibration.kernel_overhead_factor
        return max(compute_time_ms, memory_time_ms) + kernel_overhead_ms

    # 使用精细化 TP 缩放
    effective_scale = compute_effective_tp_scale(arch, tp_size)

    forward_flops = estimate_forward_flops(batch_size, seq_len, arch)

    # 应用 TP 缩放到 FLOPs
    scaled_flops = forward_flops * effective_scale

    effective_tflops = calibration.gemm_tflops * 0.9
    compute_time_ms = scaled_flops / (effective_tflops * 1e12) * 1e3

    # Memory time 缩放（内存访问也受益于 TP）
    activation_memory_bytes = (
        batch_size * seq_len * arch.hidden_size * 4 * arch.num_layers * 2
    )
    scaled_memory_bytes = activation_memory_bytes * effective_scale
    memory_time_ms = (
        scaled_memory_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
    )

    kernel_overhead_ms = arch.num_layers * calibration.kernel_overhead_factor

    return max(compute_time_ms, memory_time_ms) + kernel_overhead_ms


def estimate_forward_with_tp_from_graph(
    forward_graph_nodes: list,
    num_parameters: int,
    arch: ModelArchitecture,
    calibration: TrainCalibration,
    config: TrainConfig,
    execution: ExecutionConfig,
) -> tuple[float, list]:
    """
    基于图结构估算带 TP 的 forward pass 时间

    复用 mvp_graph.py 的 tp_shard_node_estimate 函数进行精细化 per-op 缩放。

    参数：
        forward_graph_nodes: 从 NON-TP 模型提取的前向图节点列表
        num_parameters: 模型参数数量
        arch: 模型架构
        calibration: 训练校准数据
        config: 训练配置
        execution: 执行配置（包含 tp_size 等）

    返回：
        (forward_time_ms, scaled_nodes): 前向时间 (ms) 和缩放后的节点列表
    """
    if execution.parallel_mode != "tp" or execution.tp_size <= 1:
        # 无 TP，直接估算
        from mvp_train_estimator import estimate_forward_flops

        forward_flops = estimate_forward_flops(
            config.batch_size, config.seq_len, arch
        )
        effective_tflops = calibration.gemm_tflops * calibration.effective_tflops_scale
        compute_time_ms = forward_flops / (effective_tflops * 1e12) * 1e3

        activation_memory_bytes = (
            config.batch_size * config.seq_len * arch.hidden_size * 4 * arch.num_layers * 2
        )
        memory_time_ms = (
            activation_memory_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
        )

        kernel_overhead_ms = arch.num_layers * calibration.kernel_overhead_factor
        total_time_ms = max(compute_time_ms, memory_time_ms) + kernel_overhead_ms

        return total_time_ms, forward_graph_nodes

    # 使用 tp_shard_node_estimate 对每个节点进行精细化缩放
    scaled_nodes = []
    total_compute_time = 0.0
    total_memory_time = 0.0
    total_flops = 0.0
    total_bytes_moved = 0.0

    for node in forward_graph_nodes:
        scaled_node = tp_shard_node_estimate(node, execution)
        scaled_nodes.append(scaled_node)

        total_compute_time += scaled_node.compute_time_ms
        total_memory_time += scaled_node.memory_time_ms
        total_flops += scaled_node.flops
        total_bytes_moved += scaled_node.bytes_moved

    # Kernel launch 开销
    kernel_overhead_ms = len(forward_graph_nodes) * calibration.kernel_overhead_factor

    # 估算内存访问时间（使用更精确的带宽模型）
    # 考虑 TP 后，每个节点的内存访问量会减少
    memory_time_ms = (
        total_bytes_moved / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
    )

    total_time_ms = max(total_compute_time, memory_time_ms) + kernel_overhead_ms

    return total_time_ms, scaled_nodes


# =============================================================================
# 完整的 Forward TP 估算（与推理逻辑对齐）
# =============================================================================

def estimate_forward_phase_with_tp(
    batch_size: int,
    seq_len: int,
    arch: ModelArchitecture,
    calibration: TrainCalibration,
    tp_size: int,
    use精细化缩放: bool = True,
) -> dict[str, float]:
    """
    估算带 TP 的 forward pass 各组件时间

    返回详细的组件时间字典：
    - compute_time_ms: 计算时间
    - memory_time_ms: 内存访问时间
    - kernel_overhead_ms: Kernel 启动开销
    - total_time_ms: 总时间
    - effective_tp_scale: 有效 TP 缩放比例
    """
    from mvp_train_estimator import estimate_forward_flops

    if tp_size <= 1 or not use精细化缩放:
        effective_scale = 1.0
    else:
        effective_scale = compute_effective_tp_scale(arch, tp_size)

    forward_flops = estimate_forward_flops(batch_size, seq_len, arch)
    scaled_flops = forward_flops * effective_scale

    effective_tflops = calibration.gemm_tflops * 0.9
    compute_time_ms = scaled_flops / (effective_tflops * 1e12) * 1e3

    activation_memory_bytes = (
        batch_size * seq_len * arch.hidden_size * 4 * arch.num_layers * 2
    )
    scaled_memory_bytes = activation_memory_bytes * effective_scale
    memory_time_ms = (
        scaled_memory_bytes / (calibration.memory_bandwidth_gbps * 1e9) * 1e3
    )

    kernel_overhead_ms = arch.num_layers * calibration.kernel_overhead_factor

    total_time_ms = max(compute_time_ms, memory_time_ms) + kernel_overhead_ms

    return {
        "compute_time_ms": compute_time_ms,
        "memory_time_ms": memory_time_ms,
        "kernel_overhead_ms": kernel_overhead_ms,
        "total_time_ms": total_time_ms,
        "effective_tp_scale": effective_scale,
        "flops": scaled_flops,
        "bytes_moved": scaled_memory_bytes,
    }


# =============================================================================
# 集成接口：用于替换 estimate_train_step 中的 Forward 部分
# =============================================================================

def get_forward_estimator_for_tp(
    use_graph_based: bool = False,
):
    """
    获取适合当前配置的 Forward 估算器

    参数：
        use_graph_based: 是否使用基于图的估算（需要先提取图）

    返回：
        合适的 forward 估算函数
    """
    if use_graph_based:
        return estimate_forward_with_tp_from_graph
    else:
        return estimate_forward_with_tp


# =============================================================================
# 工具函数：分析 TP 缩放效果
# =============================================================================

def analyze_tp_scale_breakdown(
    arch: ModelArchitecture,
    tp_size: int,
) -> dict[str, Any]:
    """
    分析 TP 缩放对各组件的影响

    返回：
    - attention_scale: Attention 缩放比例
    - mlp_scale: MLP 缩放比例
    - other_scale: 其他操作缩放比例
    - effective_scale: 整体有效缩放比例
    - speedup_expected: 预期加速比
    """
    if tp_size <= 1:
        return {
            "attention_scale": 1.0,
            "mlp_scale": 1.0,
            "other_scale": 1.0,
            "effective_scale": 1.0,
            "speedup_expected": 1.0,
        }

    # Attention 缩放
    attention_scale = 1.0 / tp_size

    # MLP GEMM 缩放 (gate + down projections)
    mlp_scalable_ratio = 2.0 / 3.0  # gate 和 down 可缩放，up 不缩放
    mlp_scale = (mlp_scalable_ratio / tp_size) + (1 - mlp_scalable_ratio)

    # LayerNorm, embedding 等不缩放
    other_scale = 1.0

    effective_scale = compute_effective_tp_scale(arch, tp_size)
    speedup_expected = 1.0 / effective_scale

    return {
        "attention_scale": attention_scale,
        "mlp_scale": mlp_scale,
        "other_scale": other_scale,
        "effective_scale": effective_scale,
        "speedup_expected": speedup_expected,
        "tp_size": tp_size,
    }
