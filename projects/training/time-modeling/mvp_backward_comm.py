"""
分层 Backward 通信模型 (Task 2: 实现分层 Backward 通信模型)

本模块实现基于延迟-带宽分离模型的 backward 通信时间估算。

依赖：任务一 (extract_backward_graphs) 提取的梯度信息

公式：
    通信时间 = latency + gradient_bytes / bandwidth

其中：
1. latency: AllReduce 启动开销 (~0.3ms NVLink, ~5ms PCIe)
2. gradient_bytes: 从 backward 图获取的每层输出梯度大小 * tp_size
3. bandwidth: 有效带宽（考虑 ring all-reduce 的 2*(tp_size-1)/tp_size 因子）

重叠建模：
    最终时间 = max(计算时间, 通信时间 * (1 - overlap_ratio)) + kernel_overhead
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


# =============================================================================
# 假定 API（来自任务一 - extract_backward_graphs）
# =============================================================================
# 以下是从任务一获取的后向图数据结构，供参考：
#
# @dataclass
# class BackwardGraphNode:
#     """Backward 图中的单个节点"""
#     module_scope: str          # 模块作用域，如 "model.layers.0.self_attn"
#     op_family: str              # op 类型：gemm, attention, pointwise, etc.
#     output_shape: list[int]     # 输出张量形状
#     output_dtype: str           # 输出数据类型
#     grad_shape: list[int]       # 梯度张量形状
#     grad_dtype: str             # 梯度数据类型
#     bytes_moved: float          # 该节点移动的字节数
#
# def extract_backward_graphs(
#     model: torch.nn.Module,
#     input_ids: torch.Tensor,
#     attention_mask: torch.Tensor,
# ) -> dict[str, Any]:
#     """
#     从任务一实现的图提取函数。
#
#     返回 dict 包含：
#     - backward_nodes: list[BackwardGraphNode]  # 每层梯度节点
#     - num_layers: int                          # 层数
#     - total_grad_bytes: float                  # 总梯度字节数
#     - embedding_grad_bytes: float              # embedding 层梯度字节数
#     """
# =============================================================================


# =============================================================================
# 硬件参数 (Defaults - should be overridden by calibration/config)
# =============================================================================

# AllReduce 启动延迟 (ms)
# - NVLink: ~0.3ms
# - PCIe: ~5ms
# NOTE: These defaults are overridden by config/tp/communication settings
NVLINK_LATENCY_MS = 0.3
PCIE_LATENCY_MS = 5.0

# 带宽 (GB/s)
# - NVLink: ~900 GB/s per link, 2 links used for ring all-reduce
# - PCIe Gen4 x16: ~32 GB/s
# NOTE: These defaults are overridden by config/tp/communication settings
NVLINK_BANDWIDTH_GBPS = 450.0  # 有效带宽（考虑 ring 拓扑）
PCIE_BANDWIDTH_GBPS = 32.0


def get_comm_params_from_calibration(calibration: "TrainCalibration") -> tuple[float, float, float, float]:
    """Get communication parameters from calibration or use defaults.

    Returns (nvlink_latency_ms, pcie_latency_ms, nvlink_bandwidth_gbps, pcie_bandwidth_gbps)
    from calibration if available, otherwise returns hardcoded defaults.
    """
    # Try to get from tp.communication in calibration if it has that structure
    tp_cfg = getattr(calibration, 'tp_config', None)
    if tp_cfg is not None:
        comm_cfg = tp_cfg.get('communication', {})
        return (
            comm_cfg.get('nvlink_latency_ms', NVLINK_LATENCY_MS),
            comm_cfg.get('pcie_latency_ms', PCIE_LATENCY_MS),
            comm_cfg.get('nvlink_bandwidth_gbps', NVLINK_BANDWIDTH_GBPS),
            comm_cfg.get('pcie_bandwidth_gbps', PCIE_BANDWIDTH_GBPS),
        )
    return (NVLINK_LATENCY_MS, PCIE_LATENCY_MS, NVLINK_BANDWIDTH_GBPS, PCIE_BANDWIDTH_GBPS)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class BackwardCommEstimate:
    """Backward 通信估算结果"""
    total_comm_time_ms: float          # 总通信时间
    latency_ms: float                 # 延迟开销
    bandwidth_ms: float               # 带宽开销
    gradient_bytes: float             # 梯度总字节数
    num_allreduces: int               # AllReduce 次数
    comm_overhead_ms: float           # kernel launch 开销
    overlap_ratio: float              # 重叠比例
    effective_comm_time_ms: float     # 考虑重叠后的有效通信时间


@dataclass
class LayerGradientInfo:
    """单层梯度信息"""
    module_scope: str
    op_family: str
    grad_bytes: float
    is_tp_parallel: bool  # 是否是 TP 并行作用域


# =============================================================================
# 辅助函数
# =============================================================================

def is_tp_parallel_scope(scope: str) -> bool:
    """判断是否是 TP 并行作用域"""
    return ".self_attn" in scope or ".mlp" in scope


def get_latency_ms(has_nvlink: bool, calibration: "TrainCalibration | None" = None) -> float:
    """获取 AllReduce 启动延迟

    Args:
        has_nvlink: Whether NVLink is available
        calibration: Optional calibration object with comm params override
    """
    if calibration is not None:
        tp_cfg = getattr(calibration, 'tp_config', None)
        if tp_cfg is not None:
            comm_cfg = tp_cfg.get('communication', {})
            if has_nvlink:
                return comm_cfg.get('nvlink_latency_ms', NVLINK_LATENCY_MS)
            else:
                return comm_cfg.get('pcie_latency_ms', PCIE_LATENCY_MS)
    return NVLINK_LATENCY_MS if has_nvlink else PCIE_LATENCY_MS


def get_bandwidth_gbps(has_nvlink: bool, calibration: "TrainCalibration | None" = None) -> float:
    """获取有效带宽

    Args:
        has_nvlink: Whether NVLink is available
        calibration: Optional calibration object with comm params override
    """
    if calibration is not None:
        tp_cfg = getattr(calibration, 'tp_config', None)
        if tp_cfg is not None:
            comm_cfg = tp_cfg.get('communication', {})
            if has_nvlink:
                return comm_cfg.get('nvlink_bandwidth_gbps', NVLINK_BANDWIDTH_GBPS)
            else:
                return comm_cfg.get('pcie_bandwidth_gbps', PCIE_BANDWIDTH_GBPS)
    return NVLINK_BANDWIDTH_GBPS if has_nvlink else PCIE_BANDWIDTH_GBPS


def compute_ring_allreduce_efficiency(tp_size: int) -> float:
    """
    计算 ring all-reduce 的带宽效率因子。

    Ring AllReduce 传输因子：
    - 发送：2 * (tp_size - 1) / tp_size
    - 接收：同上
    - 总计：4 * (tp_size - 1) / tp_size，但有效数据只有一半

    简化模型：有效带宽利用率 ≈ 2 * (tp_size - 1) / tp_size
    """
    if tp_size <= 1:
        return 1.0
    return 2.0 * (tp_size - 1) / tp_size


def estimate_gradient_bytes_per_layer(
    backward_graph: dict[str, Any],
) -> list[LayerGradientInfo]:
    """
    从 backward 图中提取每层梯度信息。

    假设 backward_graph 包含 backward_nodes 列表。
    """
    nodes = backward_graph.get("backward_nodes", [])
    grad_infos: list[LayerGradientInfo] = []

    for node in nodes:
        scope = node.get("module_scope", "")
        op_family = node.get("op_family", "misc")
        grad_bytes = node.get("grad_bytes", 0.0)

        grad_infos.append(LayerGradientInfo(
            module_scope=scope,
            op_family=op_family,
            grad_bytes=grad_bytes,
            is_tp_parallel=is_tp_parallel_scope(scope),
        ))

    return grad_infos


# =============================================================================
# 核心估算函数
# =============================================================================

def estimate_backward_comm_time(
    backward_graph: dict[str, Any],
    tp_size: int,
    calibration: "TrainCalibration",
) -> BackwardCommEstimate:
    """
    分层估算 backward 通信时间（延迟-带宽分离模型）。

    通信时间 = latency + gradient_bytes / bandwidth

    参数:
        backward_graph: 从 extract_backward_graphs 获取的后向图信息
        tp_size: Tensor Parallel 大小
        calibration: 训练校准参数，包含:
            - has_nvlink: bool  # 是否使用 NVLink
            - overlap_ratio: float  # 计算-通信重叠比例 (0.0-1.0)

    返回:
        BackwardCommEstimate: 包含详细通信时间分解
    """
    # 处理非 TP 情况
    if tp_size <= 1:
        return BackwardCommEstimate(
            total_comm_time_ms=0.0,
            latency_ms=0.0,
            bandwidth_ms=0.0,
            gradient_bytes=0.0,
            num_allreduces=0,
            comm_overhead_ms=0.0,
            overlap_ratio=0.0,
            effective_comm_time_ms=0.0,
        )

    # 获取硬件参数
    has_nvlink = getattr(calibration, "has_nvlink", True)
    overlap_ratio = getattr(calibration, "overlap_ratio", 0.3)

    latency_ms = get_latency_ms(has_nvlink, calibration)
    raw_bandwidth_gbps = get_bandwidth_gbps(has_nvlink, calibration)

    # 计算 ring all-reduce 效率因子
    ring_efficiency = compute_ring_allreduce_efficiency(tp_size)
    effective_bandwidth_gbps = raw_bandwidth_gbps * ring_efficiency

    # 提取梯度信息
    grad_infos = estimate_gradient_bytes_per_layer(backward_graph)

    # 计算总梯度字节数
    # 注意：每个 TP rank 持有完整的梯度副本，所以需要 * tp_size
    total_grad_bytes = 0.0
    tp_grad_bytes = 0.0  # TP 并行层需要 AllReduce 的梯度

    for info in grad_infos:
        total_grad_bytes += info.grad_bytes
        if info.is_tp_parallel:
            tp_grad_bytes += info.grad_bytes * tp_size

    # 对 embedding/lm_head 等非 TP 层，梯度也需要 AllReduce（DDP 行为）
    # 这里简化为：TP 层内 AllReduce + embedding 层 AllReduce
    num_allreduces = len([i for i in grad_infos if i.is_tp_parallel]) + 1  # +1 for embedding

    # 计算带宽开销
    # bandwidth_gbps 是 GB/s (10^9 bytes/s), tp_grad_bytes 是 bytes
    # bytes / (GB/s) = bytes / (10^9 bytes/s) = s * 10^-9
    # 转换为 ms: * 1000, 所以 total = bytes / (GB/s * 1e9) * 1000 = bytes / (GB/s * 1e6)
    bandwidth_ms = tp_grad_bytes / (effective_bandwidth_gbps * 1e6)

    # 通信时间 = latency + bandwidth
    comm_time_ms = latency_ms + bandwidth_ms

    # kernel launch 开销
    comm_overhead_ms = num_allreduces * 0.05  # 每次 AllReduce 约 0.05ms overhead

    # 考虑重叠的有效通信时间
    # 重叠建模: effective_comm = comm * (1 - overlap_ratio) + overhead
    effective_comm_time_ms = comm_time_ms * (1.0 - overlap_ratio) + comm_overhead_ms

    return BackwardCommEstimate(
        total_comm_time_ms=comm_time_ms,
        latency_ms=latency_ms,
        bandwidth_ms=bandwidth_ms,
        gradient_bytes=tp_grad_bytes,
        num_allreduces=num_allreduces,
        comm_overhead_ms=comm_overhead_ms,
        overlap_ratio=overlap_ratio,
        effective_comm_time_ms=effective_comm_time_ms,
    )


def estimate_backward_with_comm(
    backward_graph: dict[str, Any],
    backward_compute_time_ms: float,
    tp_size: int,
    calibration: "TrainCalibration",
) -> tuple[float, BackwardCommEstimate]:
    """
    估算考虑通信开销后的 backward 总时间。

    最终时间 = max(计算时间, 通信时间 * (1 - overlap_ratio)) + kernel_overhead

    参数:
        backward_graph: 后向图信息
        backward_compute_time_ms: 后向计算时间（不含通信）
        tp_size: TP 大小
        calibration: 校准参数

    返回:
        (total_backward_time_ms, comm_estimate)
    """
    comm_estimate = estimate_backward_comm_time(backward_graph, tp_size, calibration)

    # 计算考虑重叠后的总时间
    # 如果计算时间 > 通信时间 * (1 - overlap_ratio)，则可以完全重叠
    overlapped_comm = comm_estimate.effective_comm_time_ms
    compute_vs_comm = backward_compute_time_ms - overlapped_comm

    if compute_vs_comm > 0:
        # 计算时间更长，通信完全被隐藏
        total_time_ms = backward_compute_time_ms + comm_estimate.comm_overhead_ms
    else:
        # 通信时间更长，需要额外时间
        total_time_ms = overlapped_comm + comm_estimate.comm_overhead_ms

    return total_time_ms, comm_estimate


# =============================================================================
# 与现有 TrainCalibration 集成
# =============================================================================

@dataclass
class CommCalibration:
    """通信校准参数（独立于 TrainCalibration）"""
    has_nvlink: bool = True
    overlap_ratio: float = 0.3  # 计算-通信重叠比例
    # 延迟参数 (ms)
    nvlink_latency_ms: float = 0.3
    pcie_latency_ms: float = 5.0
    # 带宽参数 (GB/s)
    nvlink_bandwidth_gbps: float = 450.0
    pcie_bandwidth_gbps: float = 32.0


def create_train_calibration_with_comm_params(
    base_calibration: "TrainCalibration",
    has_nvlink: bool = True,
    overlap_ratio: float = 0.3,
) -> tuple["TrainCalibration", CommCalibration]:
    """
    为 TrainCalibration 添加通信相关参数。

    返回 (base_calibration, CommCalibration) 元组，
    这样可以在不修改 TrainCalibration 的情况下传递通信参数。

    使用方式:
        base_cal, comm_cal = create_train_calibration_with_comm_params(...)
        result = estimate_backward_comm_simple(..., calibration=comm_cal)
    """
    comm_cal = CommCalibration(
        has_nvlink=has_nvlink,
        overlap_ratio=overlap_ratio,
    )
    return base_calibration, comm_cal


# =============================================================================
# 简化接口（不依赖 extract_backward_graphs）
# =============================================================================

def estimate_backward_comm_simple(
    num_layers: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    tp_size: int,
    calibration: "TrainCalibration",
    num_parameters: int,
) -> BackwardCommEstimate:
    """
    简化版 backward 通信估算（不依赖 extract_backward_graphs）。

    使用经验公式估算梯度大小：
    - 激活梯度: batch_size * seq_len * hidden_size * 4 bytes (fp32) per layer per rank
    - 权重梯度: num_parameters * 4 bytes (fp32) - 每个 rank 计算完整权重梯度
    - TP AllReduce: (activation_grad_per_layer * num_layers + weight_grad_bytes) * tp_size

    参数:
        num_layers: 模型层数
        hidden_size: 隐藏维度
        batch_size: 批次大小
        seq_len: 序列长度
        tp_size: TP 大小
        calibration: 校准参数
        num_parameters: 模型总参数量
    """
    if tp_size <= 1:
        return BackwardCommEstimate(
            total_comm_time_ms=0.0,
            latency_ms=0.0,
            bandwidth_ms=0.0,
            gradient_bytes=0.0,
            num_allreduces=0,
            comm_overhead_ms=0.0,
            overlap_ratio=0.0,
            effective_comm_time_ms=0.0,
        )

    # 激活梯度: batch_size * seq_len * hidden_size * 4 (fp32) per layer per rank
    activation_grad_per_layer = batch_size * seq_len * hidden_size * 4

    # 权重梯度: num_parameters * 4 (fp32) - 每个 rank 计算完整权重梯度
    weight_grad_bytes = num_parameters * 4

    # TP 层内需要 AllReduce
    has_nvlink = getattr(calibration, "has_nvlink", True)
    overlap_ratio = getattr(calibration, "overlap_ratio", 0.3)

    latency_ms = get_latency_ms(has_nvlink, calibration)
    raw_bandwidth_gbps = get_bandwidth_gbps(has_nvlink, calibration)
    ring_efficiency = compute_ring_allreduce_efficiency(tp_size)
    effective_bandwidth_gbps = raw_bandwidth_gbps * ring_efficiency

    # 总梯度字节数 = (激活梯度 + 权重梯度) * tp_size
    tp_grad_bytes = (activation_grad_per_layer * num_layers + weight_grad_bytes) * tp_size

    # 带宽开销: bytes / (GB/s * 1e9) * 1000 = ms
    # bytes / (GB/s * 1e9) = seconds, then * 1000 = ms
    bandwidth_ms = tp_grad_bytes / (effective_bandwidth_gbps * 1e9) * 1000

    # 通信时间
    comm_time_ms = latency_ms + bandwidth_ms

    # 开销
    num_allreduces = num_layers
    comm_overhead_ms = num_allreduces * 0.05

    # 有效通信时间
    effective_comm_time_ms = comm_time_ms * (1.0 - overlap_ratio) + comm_overhead_ms

    return BackwardCommEstimate(
        total_comm_time_ms=comm_time_ms,
        latency_ms=latency_ms,
        bandwidth_ms=bandwidth_ms,
        gradient_bytes=tp_grad_bytes,
        num_allreduces=num_allreduces,
        comm_overhead_ms=comm_overhead_ms,
        overlap_ratio=overlap_ratio,
        effective_comm_time_ms=effective_comm_time_ms,
    )
