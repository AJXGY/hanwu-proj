"""
Optimizer TP 通信模型 - 延迟-带宽分离实现

任务四：实现延迟-带宽分离的 Optimizer 通信模型

================================================================================
假设的依赖 API（来自其他任务，可能尚未实现）:
================================================================================

1. TrainCalibration (来自 mvp_train_types.py)
   - 假设已有字段: device_name, gemm_tflops, memory_bandwidth_gbps
   - 需要新增字段 (如果尚未存在):
     - has_nvlink: bool = True  # 是否使用 NVLink
     - overlap_ratio: float = 0.3  # 计算-通信重叠比例

2. ModelArchitecture (来自 mvp_train_types.py)
   - 假设已有字段: num_layers, hidden_size, parameters
   - 需要参数数量: parameters (int)

3. estimate_optimizer_flops() (来自 mvp_train_estimator.py)
   - 假设签名: estimate_optimizer_flops(num_parameters, batch_size, seq_len, calibration)
   - 返回: tuple[flops, bytes_moved]

================================================================================
核心公式:
================================================================================

optimizer_time = latency + param_bytes / bandwidth

TP Optimizer 行为:
- 参数分片后每个 rank 持有完整 optimizer 状态副本
- 更新时无需通信（每个 rank 独立更新自己的分片）
- 所以纯 TP 的 optimizer 时间与 tp_size 无关（只有 DDP+TP 需要 AllReduce）

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ============================================================================
# 常量定义 (Imported from mvp_backward_comm.py as source of truth)
# ============================================================================
# NOTE: All communication params should come from calibration.tp_config['communication']
# The module-level constants below are only fallbacks for backward compatibility.
# Import them from mvp_backward_comm to ensure consistency:
#   from mvp_backward_comm import (
#       NVLINK_LATENCY_MS, PCIE_LATENCY_MS,
#       NVLINK_BANDWIDTH_GBPS, PCIE_BANDWIDTH_GBPS,
#   )

# Default values (should match mvp_backward_comm.py)
_NVLINK_LATENCY_MS = 0.3
_PCIE_LATENCY_MS = 5.0
_NVLINK_BANDWIDTH_GBPS = 450.0  # Must match config: tp.communication.nvlink_bandwidth_gbps
_PCIE_BANDWIDTH_GBPS = 32.0  # Must match config: tp.communication.pcie_bandwidth_gbps

# Adam optimizer 每个参数字节数 (fp32 params + 2 fp32 moments = 12 bytes)
ADAM_PARAM_BYTES = 12  # 4 bytes (param) + 4 bytes (m1) + 4 bytes (m2)

# 校准因子：实际 optimizer 吞吐约为理论带宽的 45-50%
# Adam优化器是内存密集型操作，实际测量显示效率约45%
# NOTE: This default is overridden by config/tp/communication/optimizer_efficiency
DEFAULT_OPTIMIZER_EFFICIENCY = 0.45


def get_optimizer_efficiency(calibration: "TrainCalibration | None" = None) -> float:
    """Get optimizer efficiency from calibration or return default.

    Args:
        calibration: Optional calibration object with tp_config.communication.optimizer_efficiency

    Returns:
        Optimizer efficiency factor (0.0 - 1.0)
    """
    if calibration is not None:
        tp_cfg = getattr(calibration, 'tp_config', None)
        if tp_cfg is not None:
            comm_cfg = tp_cfg.get('communication', {})
            return comm_cfg.get('optimizer_efficiency', DEFAULT_OPTIMIZER_EFFICIENCY)
    return DEFAULT_OPTIMIZER_EFFICIENCY


# Backward compatibility: OPTIMIZER_EFFICIENCY as module-level constant
# Use get_optimizer_efficiency() for config-aware retrieval
OPTIMIZER_EFFICIENCY = DEFAULT_OPTIMIZER_EFFICIENCY


# ============================================================================
# 带宽估算辅助函数
# ============================================================================

def get_comm_latency_ms(has_nvlink: bool, calibration: "TrainCalibration | None" = None) -> float:
    """获取通信延迟 (ms)。

    Args:
        has_nvlink: Whether NVLink is available
        calibration: Optional calibration object with comm params override
    """
    if calibration is not None:
        tp_cfg = getattr(calibration, 'tp_config', None)
        if tp_cfg is not None:
            comm_cfg = tp_cfg.get('communication', {})
            if has_nvlink:
                return comm_cfg.get('nvlink_latency_ms', _NVLINK_LATENCY_MS)
            else:
                return comm_cfg.get('pcie_latency_ms', _PCIE_LATENCY_MS)
    return _NVLINK_LATENCY_MS if has_nvlink else _PCIE_LATENCY_MS


def get_comm_bandwidth_gbps(has_nvlink: bool, calibration: "TrainCalibration | None" = None) -> float:
    """获取有效通信带宽 (GB/s)。

    Args:
        has_nvlink: Whether NVLink is available
        calibration: Optional calibration object with comm params override
    """
    if calibration is not None:
        tp_cfg = getattr(calibration, 'tp_config', None)
        if tp_cfg is not None:
            comm_cfg = tp_cfg.get('communication', {})
            if has_nvlink:
                return comm_cfg.get('nvlink_bandwidth_gbps', _NVLINK_BANDWIDTH_GBPS)
            else:
                return comm_cfg.get('pcie_bandwidth_gbps', _PCIE_BANDWIDTH_GBPS)
    return _NVLINK_BANDWIDTH_GBPS if has_nvlink else _PCIE_BANDWIDTH_GBPS


def compute_ring_allreduce_efficiency(tp_size: int) -> float:
    """计算 ring all-reduce 的效率因子。

    Ring all-reduce 效率 = 2 * (tp_size - 1) / tp_size
    对于 tp_size=2: 效率 = 0.5
    对于 tp_size=8: 效率 = 0.875
    """
    if tp_size <= 1:
        return 1.0
    return 2.0 * (tp_size - 1) / tp_size


# ============================================================================
# 核心估算函数
# ============================================================================


def estimate_optimizer_tp_overhead(
    num_parameters: int,
    tp_size: int,
    ddp_enabled: bool,
    has_nvlink: bool = True,
    overlap_ratio: float = 0.3,
) -> dict[str, float]:
    """
    估算 TP/DDP 模式下 optimizer 的通信开销。

    参数:
        num_parameters: 模型参数量
        tp_size: Tensor Parallelism 大小
        ddp_enabled: 是否启用 DDP
        has_nvlink: 是否使用 NVLink (否则使用 PCIe)
        overlap_ratio: 计算-通信重叠比例 (0.0 - 1.0)

    返回:
        dict，包含:
        - comm_time_ms: 通信时间 (ms)
        - latency_ms: 延迟开销 (ms)
        - bandwidth_time_ms: 带宽时间 (ms)
        - param_bytes: 需要通信的参数字节数
        - effective_bandwidth_gbps: 有效带宽 (GB/s)

    =============================================================================
    关键说明:
    =============================================================================

    纯 TP 模式 (tp_size > 1, ddp_enabled = False):
    - 每个 TP rank 持有完整的 optimizer 状态副本
    - 每个 rank 独立更新自己的参数分片
    - 无需通信，所以通信开销 = 0

    DDP 模式 (ddp_enabled = True, tp_size = 1):
    - 需要 AllReduce 同步梯度
    - 通信量 = 梯度字节数 * 2 (bidirectional)

    DDP + TP 模式 (ddp_enabled = True, tp_size > 1):
    - 同时需要 TP AllReduce 和 DDP AllReduce
    - TP AllReduce: 嵌入层和 LM head 的梯度
    - DDP AllReduce: 所有参数的梯度
    """
    result = {
        "comm_time_ms": 0.0,
        "latency_ms": 0.0,
        "bandwidth_time_ms": 0.0,
        "param_bytes": 0.0,
        "effective_bandwidth_gbps": get_comm_bandwidth_gbps(has_nvlink),
    }

    # 纯 TP 模式：每个 rank 独立更新，无需通信
    if tp_size > 1 and not ddp_enabled:
        return result

    # 单卡模式：无通信
    if tp_size <= 1 and not ddp_enabled:
        return result

    # DDP 模式：需要 AllReduce 通信
    if ddp_enabled:
        # 梯度大小 (fp32)
        gradient_bytes = num_parameters * 4  # 4 bytes per param

        # Ring all-reduce 通信量 (bidirectional)
        if tp_size > 1:
            # DDP + TP: 两阶段 AllReduce
            # 1. TP AllReduce (嵌入层, LM head)
            # 2. DDP AllReduce (所有参数)
            # 简化估算：2x 通信量
            total_bytes = gradient_bytes * 2 * compute_ring_allreduce_efficiency(tp_size)
        else:
            # 纯 DDP: 单阶段 AllReduce
            total_bytes = gradient_bytes * 2 * compute_ring_allreduce_efficiency(2)

        result["param_bytes"] = total_bytes

        # 延迟
        latency_ms = get_comm_latency_ms(has_nvlink)
        result["latency_ms"] = latency_ms

        # 带宽时间
        bandwidth = get_comm_bandwidth_gbps(has_nvlink)
        result["effective_bandwidth_gbps"] = bandwidth
        bandwidth_time_ms = (total_bytes / (bandwidth * 1e9)) * 1e3
        result["bandwidth_time_ms"] = bandwidth_time_ms

        # 总通信时间（考虑重叠）
        comm_time_ms = latency_ms + bandwidth_time_ms * (1.0 - overlap_ratio)
        result["comm_time_ms"] = comm_time_ms

    return result


def estimate_optimizer_time_latency_bandwidth(
    num_parameters: int,
    tp_size: int,
    ddp_enabled: bool,
    has_nvlink: bool = True,
    overlap_ratio: float = 0.3,
    memory_bandwidth_gbps: float = 1000.0,
    optimizer_efficiency: float = OPTIMIZER_EFFICIENCY,
) -> dict[str, float]:
    """
    完整估算 optimizer 步骤时间，使用延迟-带宽分离模型。

    公式:
        optimizer_time = memory_time + comm_time
        memory_time = param_bytes * 3 / (memory_bandwidth * optimizer_efficiency)
        comm_time = latency + comm_bytes / bandwidth

    参数:
        num_parameters: 模型参数量
        tp_size: Tensor Parallelism 大小
        ddp_enabled: 是否启用 DDP
        has_nvlink: 是否使用 NVLink
        overlap_ratio: 计算-通信重叠比例
        memory_bandwidth_gbps: 内存带宽 (GB/s)
        optimizer_efficiency: optimizer 实际效率 (0.0 - 1.0)

    返回:
        dict，包含:
        - total_time_ms: 总时间 (ms)
        - memory_time_ms: 内存访问时间 (ms)
        - comm_time_ms: 通信时间 (ms)
        - compute_time_ms: 计算时间 (ms, 通常很小)
        - latency_breakdown: 延迟分解
        - bandwidth_breakdown: 带宽分解
    """
    result = {
        "total_time_ms": 0.0,
        "memory_time_ms": 0.0,
        "comm_time_ms": 0.0,
        "compute_time_ms": 0.0,
        "latency_breakdown": {},
        "bandwidth_breakdown": {},
    }

    # ===== 1. 内存访问时间 =====
    # Adam: 每个参数需要读写 3 个 tensor (param, m1, m2)
    # 每个 tensor 是 fp32 = 4 bytes
    adam_bytes_per_param = ADAM_PARAM_BYTES  # 12 bytes
    total_memory_bytes = num_parameters * adam_bytes_per_param

    # 考虑 optimizer 实际效率
    effective_memory_bandwidth = memory_bandwidth_gbps * optimizer_efficiency
    memory_time_ms = (total_memory_bytes / (effective_memory_bandwidth * 1e9)) * 1e3
    result["memory_time_ms"] = memory_time_ms

    # ===== 2. 计算时间 (很小，optimizer 是 memory-bound) =====
    # Adam 每个参数约 15 FLOPs
    adam_flops_per_param = 15.0
    total_flops = num_parameters * adam_flops_per_param
    # 假设计算吞吐为 1000 TFLOPs (实际 optimizer 计算很轻量)
    compute_tflops = 1000.0
    compute_time_ms = (total_flops / (compute_tflops * 1e12)) * 1e3
    result["compute_time_ms"] = compute_time_ms

    # ===== 3. 通信开销 =====
    comm_result = estimate_optimizer_tp_overhead(
        num_parameters=num_parameters,
        tp_size=tp_size,
        ddp_enabled=ddp_enabled,
        has_nvlink=has_nvlink,
        overlap_ratio=overlap_ratio,
    )
    result["comm_time_ms"] = comm_result["comm_time_ms"]
    result["latency_breakdown"] = {
        "ms": comm_result["latency_ms"],
        "type": "nvlink" if has_nvlink else "pcie",
    }
    result["bandwidth_breakdown"] = {
        "ms": comm_result["bandwidth_time_ms"],
        "bytes": comm_result["param_bytes"],
        "effective_gbps": comm_result["effective_bandwidth_gbps"],
    }

    # ===== 4. 总时间 =====
    # optimizer 是 memory-bound，取 memory_time 和 compute_time 的较大值
    compute_memory_time = max(memory_time_ms, compute_time_ms)
    result["total_time_ms"] = compute_memory_time + comm_result["comm_time_ms"]

    return result


def compute_effective_tp_scale_for_optimizer(
    tp_size: int,
    ddp_enabled: bool,
) -> float:
    """
    计算 TP 对 optimizer 时间的有效缩放因子。

    关键洞察（来自 TP.md）：
    - 纯 TP 模式：每个 rank 独立更新自己的分片，optimizer 时间与 tp_size 无关
    - DDP + TP 模式：需要额外的梯度 AllReduce，tp_size 增加会略微增加通信时间

    返回:
        float: 有效缩放因子 (通常为 1.0 for pure TP)
    """
    if tp_size <= 1:
        return 1.0

    if not ddp_enabled:
        # 纯 TP: 无额外通信，每个 rank 独立工作
        return 1.0
    else:
        # DDP + TP: 有通信开销，但主要取决于 DDP
        # 简化：轻微增加，但不是线性关系
        # 实际上 ring all-reduce 的有效带宽随 tp_size 增加而增加
        return 1.0 + 0.1 * (tp_size - 1) / tp_size


# ============================================================================
# 与现有 TrainCalibration 兼容的包装器
# ============================================================================

def estimate_optimizer_with_calibration(
    num_parameters: int,
    calibration: "TrainCalibration",  # 类型提示，假设存在
    tp_size: int = 1,
    ddp_enabled: bool = False,
) -> dict[str, float]:
    """
    使用 TrainCalibration 对象估算 optimizer 时间。

    此函数作为桥梁，连接新的延迟-带宽模型与现有的 TrainCalibration 接口。

    参数:
        num_parameters: 模型参数量
        calibration: TrainCalibration 对象
            - 假设已有: memory_bandwidth_gbps
            - 假设可选: has_nvlink (默认 True), overlap_ratio (默认 0.3)
        tp_size: Tensor Parallelism 大小
        ddp_enabled: 是否启用 DDP

    返回:
        dict，包含 optimizer 时间分解
    """
    # 从 calibration 提取参数，提供默认值以防字段不存在
    has_nvlink = getattr(calibration, "has_nvlink", True)
    overlap_ratio = getattr(calibration, "overlap_ratio", 0.3)
    memory_bandwidth_gbps = calibration.memory_bandwidth_gbps

    return estimate_optimizer_time_latency_bandwidth(
        num_parameters=num_parameters,
        tp_size=tp_size,
        ddp_enabled=ddp_enabled,
        has_nvlink=has_nvlink,
        overlap_ratio=overlap_ratio,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
    )


# ============================================================================
# 便捷函数
# ============================================================================

def quick_estimate_optimizer_time(
    num_parameters: int,
    tp_size: int = 1,
    ddp_enabled: bool = False,
    has_nvlink: bool = True,
) -> float:
    """
    快速估算 optimizer 时间 (ms)。

    使用典型值:
    - memory_bandwidth_gbps = 1000.0 (A100)
    - optimizer_efficiency = 0.65
    - overlap_ratio = 0.3
    """
    result = estimate_optimizer_time_latency_bandwidth(
        num_parameters=num_parameters,
        tp_size=tp_size,
        ddp_enabled=ddp_enabled,
        has_nvlink=has_nvlink,
        overlap_ratio=0.3,
        memory_bandwidth_gbps=1000.0,
    )
    return result["total_time_ms"]
