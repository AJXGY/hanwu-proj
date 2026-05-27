"""
TrainCalibration TP 扩展

为 TrainCalibration 添加 TP (Tensor Parallelism) 相关的校准参数。

================================================================================
此模块提供:
================================================================================

1. TPTrainCalibration - 扩展的校准类，包含 TP 所需的新参数

2. from_calibration() - 从现有 TrainCalibration 转换的工厂函数

3. get_default_calibration() - 获取带 TP 参数的默认校准

================================================================================
新增参数说明:
================================================================================

1. has_nvlink: bool = True
   - 含义: 是否使用 NVLink 互联
   - 影响: 决定使用 NVLink 延迟 (~0.3ms) 还是 PCIe 延迟 (~5ms)
   - 来源: 硬件检测或用户指定

2. overlap_ratio: float = 0.3
   - 含义: 计算-通信重叠比例
   - 范围: 0.0 - 1.0
   - 影响: 通信时间 * (1 - overlap_ratio) 为有效开销
   - 说明: CUDA graph 可提高重叠比例到 0.5-0.7

3. nvlink_bandwidth_gbps: float = 900.0
   - 含义: NVLink 有效带宽 (GB/s)
   - 典型值: A100 ~900 GB/s, H100 ~900 GB/s

4. pcie_bandwidth_gbps: float = 50.0
   - 含义: PCIe 有效带宽 (GB/s)
   - 典型值: PCIe 4.0 x16 ~50 GB/s

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, TypeVar

import torch

# 尝试导入现有 TrainCalibration，如果不存在则使用类型别名
try:
    from mvp_train_types import TrainCalibration
except ImportError:
    # 如果导入失败，定义一个临时的 TrainCalibration 类型
    # 这允许模块在没有完整依赖的情况下被加载（用于文档和类型检查）
    TrainCalibration = None  # type: ignore


# ============================================================================
# 常量
# ============================================================================

# NVLink 延迟 (ms)
NVLINK_LATENCY_MS = 0.3

# PCIe 延迟 (ms)
PCIE_LATENCY_MS = 5.0

# NVLink 有效带宽 (GB/s)
NVLINK_BANDWIDTH_GBPS = 900.0

# PCIe 有效带宽 (GB/s)
PCIE_BANDWIDTH_GBPS = 50.0

# 默认计算-通信重叠比例
DEFAULT_OVERLAP_RATIO = 0.3

# CUDA graph 启用时的重叠比例
CUDA_GRAPH_OVERLAP_RATIO = 0.5


# ============================================================================
# TPTrainCalibration 类
# ============================================================================

T = TypeVar("T", bound="TrainCalibration")


@dataclass
class TPTrainCalibration:
    """扩展的 TrainCalibration，包含 TP 所需的校准参数。

    此类的字段与 TrainCalibration 兼容，并添加了以下 TP 相关参数：
    - has_nvlink: 是否使用 NVLink
    - overlap_ratio: 计算-通信重叠比例
    - nvlink_bandwidth_gbps: NVLink 带宽
    - pcie_bandwidth_gbps: PCIe 带宽
    """

    # ---- 基础 TrainCalibration 字段 ----
    device_name: str = "cuda"
    device_index: int = 0
    gemm_tflops: float = 1000.0
    attention_tflops: float = 800.0
    memory_bandwidth_gbps: float = 1000.0
    launch_overhead_ms: float = 0.01
    backward_compute_scale: float = 2.5
    optimizer_steps_per_sec: float = 1000.0
    gradient_allreduce_tflops: float = 50.0

    # ---- TP 相关新增字段 ----
    has_nvlink: bool = True
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO
    nvlink_bandwidth_gbps: float = NVLINK_BANDWIDTH_GBPS
    pcie_bandwidth_gbps: float = PCIE_BANDWIDTH_GBPS

    @property
    def comm_latency_ms(self) -> float:
        """获取通信延迟 (ms)。"""
        return NVLINK_LATENCY_MS if self.has_nvlink else PCIE_LATENCY_MS

    @property
    def comm_bandwidth_gbps(self) -> float:
        """获取有效通信带宽 (GB/s)。"""
        return self.nvlink_bandwidth_gbps if self.has_nvlink else self.pcie_bandwidth_gbps

    def with_overlap_ratio(self, overlap_ratio: float) -> "TPTrainCalibration":
        """创建具有新重叠比例的副本。"""
        return TPTrainCalibration(
            device_name=self.device_name,
            device_index=self.device_index,
            gemm_tflops=self.gemm_tflops,
            attention_tflops=self.attention_tflops,
            memory_bandwidth_gbps=self.memory_bandwidth_gbps,
            launch_overhead_ms=self.launch_overhead_ms,
            backward_compute_scale=self.backward_compute_scale,
            optimizer_steps_per_sec=self.optimizer_steps_per_sec,
            gradient_allreduce_tflops=self.gradient_allreduce_tflops,
            has_nvlink=self.has_nvlink,
            overlap_ratio=overlap_ratio,
            nvlink_bandwidth_gbps=self.nvlink_bandwidth_gbps,
            pcie_bandwidth_gbps=self.pcie_bandwidth_gbps,
        )

    def with_nvlink(self, has_nvlink: bool) -> "TPTrainCalibration":
        """创建具有新 NVLink 设置的副本。"""
        return TPTrainCalibration(
            device_name=self.device_name,
            device_index=self.device_index,
            gemm_tflops=self.gemm_tflops,
            attention_tflops=self.attention_tflops,
            memory_bandwidth_gbps=self.memory_bandwidth_gbps,
            launch_overhead_ms=self.launch_overhead_ms,
            backward_compute_scale=self.backward_compute_scale,
            optimizer_steps_per_sec=self.optimizer_steps_per_sec,
            gradient_allreduce_tflops=self.gradient_allreduce_tflops,
            has_nvlink=has_nvlink,
            overlap_ratio=self.overlap_ratio,
            nvlink_bandwidth_gbps=self.nvlink_bandwidth_gbps,
            pcie_bandwidth_gbps=self.pcie_bandwidth_gbps,
        )


# ============================================================================
# 工厂函数
# ============================================================================


def from_calibration(
    calibration: TrainCalibration,
    has_nvlink: Optional[bool] = None,
    overlap_ratio: Optional[float] = None,
    nvlink_bandwidth_gbps: Optional[float] = None,
    pcie_bandwidth_gbps: Optional[float] = None,
) -> TPTrainCalibration:
    """从现有 TrainCalibration 创建 TPTrainCalibration。

    参数:
        calibration: 现有的 TrainCalibration 对象
        has_nvlink: 是否使用 NVLink (默认: True)
        overlap_ratio: 计算-通信重叠比例 (默认: 0.3)
        nvlink_bandwidth_gbps: NVLink 带宽 (默认: 900.0)
        pcie_bandwidth_gbps: PCIe 带宽 (默认: 50.0)

    返回:
        TPTrainCalibration: 扩展的校准对象
    """
    return TPTrainCalibration(
        # 基础字段
        device_name=calibration.device_name,
        device_index=calibration.device_index,
        gemm_tflops=calibration.gemm_tflops,
        attention_tflops=calibration.attention_tflops,
        memory_bandwidth_gbps=calibration.memory_bandwidth_gbps,
        launch_overhead_ms=calibration.launch_overhead_ms,
        backward_compute_scale=calibration.backward_compute_scale,
        optimizer_steps_per_sec=calibration.optimizer_steps_per_sec,
        gradient_allreduce_tflops=calibration.gradient_allreduce_tflops,
        # TP 扩展字段
        has_nvlink=has_nvlink if has_nvlink is not None else True,
        overlap_ratio=overlap_ratio if overlap_ratio is not None else DEFAULT_OVERLAP_RATIO,
        nvlink_bandwidth_gbps=nvlink_bandwidth_gbps if nvlink_bandwidth_gbps is not None else NVLINK_BANDWIDTH_GBPS,
        pcie_bandwidth_gbps=pcie_bandwidth_gbps if pcie_bandwidth_gbps is not None else PCIE_BANDWIDTH_GBPS,
    )


def get_default_calibration(
    device_name: str = "cuda",
    has_nvlink: bool = True,
) -> TPTrainCalibration:
    """获取 TP 训练的默认校准参数。

    参数:
        device_name: 设备名称
        has_nvlink: 是否使用 NVLink

    返回:
        TPTrainCalibration: 默认校准对象
    """
    return TPTrainCalibration(
        device_name=device_name,
        has_nvlink=has_nvlink,
        overlap_ratio=DEFAULT_OVERLAP_RATIO,
        nvlink_bandwidth_gbps=NVLINK_BANDWIDTH_GBPS,
        pcie_bandwidth_gbps=PCIE_BANDWIDTH_GBPS,
    )


def detect_hardware_capabilities() -> dict[str, any]:
    """检测硬件能力并返回推荐配置。

    此函数尝试检测：
    - GPU 类型 (通过 torch.cuda.get_device_name)
    - NVLink 可用性 (通过检查多 GPU 配置)

    注意: 这是一个简化实现，实际产品可能需要更详细的检测逻辑。

    返回:
        dict: 包含 has_nvlink, recommended_bandwidth 等字段
    """
    result = {
        "has_nvlink": True,  # 默认假设有 NVLink
        "recommended_bandwidth_gbps": NVLINK_BANDWIDTH_GBPS,
        "recommended_latency_ms": NVLINK_LATENCY_MS,
        "device_name": "unknown",
    }

    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            result["device_name"] = device_name

            # 检测是否为 NVIDIA 数据中心 GPU (通常有 NVLink)
            # A100, H100, V100 等
            datacenter_gpus = ["A100", "H100", "H800", "V100", "A30", "A40"]
            has_datacenter_gpu = any(gpu in device_name for gpu in datacenter_gpus)

            if has_datacenter_gpu and device_count > 1:
                result["has_nvlink"] = True
                result["recommended_bandwidth_gbps"] = NVLINK_BANDWIDTH_GBPS
                result["recommended_latency_ms"] = NVLINK_LATENCY_MS
            elif device_count > 1:
                # 多 GPU 但非数据中心型号，可能使用 PCIe
                result["has_nvlink"] = False
                result["recommended_bandwidth_gbps"] = PCIE_BANDWIDTH_GBPS
                result["recommended_latency_ms"] = PCIE_LATENCY_MS
        except Exception:
            pass

    return result


# ============================================================================
# 向后兼容扩展
# ============================================================================

def extend_train_calibration(
    calibration: TrainCalibration,
) -> TPTrainCalibration:
    """扩展现有的 TrainCalibration，添加 TP 相关字段的默认值。

    这是一个便捷函数，等同于 from_calibration(calibration)。

    参数:
        calibration: 现有的 TrainCalibration 对象

    返回:
        TPTrainCalibration: 扩展后的校准对象
    """
    return from_calibration(calibration)
