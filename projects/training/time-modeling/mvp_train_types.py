from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainCalibration:
    """Training-specific hardware calibration.

    =============================================================================
    关键参数说明 (KEY PARAMETERS):
    =============================================================================

    1. gemm_tflops: float (从 mvp_calibration 复用)
       - 含义: GEMM操作的峰值吞吐量 (TFLOPS)
       - 来源: benchmark_linear_tflops()
       - 影响: Forward/Backward/Optimizer compute时间估算

    2. memory_bandwidth_gbps: float (从 mvp_calibration 复用)
       - 含义: 内存带宽 (GB/s)
       - 来源: benchmark_memory_bandwidth_gbps()
       - 影响: 所有memory-bound操作的时间估算

    3. backward_compute_scale: float = 2.5 → 建议调至 3.5
       - 含义: 反向传播FLOPs相对于前向传播的倍数
       - 经验值: 实际测量约为3-4倍
       - 影响: 反向传播时间估算 (~20-30% 误差)
       - 调参建议: 通过 calibrate_train_params.py 调整

    4. optimizer_scale_factor: float = 1.4
       - 含义: Adam优化器时间校准系数
       - 原因: Adam优化器实际吞吐约为理论内存带宽的60-70%
       - 影响: 优化器时间估算 (~40-60% 误差如果不校准)

    5. attention_tflops: float (从 mvp_calibration 复用)
       - 含义: Attention操作的峰值算力
       - 注意: 训练时此值可能不适用，代码中会乘以0.8折扣
       - 影响: 较小 (Attention FLOPs在整体中占比较小)

    =============================================================================
    """
    device_name: str
    device_index: int
    gemm_tflops: float
    attention_tflops: float
    memory_bandwidth_gbps: float
    launch_overhead_ms: float
    # Training-specific parameters
    # [KEY PARAMETER] 反向传播FLOPs放大系数 (默认2.5, 建议3.0-3.5)
    # 原因: 反向需要计算激活梯度+权重梯度，并涉及额外归约操作
    backward_compute_scale: float = 2.5
    optimizer_scale_factor: float = 1.4  # Adam优化器时间校准系数 (来自 config/common/optimizer/scale_factor)
    # Forward有效TFLOPs缩放因子: 实际forward有效TFLOPs = gemm_tflops * effective_tflops_scale
    effective_tflops_scale: float = 0.9
    # Backward有效TFLOPs缩放因子: 实际backward有效TFLOPs = gemm_tflops * backward_efficiency_scale
    # 这是需要校准的参数，反映后向传播的实际效率
    backward_efficiency_scale: float = 0.07
    # Forward每层kernel launch overhead系数
    kernel_overhead_factor: float = 0.15
    # Forward并行度因子: 串行node估算需要乘以这个因子来反映GPU并行执行
    # 实测 Llama-3.2-3B: serial_sum ≈ 25ms, measured ≈ 19ms, ratio ≈ 0.76
    forward_parallelism_factor: float = 0.76
    # Backward并行度因子: 估算的backward时间需要乘以这个因子
    parallelism_factor: float = 0.25
    # Backward overhead缩放因子
    overhead_scale: float = 0.3
    # TP-related parameters (added for Missions 2, 4)
    has_nvlink: bool = True  # 是否使用 NVLink (影响通信延迟估算)
    overlap_ratio: float = 0.3  # 计算-通信重叠比例 (CUDA graph可提高至0.5)
    # TP backward效率因子: 经验值约0.03-0.13 (3-13%)
    # 含义: TP模式下backward实际效率相对于peak TFLOPs的比例
    # 来源: 实测Llama-3.2-3B TP=2时，backward_effective_tflops约为gemm_tflops*tp_backward_efficiency
    tp_backward_efficiency: float = 0.13
    # TP forward效率因子: 用于校准TP forward时间估算
    tp_forward_efficiency: float = 0.05
    # DDP梯度AllReduce有效带宽（GB/s）
    gradient_allreduce_tflops: float = 100.0
    # Full TP config section from config file (for communication params access)
    tp_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainPhaseSummary:
    """Summary of training phase estimation."""
    phase: str  # "forward", "backward", "optimizer"
    estimated_time_ms: float
    flops: float
    bytes_moved: float
    compute_time_ms: float
    memory_time_ms: float
    comm_time_ms: float = 0.0
    node_count: int = 0
    top_ops: list[dict[str, Any]] = field(default_factory=list)
    op_family_breakdown_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainStepEstimate:
    """Complete training step estimation."""
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    total_time_ms: float
    samples_per_sec: float
    tokens_per_sec: float | None = None
    forward_summary: TrainPhaseSummary | None = None
    backward_summary: TrainPhaseSummary | None = None
    optimizer_summary: TrainPhaseSummary | None = None


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 1
    seq_len: int = 512
    num_epochs: int = 1
    global_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ddp_enabled: bool = False
    tp_size: int = 1
    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    vocab_size: int | None = None


@dataclass
class ModelArchitecture:
    """Model architecture information for training estimation."""
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    intermediate_size: int | None = None
    model_type: str = "causal_lm"
    adapter_param_count: int | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def parameters(self) -> int:
        """Approximate parameter count."""
        # Simplified estimate: embedding + layers * (attention + mlp) + lm_head
        vocab_params = self.vocab_size * self.hidden_size
        layer_params = self.num_layers * (
            # Attention
            4 * self.hidden_size * self.hidden_size +  # Q, K, V, O projections
            # MLP
            (self.intermediate_size or 4 * self.hidden_size) * self.hidden_size * 2 +  # Gate and Up, Down
            # Layer norms
            2 * 2 * self.hidden_size
        )
        head_params = self.vocab_size * self.hidden_size
        return vocab_params + layer_params + head_params

    @property
    def optimizer_param_count(self) -> int:
        """Parameter count for optimizer estimation (adapter params in LoRA mode)."""
        return self.adapter_param_count if self.adapter_param_count is not None else self.parameters
