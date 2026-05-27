from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "config/train_config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file.

    Searches in this order:
    1. Absolute path or relative path as given
    2. Relative to this module's directory (for multi-node runs where cwd != project root)
    3. Empty dict if not found (graceful fallback)
    """
    path = Path(config_path)
    if not path.exists():
        # Try relative to this module's directory
        module_dir = Path(__file__).parent
        alt_path = module_dir / config_path
        if alt_path.exists():
            path = alt_path
        else:
            # Graceful fallback - return empty dict instead of raising
            return {}
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_tp_comm_params(
    config: dict,
    interconnect: str = "local",
    nnodes: int = 1,
) -> tuple[float, float]:
    """Get TP communication bandwidth and latency (ms).

    Args:
        config: Loaded configuration dictionary
        interconnect: Interconnect type ("local", "NV", "PIX", "PXB", "PHB", "NODE", "infiniband", "roce")
        nnodes: Number of nodes

    Returns:
        tuple of (bandwidth_gbps, latency_ms)
    """
    tp_comm = config.get("tp", {}).get("communication", {})

    # Multi-node: use network interconnect params from config
    if nnodes > 1:
        if interconnect == "infiniband":
            return (
                tp_comm.get("infiniband_bandwidth_gbps", 25.0),
                tp_comm.get("infiniband_latency_ms", 0.08),
            )
        elif interconnect == "roce":
            return (
                tp_comm.get("roce_bandwidth_gbps", 18.0),
                tp_comm.get("roce_latency_ms", 0.12),
            )
        else:
            return (
                tp_comm.get("ethernet_bandwidth_gbps", 12.5),
                tp_comm.get("ethernet_latency_ms", 0.25),
            )

    # Single-node: use local topology
    if interconnect.startswith("NV"):
        # NVLink
        return (
            tp_comm.get("nvlink_bandwidth_gbps", 450.0),
            tp_comm.get("nvlink_latency_ms", 0.3),
        )
    elif interconnect in {"PIX", "PXB", "PHB", "NODE"}:
        # PCIe
        return (
            tp_comm.get("pcie_bandwidth_gbps", 32.0),
            tp_comm.get("pcie_latency_ms", 5.0),
        )
    else:
        # Default to PCIe
        return (
            tp_comm.get("pcie_bandwidth_gbps", 32.0),
            tp_comm.get("pcie_latency_ms", 5.0),
        )


def get_ddp_comm_params(config: dict) -> float:
    """Get DDP gradient allreduce bandwidth (GB/s).

    Args:
        config: Loaded configuration dictionary

    Returns:
        Gradient allreduce bandwidth in GB/s
    """
    return config.get("single_ddp", {}).get("communication", {}).get(
        "gradient_allreduce_tflops", 100.0
    )


def get_hardware_params(config: dict) -> dict[str, Any]:
    """Get hardware parameters from config.

    Args:
        config: Loaded configuration dictionary

    Returns:
        Dictionary of hardware parameters
    """
    hw = config.get("hardware", {})
    return {
        "gemm_tflops": hw.get("gemm_tflops"),
        "attention_tflops": hw.get("attention_tflops"),
        "memory_bandwidth_gbps": hw.get("memory_bandwidth_gbps"),
        "launch_overhead_ms": hw.get("launch_overhead_ms"),
        "device_name": hw.get("device_name"),
    }


def get_forward_params(config: dict) -> dict[str, float]:
    """Get forward pass parameters from config.

    Args:
        config: Loaded configuration dictionary

    Returns:
        Dictionary of forward parameters
    """
    common = config.get("common", {}).get("forward", {})
    return {
        "effective_tflops_scale": common.get("effective_tflops_scale", 0.9),
        "kernel_overhead_factor": common.get("kernel_overhead_factor", 0.15),
    }


def get_backward_params(config: dict) -> dict[str, float]:
    """Get backward pass parameters from config.

    Args:
        config: Loaded configuration dictionary

    Returns:
        Dictionary of backward parameters
    """
    common = config.get("common", {}).get("backward", {})
    return {
        "compute_scale": common.get("compute_scale", 3.0666),
        "parallelism_factor": common.get("parallelism_factor", 0.25),
        "overhead_scale": common.get("overhead_scale", 0.3),
        "effective_tflops_min": common.get("effective_tflops_min", 14.5),
        "effective_tflops_scale": common.get("effective_tflops_scale", 0.9),
    }


def get_optimizer_params(config: dict) -> dict[str, float]:
    """Get optimizer parameters from config.

    Args:
        config: Loaded configuration dictionary

    Returns:
        Dictionary of optimizer parameters
    """
    common = config.get("common", {}).get("optimizer", {})
    return {
        "scale_factor": common.get("scale_factor", 1.4),
    }


def get_tp_params(config: dict) -> dict[str, Any]:
    """Get TP-specific parameters from config.

    Args:
        config: Loaded configuration dictionary

    Returns:
        Dictionary of TP parameters
    """
    tp = config.get("tp", {})
    return {
        "backward": tp.get("backward", {}),
        "tp_backward_efficiency": tp.get("backward", {}).get("tp_backward_efficiency", 0.012),
        "tp_forward_efficiency": tp.get("backward", {}).get("tp_forward_efficiency", 0.05),
        "communication": tp.get("communication", {}),
        "overlap_ratio": tp.get("communication", {}).get("overlap_ratio", 0.9),
        "optimizer_efficiency": tp.get("communication", {}).get("optimizer_efficiency", 0.45),
    }


def get_ddp_params(config: dict) -> dict[str, Any]:
    """Get DDP-specific parameters from config.

    Args:
        config: Loaded configuration dictionary

    Returns:
        Dictionary of DDP parameters
    """
    ddp = config.get("single_ddp", {})
    return {
        "backward": ddp.get("backward", {}),
        "effective_tflops_scale": ddp.get("backward", {}).get("effective_tflops_scale", 0.07),
        "communication": ddp.get("communication", {}),
        "gradient_allreduce_tflops": ddp.get("communication", {}).get(
            "gradient_allreduce_tflops", 100.0
        ),
        "parallelism": ddp.get("parallelism", {}),
    }
