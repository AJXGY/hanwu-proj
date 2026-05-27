from __future__ import annotations

import os
import subprocess
from typing import Any

import torch
from torch.profiler import ProfilerActivity

ACCELERATOR_VISIBLE_DEVICE_ENVS = {
    "cuda": ["CUDA_VISIBLE_DEVICES"],
    "mlu": ["MLU_VISIBLE_DEVICES", "MLU_VISIBLE_DEVICE", "CAMBRICON_VISIBLE_DEVICES"],
}

ACCELERATOR_DISTRIBUTED_BACKENDS = {
    "cuda": "nccl",
    "mlu": "cncl",
}


def accelerator_module(kind: str):
    module = getattr(torch, kind, None)
    if module is None:
        raise RuntimeError(f"torch.{kind} is unavailable in this environment")
    return module


def accelerator_available(kind: str) -> bool:
    module = getattr(torch, kind, None)
    if module is None:
        return False
    is_available = getattr(module, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return False


def detect_accelerator_kind(preferred_device: str | None = None) -> str:
    text = str(preferred_device or "").strip().lower()
    if text.startswith("mlu"):
        return "mlu"
    if text.startswith("cuda"):
        return "cuda"
    if accelerator_available("mlu"):
        return "mlu"
    if accelerator_available("cuda"):
        return "cuda"
    raise RuntimeError("No supported accelerator detected; expected CUDA or MLU")


def default_device_string(kind: str | None = None) -> str:
    accelerator = kind or detect_accelerator_kind()
    return f"{accelerator}:0"


def normalize_device_string(device: str | None, kind: str | None = None) -> str:
    accelerator = kind or detect_accelerator_kind(device)
    text = str(device or "").strip()
    if not text:
        return default_device_string(accelerator)
    if text.startswith(f"{accelerator}:"):
        return text
    if text.startswith(("cuda:", "mlu:")):
        _, _, index_text = text.partition(":")
        return f"{accelerator}:{index_text or '0'}"
    if text.isdigit():
        return f"{accelerator}:{text}"
    return text


def uses_visible_device_remap(kind: str) -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in ACCELERATOR_VISIBLE_DEVICE_ENVS.get(kind, [])
    )


def visible_device_env_names(kind: str) -> list[str]:
    return list(ACCELERATOR_VISIBLE_DEVICE_ENVS.get(kind, []))


def synchronize(kind: str | None = None) -> None:
    accelerator = kind or detect_accelerator_kind()
    accelerator_module(accelerator).synchronize()


def empty_cache(kind: str | None = None) -> None:
    accelerator = kind or detect_accelerator_kind()
    cache_clear = getattr(accelerator_module(accelerator), "empty_cache", None)
    if callable(cache_clear):
        cache_clear()


def set_device(kind: str, index: int) -> None:
    accelerator_module(kind).set_device(index)


def event(kind: str, **kwargs: Any):
    return accelerator_module(kind).Event(**kwargs)


def get_device_properties(kind: str, device: torch.device | int):
    return accelerator_module(kind).get_device_properties(device)


def distributed_backend(kind: str) -> str:
    return ACCELERATOR_DISTRIBUTED_BACKENDS[kind]


def profiler_activity(kind: str):
    if kind == "cuda":
        return ProfilerActivity.CUDA
    activity = getattr(ProfilerActivity, kind.upper(), None)
    if activity is None:
        raise RuntimeError(f"Profiler activity is unavailable for accelerator '{kind}'")
    return activity


def local_topology(kind: str, devices: list[int]) -> str:
    if len(devices) < 2:
        return "local"
    if kind != "cuda":
        return "local"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    lines = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return "unknown"
    headers = lines[0].split()
    gpu_headers = [token for token in headers if token.startswith("GPU")]
    row_lookup: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        parts = line.split()
        if not parts or not parts[0].startswith("GPU"):
            continue
        row_lookup[parts[0]] = {
            header: parts[index + 1]
            for index, header in enumerate(gpu_headers)
            if index + 1 < len(parts)
        }
    src = f"GPU{devices[0]}"
    dst = f"GPU{devices[1]}"
    return row_lookup.get(src, {}).get(dst, "unknown")
