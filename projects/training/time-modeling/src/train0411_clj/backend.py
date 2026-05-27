from __future__ import annotations

import os
from typing import Any

import torch

ACCELERATOR_VISIBLE_DEVICE_ENVS = {
    "cuda": ["CUDA_VISIBLE_DEVICES"],
    "mlu": ["MLU_VISIBLE_DEVICES", "MLU_VISIBLE_DEVICE", "CAMBRICON_VISIBLE_DEVICES"],
    "cpu": [],
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
    if text.startswith("cpu"):
        return "cpu"
    if accelerator_available("mlu"):
        return "mlu"
    if accelerator_available("cuda"):
        return "cuda"
    raise RuntimeError("No supported accelerator detected; expected CUDA or MLU")


def default_device_string(kind: str | None = None) -> str:
    accelerator = kind or detect_accelerator_kind()
    if accelerator == "cpu":
        return "cpu"
    return f"{accelerator}:0"


def normalize_device_string(device: str | None, kind: str | None = None) -> str:
    accelerator = kind or detect_accelerator_kind(device)
    text = str(device or "").strip()
    if not text:
        return default_device_string(accelerator)
    if text.startswith(f"{accelerator}:"):
        return text
    if accelerator == "cpu":
        return "cpu"
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


def set_device(kind: str, index: int) -> None:
    if kind == "cpu":
        return
    accelerator_module(kind).set_device(index)


def synchronize(kind: str) -> None:
    if kind == "cpu":
        return
    accelerator_module(kind).synchronize()


def empty_cache(kind: str) -> None:
    if kind == "cpu":
        return
    cache_clear = getattr(accelerator_module(kind), "empty_cache", None)
    if callable(cache_clear):
        cache_clear()


def event(kind: str, **kwargs: Any):
    return accelerator_module(kind).Event(**kwargs)
