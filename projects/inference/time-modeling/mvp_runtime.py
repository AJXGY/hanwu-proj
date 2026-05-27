from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from mvp_backend import synchronize

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover
    DynamicCache = None

from mvp_graph import flatten_past_key_values
from mvp_types import NodeEstimate, RuntimeInputs

GRAPH_CACHE_SCHEMA_VERSION = "mvp_graph_cache_v1"


class PrefillWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits


class DecodeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, use_dynamic_cache: bool = False) -> None:
        super().__init__()
        self.model = model
        self.use_dynamic_cache = use_dynamic_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        past_key_values = []
        for index in range(0, len(past_flat), 2):
            past_key_values.append((past_flat[index], past_flat[index + 1]))
        cache_value: Any
        if self.use_dynamic_cache and DynamicCache is not None:
            cache_value = DynamicCache(past_key_values)
        else:
            cache_value = tuple(past_key_values)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache_value,
            use_cache=True,
        )
        flat_outputs = [outputs.logits]
        if hasattr(outputs.past_key_values, "layers"):
            pairs = [
                (layer.keys, layer.values)
                for layer in outputs.past_key_values.layers
                if hasattr(layer, "keys") and hasattr(layer, "values")
            ]
        else:
            pairs = list(outputs.past_key_values)
        for key, value in pairs:
            flat_outputs.extend([key, value])
        return tuple(flat_outputs)


def prepare_inputs(
    tokenizer: AutoTokenizer, prompt: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def prepare_runtime_inputs(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> RuntimeInputs:
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
    next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
    next_attention_mask = torch.ones(
        (attention_mask.shape[0], attention_mask.shape[1] + 1),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return RuntimeInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        next_token=next_token,
        next_attention_mask=next_attention_mask,
        decode_past=prefill_outputs.past_key_values,
    )


def export_inference_graphs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    runtime_inputs: RuntimeInputs,
) -> dict[str, Any]:
    decode_args = (
        runtime_inputs.next_token,
        runtime_inputs.next_attention_mask,
        *flatten_past_key_values(runtime_inputs.decode_past),
    )

    prefill_export = torch.export.export(
        PrefillWrapper(model), (input_ids, attention_mask)
    )
    decode_export = torch.export.export(
        DecodeWrapper(
            model,
            use_dynamic_cache=hasattr(runtime_inputs.decode_past, "layers"),
        ),
        decode_args,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prefill_outputs": runtime_inputs,
        "prefill_export": prefill_export,
        "decode_export": decode_export,
        "next_token": runtime_inputs.next_token,
        "next_attention_mask": runtime_inputs.next_attention_mask,
        "decode_args": decode_args,
    }


def build_graph_cache_identity(
    *,
    model_id: str,
    dtype: str,
    batch_size: int,
    prompt_tokens: int,
    execution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": GRAPH_CACHE_SCHEMA_VERSION,
        "runtime_model": "torch_eager_v1",
        "model_id": model_id,
        "dtype": dtype,
        "batch_size": int(batch_size),
        "prompt_tokens": int(prompt_tokens),
        "parallel_mode": execution["parallel_mode"],
        "tp_size": int(execution["tp_size"]),
        "world_size": int(execution["world_size"]),
        "nnodes": int(execution["nnodes"]),
        "interconnect": execution["interconnect"],
    }


def graph_cache_key(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def graph_cache_paths(cache_dir: Path, identity: dict[str, Any]) -> dict[str, Path]:
    root = cache_dir / graph_cache_key(identity)
    return {
        "root": root,
        "metadata": root / "metadata.json",
        "prefill": root / "prefill_graph.pt2",
        "decode": root / "decode_graph.pt2",
    }


def load_cached_inference_graphs(
    cache_dir: Path, identity: dict[str, Any]
) -> dict[str, Any] | None:
    paths = graph_cache_paths(cache_dir, identity)
    if not paths["metadata"].exists():
        return None
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("identity") != identity:
        return None
    return {
        "cache_metadata": metadata,
        "cache_key": graph_cache_key(identity),
        "graph_counts": dict(metadata.get("graph_counts", {})),
        "prefill_estimates": [
            NodeEstimate(**item) for item in metadata.get("prefill_estimates", [])
        ],
        "decode_estimates": [
            NodeEstimate(**item) for item in metadata.get("decode_estimates", [])
        ],
    }


def save_inference_graphs_to_cache(
    cache_dir: Path,
    identity: dict[str, Any],
    graph_counts: dict[str, int],
    prefill_estimates: list[NodeEstimate],
    decode_estimates: list[NodeEstimate],
) -> dict[str, Any]:
    paths = graph_cache_paths(cache_dir, identity)
    paths["root"].mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": GRAPH_CACHE_SCHEMA_VERSION,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "graph_counts": dict(graph_counts),
        "prefill_estimates": [asdict(item) for item in prefill_estimates],
        "decode_estimates": [asdict(item) for item in decode_estimates],
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "cache_metadata": metadata,
        "cache_key": graph_cache_key(identity),
        "cache_path": str(paths["root"]),
    }


def clone_past_key_values(
    past_key_values: Any,
) -> Any:
    if hasattr(past_key_values, "layers"):
        pairs = [
            (layer.keys, layer.values)
            for layer in past_key_values.layers
            if hasattr(layer, "keys") and hasattr(layer, "values")
        ]
        cloned_pairs = [(key.clone(), value.clone()) for key, value in pairs]
        if DynamicCache is not None:
            return DynamicCache(cloned_pairs)
        return tuple(cloned_pairs)
    else:
        pairs = list(past_key_values)
    return tuple((key.clone(), value.clone()) for key, value in pairs)


def extract_inference_graphs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    runtime_inputs = prepare_runtime_inputs(model, input_ids, attention_mask)
    return export_inference_graphs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        runtime_inputs=runtime_inputs,
    )


def run_short_request(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> list[int]:
    generated: list[int] = []
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        past_key_values = outputs.past_key_values
        current_attention_mask = attention_mask
        current_token = next_token
        for _ in range(max(max_new_tokens - 1, 0)):
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )
            outputs = model(
                input_ids=current_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(current_token.item()))
            past_key_values = outputs.past_key_values
    return generated


def collect_decode_loop_step_times(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_steps: int,
) -> list[float]:
    if max_steps <= 0:
        return []
    step_times_ms: list[float] = []
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
        current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        current_attention_mask = attention_mask
        for _ in range(max_steps):
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )
            synchronize(attention_mask.device.type)
            start = time.perf_counter()
            outputs = model(
                input_ids=current_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            synchronize(attention_mask.device.type)
            step_times_ms.append((time.perf_counter() - start) * 1.0e3)
            current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
    return step_times_ms
