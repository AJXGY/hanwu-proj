from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover
    DynamicCache = None

from mvp_graph import flatten_past_key_values
from mvp_types import RuntimeInputs


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


def prepare_inputs_from_shape(
    batch_size: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create input tensors with specified batch size and sequence length.

    Uses repeated tokens to fill the sequence, avoiding tokenizer overhead.
    """
    # Use repeated token IDs (1 = "[START]" token in many vocabularies)
    # This creates a valid sequence of the desired length
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device) + 1
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    return input_ids, attention_mask


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


def extract_inference_graphs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    runtime_inputs = prepare_runtime_inputs(model, input_ids, attention_mask)
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
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(
                input_ids=current_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            torch.cuda.synchronize()
            step_times_ms.append((time.perf_counter() - start) * 1.0e3)
            current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
    return step_times_ms
