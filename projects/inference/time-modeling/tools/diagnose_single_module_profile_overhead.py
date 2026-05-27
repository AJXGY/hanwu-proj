from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mvp_backend import default_device_string
from mvp_calibration import build_calibration
from mvp_estimator import estimate_node, finalize_estimate_ordinals
from mvp_measurement import cuda_wall_time_ms
from mvp_profile import collect_module_profiles
from mvp_runtime import clone_past_key_values, extract_inference_graphs, prepare_inputs
from mvp_table import expected_profile_scopes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default=default_device_string())
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def phase_target_scope_count(module_profiles: dict[str, list]) -> int:
    return len(module_profiles)


def module_sum_ms(module_profiles: list) -> float:
    return sum(float(record.mean_ms) for record in module_profiles)


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [
        items[index : index + chunk_size] for index in range(0, len(items), chunk_size)
    ]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype)
    model.eval().to(device)

    input_ids, attention_mask = prepare_inputs(tokenizer, args.prompt, device)
    graphs = extract_inference_graphs(model, input_ids, attention_mask)
    calibration = build_calibration(dtype, device)
    prefill_estimates = finalize_estimate_ordinals(
        [
            estimate
            for node in graphs["prefill_export"].graph.nodes
            if (estimate := estimate_node(node, "prefill", calibration)) is not None
        ]
    )
    decode_estimates = finalize_estimate_ordinals(
        [
            estimate
            for node in graphs["decode_export"].graph.nodes
            if (estimate := estimate_node(node, "decode_step", calibration)) is not None
        ]
    )

    runtime_inputs = graphs["prefill_outputs"]

    def prefill_fn() -> None:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    def prepare_decode_inputs() -> dict[str, object]:
        return {
            "next_token": runtime_inputs.next_token.clone(),
            "next_attention_mask": runtime_inputs.next_attention_mask.clone(),
            "decode_past": clone_past_key_values(runtime_inputs.decode_past),
        }

    def decode_fn(state) -> None:
        with torch.no_grad():
            model(
                input_ids=state["next_token"],
                attention_mask=state["next_attention_mask"],
                past_key_values=state["decode_past"],
                use_cache=True,
            )

    measured = {
        "prefill": cuda_wall_time_ms(prefill_fn, args.warmup, args.repeat),
        "decode_step": cuda_wall_time_ms(
            decode_fn, args.warmup, args.repeat, prepare_fn=prepare_decode_inputs
        ),
    }

    diagnostics = {
        "config": {
            "model_path": args.model_path,
            "prompt_tokens": int(input_ids.shape[1]),
            "dtype": args.dtype,
            "device": args.device,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "measured_phase_wall_time_ms": measured,
        "runs": [],
        "batched_submodule_cuda_event": [],
    }

    for scope_mode in ("submodule", "layer", "layer_plus_tail"):
        for timing_mode in ("cuda_event", "wall_time"):
            module_profiles = collect_module_profiles(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                next_token=runtime_inputs.next_token,
                next_attention_mask=runtime_inputs.next_attention_mask,
                decode_past=runtime_inputs.decode_past,
                prefill_estimates=prefill_estimates,
                decode_estimates=decode_estimates,
                warmup=args.warmup,
                repeat=args.repeat,
                timing_mode={"prefill": timing_mode, "decode_step": timing_mode},
                scope_mode={"prefill": scope_mode, "decode_step": scope_mode},
            )
            diagnostics["runs"].append(
                {
                    "scope_mode": scope_mode,
                    "timing_mode": timing_mode,
                    "prefill": {
                        "module_scope_count": phase_target_scope_count(
                            module_profiles["prefill"]
                        ),
                        "hook_count": phase_target_scope_count(
                            module_profiles["prefill"]
                        )
                        * 2,
                        "module_sum_mean_ms": module_sum_ms(module_profiles["prefill"]),
                        "module_sum_vs_phase_ratio": module_sum_ms(
                            module_profiles["prefill"]
                        )
                        / float(measured["prefill"]["mean_ms"]),
                    },
                    "decode_step": {
                        "module_scope_count": phase_target_scope_count(
                            module_profiles["decode_step"]
                        ),
                        "hook_count": phase_target_scope_count(
                            module_profiles["decode_step"]
                        )
                        * 2,
                        "module_sum_mean_ms": module_sum_ms(
                            module_profiles["decode_step"]
                        ),
                        "module_sum_vs_phase_ratio": module_sum_ms(
                            module_profiles["decode_step"]
                        )
                        / float(measured["decode_step"]["mean_ms"]),
                    },
                }
            )

    submodule_scopes = {
        "prefill": expected_profile_scopes(prefill_estimates, "submodule"),
        "decode_step": expected_profile_scopes(decode_estimates, "submodule"),
    }
    for chunk_size in (16, 32, 64):
        merged_profiles = {"prefill": [], "decode_step": []}
        chunk_counts = {}
        max_hooks = 0
        for phase in ("prefill", "decode_step"):
            scope_chunks = chunked(submodule_scopes[phase], chunk_size)
            chunk_counts[phase] = len(scope_chunks)
            max_hooks = max(
                max_hooks, min(chunk_size, len(submodule_scopes[phase])) * 2
            )
            for scope_chunk in scope_chunks:
                collected = collect_module_profiles(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    next_token=runtime_inputs.next_token,
                    next_attention_mask=runtime_inputs.next_attention_mask,
                    decode_past=runtime_inputs.decode_past,
                    prefill_estimates=prefill_estimates,
                    decode_estimates=decode_estimates,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    timing_mode={"prefill": "cuda_event", "decode_step": "cuda_event"},
                    scope_mode={"prefill": "submodule", "decode_step": "submodule"},
                    allowed_module_scopes={phase: set(scope_chunk)},
                )
                merged_profiles[phase].extend(collected[phase])
        diagnostics["batched_submodule_cuda_event"].append(
            {
                "chunk_size": chunk_size,
                "prefill_chunk_count": chunk_counts["prefill"],
                "decode_step_chunk_count": chunk_counts["decode_step"],
                "max_hook_count_per_pass": max_hooks,
                "prefill": {
                    "module_scope_count": phase_target_scope_count(
                        merged_profiles["prefill"]
                    ),
                    "module_sum_mean_ms": module_sum_ms(merged_profiles["prefill"]),
                    "module_sum_vs_phase_ratio": module_sum_ms(
                        merged_profiles["prefill"]
                    )
                    / float(measured["prefill"]["mean_ms"]),
                },
                "decode_step": {
                    "module_scope_count": phase_target_scope_count(
                        merged_profiles["decode_step"]
                    ),
                    "module_sum_mean_ms": module_sum_ms(merged_profiles["decode_step"]),
                    "module_sum_vs_phase_ratio": module_sum_ms(
                        merged_profiles["decode_step"]
                    )
                    / float(measured["decode_step"]["mean_ms"]),
                },
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
