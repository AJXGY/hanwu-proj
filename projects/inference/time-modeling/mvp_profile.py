from __future__ import annotations

import statistics
import time
from collections import defaultdict
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

from mvp_backend import detect_accelerator_kind, event, profiler_activity, synchronize
from mvp_estimator import covered_estimates_for_scope
from mvp_graph import *
from mvp_measurement import cuda_wall_time_ms
from mvp_runtime import clone_past_key_values
from mvp_types import ModuleProfileRecord, NodeEstimate, PhaseAdjustmentProfile


def collect_module_profiles(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    next_attention_mask: torch.Tensor,
    decode_past,
    prefill_estimates: list[NodeEstimate],
    decode_estimates: list[NodeEstimate],
    warmup: int,
    repeat: int,
    timing_mode: str | dict[str, str] = "cuda_event",
    scope_mode: str | dict[str, str] = "submodule",
    allowed_module_scopes: dict[str, set[str]] | None = None,
) -> dict[str, list[ModuleProfileRecord]]:
    named_modules = dict(model.named_modules())

    def phase_target_modules(phase_scope_mode: str) -> dict[str, torch.nn.Module]:
        targets = {}
        for name, module in named_modules.items():
            scope_name = normalize_module_scope_name(name or "model")
            if "." not in scope_name and scope_name != "model":
                scope_name = f"model.{scope_name}"
            parts = scope_name.split(".")
            is_decoder_layer = (
                len(parts) >= 3 and parts[-2] == "layers" and parts[-1].isdigit()
            )
            if phase_scope_mode in {"layer", "layer_plus_tail"}:
                if is_decoder_layer:
                    targets[scope_name] = module
                elif phase_scope_mode == "layer_plus_tail" and scope_name in {
                    "model.norm",
                    "model.lm_head",
                }:
                    targets[scope_name] = module
                continue
            if scope_name.endswith(("self_attn", "mlp")):
                targets[scope_name] = module
                continue
            if scope_name.endswith(("input_layernorm", "post_attention_layernorm")):
                targets[scope_name] = module
                continue
            if scope_name in {"model.norm", "model.lm_head"}:
                targets[scope_name] = module
        return targets

    def module_kind_from_scope(scope: str) -> str:
        parts = scope.split(".")
        if len(parts) >= 3 and parts[-2] == "layers" and parts[-1].isdigit():
            return "decoder_layer"
        if scope in {"model.norm", "model.lm_head"}:
            return scope.split(".")[-1]
        return parts[-1]

    phase_records: dict[str, list[dict[str, float]]] = {
        "prefill": [],
        "decode_step": [],
    }

    def record_phase(phase: str, fn, prepare_fn=None) -> None:
        phase_timing_mode = (
            timing_mode.get(phase, "cuda_event")
            if isinstance(timing_mode, dict)
            else timing_mode
        )
        phase_scope_mode = (
            scope_mode.get(phase, "submodule")
            if isinstance(scope_mode, dict)
            else scope_mode
        )
        target_modules = phase_target_modules(phase_scope_mode)
        if allowed_module_scopes is not None:
            allowed = allowed_module_scopes.get(phase, set())
            if allowed:
                target_modules = {
                    name: module
                    for name, module in target_modules.items()
                    if name in allowed
                }
            else:
                target_modules = {}
        prepared = prepare_fn is not None
        for _ in range(warmup):
            state = prepare_fn() if prepared else None
            if prepared:
                fn(state)
            else:
                fn()
        synchronize()
        for _ in range(repeat):
            state = prepare_fn() if prepared else None
            module_timings: dict[str, Any] = {}
            hooks = []
            for name, module in target_modules.items():
                if phase_timing_mode == "wall_time":
                    module_timings[name] = {"start": 0.0, "end": 0.0}

                    def pre_hook(_module, _inputs, state=module_timings[name]):
                        synchronize()
                        state["start"] = time.perf_counter()

                    def post_hook(
                        _module, _inputs, _outputs, state=module_timings[name]
                    ):
                        synchronize()
                        state["end"] = time.perf_counter()

                else:
                    accelerator_kind = detect_accelerator_kind()
                    start_event = event(accelerator_kind, enable_timing=True)
                    end_event = event(accelerator_kind, enable_timing=True)
                    module_timings[name] = (start_event, end_event)

                    def pre_hook(_module, _inputs, evt=start_event):
                        evt.record()

                    def post_hook(_module, _inputs, _outputs, evt=end_event):
                        evt.record()

                hooks.append(module.register_forward_pre_hook(pre_hook))
                hooks.append(module.register_forward_hook(post_hook))

            if prepared:
                fn(state)
            else:
                fn()
            synchronize()
            sample: dict[str, float] = {}
            for name, timing in module_timings.items():
                if phase_timing_mode == "wall_time":
                    sample[name] = max(timing["end"] - timing["start"], 0.0) * 1.0e3
                else:
                    start_event, end_event = timing
                    sample[name] = start_event.elapsed_time(end_event)
            phase_records[phase].append(sample)
            for hook in hooks:
                hook.remove()

    def prefill_fn() -> None:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    def prepare_decode_inputs() -> dict[str, Any]:
        return {
            "next_token": next_token.clone(),
            "next_attention_mask": next_attention_mask.clone(),
            "decode_past": clone_past_key_values(decode_past),
        }

    def decode_fn(state: dict[str, Any]) -> None:
        with torch.no_grad():
            model(
                input_ids=state["next_token"],
                attention_mask=state["next_attention_mask"],
                past_key_values=state["decode_past"],
                use_cache=True,
            )

    record_phase("prefill", prefill_fn)
    record_phase("decode_step", decode_fn, prepare_fn=prepare_decode_inputs)

    estimate_lookup = {"prefill": prefill_estimates, "decode_step": decode_estimates}
    module_profiles: dict[str, list[ModuleProfileRecord]] = {
        "prefill": [],
        "decode_step": [],
    }
    for phase, samples in phase_records.items():
        by_scope: dict[str, list[float]] = defaultdict(list)
        for sample in samples:
            for scope, value in sample.items():
                by_scope[scope].append(value)
        phase_estimates = estimate_lookup[phase]
        for scope in sorted(by_scope.keys(), key=module_scope_key):
            covered = covered_estimates_for_scope(scope, phase_estimates)
            if not covered:
                continue
            module_kind = module_kind_from_scope(scope)
            covered_op_families = sorted({estimate.op_family for estimate in covered})
            module_profiles[phase].append(
                ModuleProfileRecord(
                    module_scope=scope,
                    module_kind=module_kind,
                    phase=phase,
                    covered_node_ids=[estimate.node_name for estimate in covered],
                    covered_op_families=covered_op_families,
                    substitution_policy="module_profile_replaces_covered_nodes",
                    mean_ms=statistics.mean(by_scope[scope]),
                    median_ms=statistics.median(by_scope[scope]),
                    min_ms=min(by_scope[scope]),
                    max_ms=max(by_scope[scope]),
                    samples_ms=list(by_scope[scope]),
                )
            )
    return module_profiles


def collect_phase_adjustments(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    next_attention_mask: torch.Tensor,
    decode_past,
    prefill_estimate_ms: float,
    decode_estimate_ms: float,
    warmup: int,
    repeat: int,
) -> dict[str, PhaseAdjustmentProfile]:
    def prefill_fn() -> None:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    def prepare_decode_inputs() -> dict[str, Any]:
        return {
            "next_token": next_token.clone(),
            "next_attention_mask": next_attention_mask.clone(),
            "decode_past": clone_past_key_values(decode_past),
        }

    def decode_fn(state: dict[str, Any]) -> None:
        with torch.no_grad():
            model(
                input_ids=state["next_token"],
                attention_mask=state["next_attention_mask"],
                past_key_values=state["decode_past"],
                use_cache=True,
            )

    phase_wall_time = {
        "prefill": cuda_wall_time_ms(prefill_fn, warmup, repeat),
        "decode_step": cuda_wall_time_ms(
            decode_fn, warmup, repeat, prepare_fn=prepare_decode_inputs
        ),
    }
    base_estimate_ms = {
        "prefill": float(prefill_estimate_ms),
        "decode_step": float(decode_estimate_ms),
    }

    adjustments: dict[str, PhaseAdjustmentProfile] = {}
    for phase in ("prefill", "decode_step"):
        samples = [
            float(sample) - base_estimate_ms[phase]
            for sample in phase_wall_time[phase]["samples_ms"]
        ]
        adjustments[phase] = PhaseAdjustmentProfile(
            phase=phase,
            mean_ms=statistics.mean(samples),
            median_ms=statistics.median(samples),
            min_ms=min(samples),
            max_ms=max(samples),
            samples_ms=samples,
        )
    return adjustments


def profiler_event_self_cuda_time_ms(event: Any) -> float:
    if hasattr(event, "self_device_time_total"):
        return float(getattr(event, "self_device_time_total")) / 1000.0
    if hasattr(event, "self_cuda_time_total"):
        return float(getattr(event, "self_cuda_time_total")) / 1000.0
    if hasattr(event, "cuda_time_total"):
        return float(getattr(event, "cuda_time_total")) / 1000.0
    return 0.0


def profiler_event_total_cuda_time_ms(event: Any) -> float:
    if hasattr(event, "device_time_total"):
        return float(getattr(event, "device_time_total")) / 1000.0
    if hasattr(event, "cuda_time_total"):
        return float(getattr(event, "cuda_time_total")) / 1000.0
    return profiler_event_self_cuda_time_ms(event)


def normalize_profiler_shapes(raw_shapes: Any) -> list[list[int]]:
    shapes: list[list[int]] = []
    if not raw_shapes:
        return shapes
    for item in raw_shapes:
        if isinstance(item, (list, tuple)):
            dims: list[int] = []
            for dim in item:
                try:
                    dims.append(int(dim))
                except (TypeError, ValueError):
                    continue
            if dims:
                shapes.append(dims)
    return shapes


def collective_kind(target: str) -> str | None:
    lowered = target.lower()
    for candidate in ("all_reduce", "all_gather", "reduce_scatter"):
        if candidate in lowered:
            return candidate
    return None


def is_backend_collective_event_target(target: str) -> bool:
    lowered = target.lower()
    return lowered.startswith("cncl:") or lowered.startswith("nccl:")


def is_operator_target(target: str) -> bool:
    return "::" in target and collective_kind(target) is None


class ModuleScopeTracker:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.handles = []
        self.stack: list[str] = []

    def current_scope(self) -> str:
        return normalize_module_scope_name(self.stack[-1] if self.stack else "global")

    def __enter__(self) -> "ModuleScopeTracker":
        for name, module in self.model.named_modules():
            scope_name = normalize_module_scope_name(name or "model")

            def pre_hook(_module, _inputs, scope=scope_name):
                self.stack.append(scope)

            def post_hook(_module, _inputs, _outputs, scope=scope_name):
                if self.stack and self.stack[-1] == scope:
                    self.stack.pop()
                elif scope in self.stack:
                    self.stack.remove(scope)

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        while self.handles:
            self.handles.pop().remove()
        self.stack.clear()


class OpTraceMode(TorchDispatchMode):
    def __init__(self, tracker: ModuleScopeTracker) -> None:
        super().__init__()
        self.tracker = tracker
        self.records: list[dict[str, Any]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        output_shapes = tensor_shapes_from_value(result)
        self.records.append(
            {
                "target": canonical_target_name(func),
                "module_scope": self.tracker.current_scope(),
                "output_shapes": output_shapes,
                "shape_signature": shape_signature(output_shapes),
            }
        )
        return result


def collect_raw_profiler_events(prof) -> list[dict[str, Any]]:
    rows = []
    for index, event in enumerate(prof.events()):
        self_cuda_ms = profiler_event_self_cuda_time_ms(event)
        total_cuda_ms = profiler_event_total_cuda_time_ms(event)
        if self_cuda_ms <= 0 and total_cuda_ms <= 0:
            continue
        raw_shapes = normalize_profiler_shapes(getattr(event, "input_shapes", None))
        rows.append(
            {
                "event_index": index,
                "target": canonical_target_name(getattr(event, "key", "")),
                "name": getattr(event, "name", getattr(event, "key", "")),
                "self_cuda_time_ms": self_cuda_ms,
                "cuda_time_total_ms": total_cuda_ms,
                "input_shapes": raw_shapes,
                "shape_signature": shape_signature(raw_shapes),
                "device_type": str(getattr(event, "device_type", "unknown")),
                "calls": 1,
            }
        )
    return rows


def aggregate_profiler_table(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for event in raw_events:
        if not is_operator_target(event["target"]):
            continue
        row = aggregated.setdefault(
            event["target"],
            {"op": event["target"], "self_cuda_time_ms": 0.0, "calls": 0},
        )
        row["self_cuda_time_ms"] += event["self_cuda_time_ms"]
        row["calls"] += 1
    rows = list(aggregated.values())
    rows.sort(key=lambda item: item["self_cuda_time_ms"], reverse=True)
    return rows[:10]


def build_measured_op_records(
    phase: str,
    rank: int,
    host_name: str,
    node_rank: int,
    local_rank: int,
    physical_device: int,
    dispatch_records: list[dict[str, Any]],
    raw_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = [
        dict(item) for item in dispatch_records if is_operator_target(item["target"])
    ]
    event_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in raw_events:
        if is_operator_target(event["target"]):
            event_buckets[event["target"]].append(event)
    event_offsets: dict[str, int] = defaultdict(int)
    measured: list[dict[str, Any]] = []
    for item in records:
        event_index = event_offsets[item["target"]]
        event_offsets[item["target"]] += 1
        event = None
        if event_index < len(event_buckets[item["target"]]):
            event = event_buckets[item["target"]][event_index]
        measured.append(
            {
                "phase": phase,
                "rank": rank,
                "host": host_name,
                "node_rank": node_rank,
                "local_rank": local_rank,
                "device": physical_device,
                "target": item["target"],
                "op_family": op_family_from_target(item["target"]),
                "module_scope": item["module_scope"],
                "output_shapes": item["output_shapes"],
                "shape_signature": item["shape_signature"],
                "measured_ms": event["self_cuda_time_ms"] if event else 0.0,
                "calls": 1,
                "profiler_event_index": event["event_index"] if event else None,
                "profiler_shape_signature": event["shape_signature"] if event else "[]",
            }
        )
    assign_ordinals_by_group(
        measured, ("phase", "target", "module_scope", "shape_signature")
    )
    return measured


def summarize_rank_comm(
    raw_events: list[dict[str, Any]],
    rank: int,
    host_name: str,
    node_rank: int,
    local_rank: int,
    physical_device: int,
) -> dict[str, Any]:
    collectives: dict[str, dict[str, Any]] = {}
    total_ms = 0.0
    collective_events = [event for event in raw_events if collective_kind(event["target"])]
    backend_collective_events = [
        event
        for event in collective_events
        if is_backend_collective_event_target(event["target"])
    ]
    effective_events = backend_collective_events or collective_events
    for event in effective_events:
        kind = collective_kind(event["target"])
        if kind is None:
            continue
        row = collectives.setdefault(
            kind,
            {
                "collective": kind,
                "count": 0,
                "total_measured_ms": 0.0,
                "rank": rank,
                "host": host_name,
                "node_rank": node_rank,
                "local_rank": local_rank,
                "device": physical_device,
            },
        )
        row["count"] += 1
        row["total_measured_ms"] += event["self_cuda_time_ms"]
        total_ms += event["self_cuda_time_ms"]
    return {
        "collectives": list(collectives.values()),
        "total_measured_ms": total_ms,
        "rank": rank,
        "host": host_name,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "device": physical_device,
    }


def profile_cuda_ops(
    model: torch.nn.Module,
    fn,
    phase: str,
    warmup: int,
    rank: int,
    host_name: str,
    node_rank: int,
    local_rank: int,
    physical_device: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    synchronize()
    tracker = ModuleScopeTracker(model)
    dispatch_mode = OpTraceMode(tracker)
    with (
        tracker,
        dispatch_mode,
        profile(
            activities=[ProfilerActivity.CPU, profiler_activity(detect_accelerator_kind())],
            record_shapes=True,
        ) as prof,
    ):
        fn()
        synchronize()
    raw_events = collect_raw_profiler_events(prof)
    return {
        "phase": phase,
        "rank": rank,
        "host": host_name,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "device": physical_device,
        "top_ops": aggregate_profiler_table(raw_events),
        "raw_events": raw_events,
        "measured_ops": build_measured_op_records(
            phase,
            rank,
            host_name,
            node_rank,
            local_rank,
            physical_device,
            dispatch_mode.records,
            raw_events,
        ),
        "comm": summarize_rank_comm(
            raw_events,
            rank,
            host_name,
            node_rank,
            local_rank,
            physical_device,
        ),
    }


def build_profile_report(phase_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    raw_events: list[dict[str, Any]] = []
    per_rank = []
    for profile_item in phase_profiles:
        per_rank.append(
            {
                "rank": profile_item["rank"],
                "host": profile_item.get("host"),
                "node_rank": profile_item.get("node_rank"),
                "local_rank": profile_item.get("local_rank"),
                "device": profile_item["device"],
                "top_cuda_ops": profile_item["top_ops"],
            }
        )
        for event in profile_item["raw_events"]:
            raw_events.append(
                {
                    **event,
                    "rank": profile_item["rank"],
                    "host": profile_item.get("host"),
                    "node_rank": profile_item.get("node_rank"),
                    "local_rank": profile_item.get("local_rank"),
                    "device": profile_item["device"],
                }
            )
    return {
        "top_cuda_ops": aggregate_profiler_table(raw_events),
        "raw_events": raw_events,
        "per_rank": per_rank,
    }
