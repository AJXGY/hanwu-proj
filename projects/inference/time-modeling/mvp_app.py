from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from mvp_backend import detect_accelerator_kind, empty_cache
from mvp_calibration import benchmark_collective_link, build_calibration
from mvp_model import stable_model_identifier
from mvp_estimator import (
    aggregate_module_profiles,
    build_estimate_compare_rows,
    build_predicted_comm,
    estimate_node,
    finalize_estimate_ordinals,
    sanitize_module_profiles,
    summarize_phase_with_module_substitution,
)
from mvp_execution import dtype_from_name, parse_args, resolve_execution_config
from mvp_graph import tp_shard_node_estimate
from mvp_measurement import (
    build_execution_report,
    build_operator_compare_rows,
    compare_summary,
    distributed_cuda_wall_time_ms,
    gather_rank_objects,
    is_primary_rank,
    merge_comm_summaries,
    relative_error_pct,
    round_nested,
    write_dashboard_status,
    write_reports,
)
from mvp_parallel import apply_tensor_parallel
from mvp_profile import build_profile_report, collect_module_profiles, profile_cuda_ops
from mvp_profile import collect_phase_adjustments
from mvp_table import (
    append_phase_adjustments_to_table,
    append_module_profiles_to_table,
    build_table_context,
    load_module_profiles_from_table,
    load_phase_adjustments_from_table,
    merge_module_profiles,
    missing_profile_scopes,
    phase_scope_mode_map,
    phase_timing_mode_map,
)
from mvp_runtime import (
    build_graph_cache_identity,
    clone_past_key_values,
    collect_decode_loop_step_times,
    export_inference_graphs,
    graph_cache_paths,
    load_cached_inference_graphs,
    prepare_inputs,
    prepare_runtime_inputs,
    run_short_request,
    save_inference_graphs_to_cache,
)

ROOT = Path(__file__).resolve().parent


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    args = parse_args()
    accelerator_kind = detect_accelerator_kind(args.device)

    analysis_started = time.perf_counter()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    execution, device = resolve_execution_config(args)
    if device.type != accelerator_kind:
        raise RuntimeError(
            f"Resolved device type {device.type} does not match accelerator {accelerator_kind}"
        )
    dtype = dtype_from_name(args.dtype)
    enable_operator_profiles = execution.nnodes == 1 and not args.estimate_only

    module_profile_mode = args.estimate_mode
    if args.estimate_only and module_profile_mode == "online":
        module_profile_mode = "table"

    estimate_stage_timings = {
        "calibration_wall_time_s": 0.0,
        "model_load_wall_time_s": 0.0,
        "graph_extract_wall_time_s": 0.0,
        "runtime_inputs_wall_time_s": 0.0,
        "torch_export_wall_time_s": 0.0,
        "graph_cache_load_wall_time_s": 0.0,
        "graph_cache_write_wall_time_s": 0.0,
        "table_lookup_wall_time_s": 0.0,
        "runtime_prepare_wall_time_s": 0.0,
        "module_profile_wall_time_s": 0.0,
        "analytical_estimate_wall_time_s": 0.0,
    }

    try:
        stage_started = time.perf_counter()
        calibration = build_calibration(dtype, device)
        estimate_stage_timings["calibration_wall_time_s"] = (
            time.perf_counter() - stage_started
        )
        if (
            execution.parallel_mode == "tp"
            and execution.collective_bandwidth_gbps is None
            and execution.collective_latency_ms is None
        ):
            collective_calibration = benchmark_collective_link(
                execution, dtype, device
            )
            if collective_calibration is not None:
                execution.collective_bandwidth_gbps = float(
                    collective_calibration["bandwidth_gbps"]
                )
                execution.collective_latency_ms = float(
                    collective_calibration["latency_ms"]
                )
        execution_report = build_execution_report(execution)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        input_ids, attention_mask = prepare_inputs(tokenizer, args.prompt, device)
        model_id = stable_model_identifier(model_path=args.model_path)
        report_created_at = _iso_now()
        graph_cache_identity = build_graph_cache_identity(
            model_id=model_id,
            dtype=args.dtype,
            batch_size=int(input_ids.shape[0]),
            prompt_tokens=int(input_ids.shape[1]),
            execution=execution_report,
        )
        graph_cache_dir = ROOT / ".graph_cache"
        graph_cache_record = {
            "status": "miss",
            "cache_key": graph_cache_paths(graph_cache_dir, graph_cache_identity)[
                "root"
            ].name,
            "cache_path": str(
                graph_cache_paths(graph_cache_dir, graph_cache_identity)["root"]
            ),
            "error": None,
        }

        stage_started = time.perf_counter()
        cached_graphs = load_cached_inference_graphs(
            graph_cache_dir, graph_cache_identity
        )
        estimate_stage_timings["graph_cache_load_wall_time_s"] = (
            time.perf_counter() - stage_started
        )

        model = None
        graph_runtime_inputs = None
        if cached_graphs is not None:
            graphs = None
            graph_counts = dict(cached_graphs["graph_counts"])
            graph_cache_record["status"] = "hit"
        else:
            stage_started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=dtype
            )
            model.eval().to(device)
            estimate_stage_timings["model_load_wall_time_s"] = (
                time.perf_counter() - stage_started
            )

            stage_started = time.perf_counter()
            graph_runtime_inputs = prepare_runtime_inputs(
                model, input_ids, attention_mask
            )
            estimate_stage_timings["runtime_inputs_wall_time_s"] = (
                time.perf_counter() - stage_started
            )
            stage_started = time.perf_counter()
            graphs = export_inference_graphs(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                runtime_inputs=graph_runtime_inputs,
            )
            estimate_stage_timings["torch_export_wall_time_s"] = (
                time.perf_counter() - stage_started
            )
            graph_counts = {
                "prefill_call_function_nodes": sum(
                    1
                    for node in graphs["prefill_export"].graph.nodes
                    if node.op == "call_function"
                ),
                "decode_call_function_nodes": sum(
                    1
                    for node in graphs["decode_export"].graph.nodes
                    if node.op == "call_function"
                ),
            }
        estimate_stage_timings["graph_extract_wall_time_s"] = (
            estimate_stage_timings["runtime_inputs_wall_time_s"]
            + estimate_stage_timings["torch_export_wall_time_s"]
        )

        stage_started = time.perf_counter()
        if cached_graphs is not None:
            prefill_estimates = list(cached_graphs["prefill_estimates"])
            decode_estimates = list(cached_graphs["decode_estimates"])
        else:
            prefill_estimates = finalize_estimate_ordinals(
                [
                    estimate
                    for node in graphs["prefill_export"].graph.nodes
                    if (estimate := estimate_node(node, "prefill", calibration))
                    is not None
                ]
            )
            decode_estimates = finalize_estimate_ordinals(
                [
                    estimate
                    for node in graphs["decode_export"].graph.nodes
                    if (estimate := estimate_node(node, "decode_step", calibration))
                    is not None
                ]
            )
            cache_started = time.perf_counter()
            try:
                cache_result = save_inference_graphs_to_cache(
                    graph_cache_dir,
                    graph_cache_identity,
                    graph_counts,
                    prefill_estimates,
                    decode_estimates,
                )
                graph_cache_record.update(
                    {
                        "status": "miss_written",
                        "cache_key": cache_result["cache_key"],
                        "cache_path": cache_result["cache_path"],
                    }
                )
            except Exception as exc:
                graph_cache_record.update(
                    {
                        "status": "miss_write_failed",
                        "error": str(exc),
                    }
                )
            estimate_stage_timings["graph_cache_write_wall_time_s"] = (
                time.perf_counter() - cache_started
            )
        single_runtime_inputs = (
            graph_runtime_inputs
            if execution.parallel_mode == "single" and not args.estimate_only
            else None
        )
        if execution.parallel_mode == "tp" and not args.estimate_only:
            if graphs is not None:
                del graphs
            gc.collect()
            empty_cache(device.type)

        timing_modes = phase_timing_mode_map(execution.tp_size, execution.nnodes)
        scope_modes = phase_scope_mode_map(execution.tp_size, execution.nnodes)
        summary_prefill_estimates = [
            tp_shard_node_estimate(estimate, execution)
            for estimate in prefill_estimates
        ]
        summary_decode_estimates = [
            tp_shard_node_estimate(estimate, execution) for estimate in decode_estimates
        ]
        table_context = build_table_context(
            model_id=model_id,
            dtype=args.dtype,
            prompt_tokens=int(input_ids.shape[1]),
            execution=execution_report,
            calibration=asdict(calibration),
        )
        estimate_stage_timings["analytical_estimate_wall_time_s"] += (
            time.perf_counter() - stage_started
        )
        aggregated_module_profiles = {"prefill": [], "decode_step": []}
        phase_adjustments = {"prefill": None, "decode_step": None}
        table_lookup_stats = None
        phase_adjustment_lookup_stats = None
        table_writeback_rows = 0
        if module_profile_mode in {"table", "hybrid"}:
            stage_started = time.perf_counter()
            (
                aggregated_module_profiles,
                table_lookup_stats,
            ) = load_module_profiles_from_table(
                table_db_path=args.table_db_path,
                context=table_context,
                prefill_estimates=prefill_estimates,
                decode_estimates=decode_estimates,
                scope_modes=scope_modes,
            )
            phase_adjustments, phase_adjustment_lookup_stats = (
                load_phase_adjustments_from_table(
                    table_db_path=args.table_db_path,
                    context=table_context,
                )
            )
            estimate_stage_timings["table_lookup_wall_time_s"] = (
                time.perf_counter() - stage_started
            )

        online_collection_enabled = not args.estimate_only and module_profile_mode in {
            "online",
            "hybrid",
        }
        allowed_module_scopes = None
        has_missing_scopes = True
        if module_profile_mode == "hybrid":
            allowed_module_scopes = {
                "prefill": missing_profile_scopes(
                    prefill_estimates,
                    scope_mode=scope_modes["prefill"],
                    existing_records=aggregated_module_profiles["prefill"],
                ),
                "decode_step": missing_profile_scopes(
                    decode_estimates,
                    scope_mode=scope_modes["decode_step"],
                    existing_records=aggregated_module_profiles["decode_step"],
                ),
            }
            has_missing_scopes = any(
                len(allowed_module_scopes[phase]) > 0
                for phase in ("prefill", "decode_step")
            )

        if module_profile_mode == "hybrid" and not has_missing_scopes:
            online_collection_enabled = False

        module_profile_meta = {
            "report_created_at": report_created_at,
            "estimate_mode_requested": args.estimate_mode,
            "estimate_mode_effective": module_profile_mode,
            "table_db_path": str(Path(args.table_db_path).expanduser().resolve()),
            "table_lookup": table_lookup_stats,
            "phase_adjustment_lookup": phase_adjustment_lookup_stats,
            "table_writeback": {
                "enabled": bool(args.table_writeback),
                "rows_appended": 0,
            },
            "online_collection_enabled": online_collection_enabled,
        }

        runtime_model = model
        if not args.estimate_only:
            if execution.parallel_mode == "tp":
                stage_started = time.perf_counter()
                if model is not None:
                    del model
                    gc.collect()
                    empty_cache(device.type)
                runtime_model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=dtype,
                )
                runtime_model.eval().to(device)
                apply_tensor_parallel(runtime_model, execution)
                if dist.is_initialized():
                    dist.barrier()
                runtime_inputs = prepare_runtime_inputs(
                    runtime_model, input_ids, attention_mask
                )
                estimate_stage_timings["runtime_prepare_wall_time_s"] = (
                    time.perf_counter() - stage_started
                )
            else:
                if runtime_model is None:
                    stage_started = time.perf_counter()
                    runtime_model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        torch_dtype=dtype,
                    )
                    runtime_model.eval().to(device)
                    single_runtime_inputs = prepare_runtime_inputs(
                        runtime_model, input_ids, attention_mask
                    )
                    estimate_stage_timings["runtime_prepare_wall_time_s"] = (
                        time.perf_counter() - stage_started
                    )
                runtime_inputs = single_runtime_inputs

            next_token = runtime_inputs.next_token
            next_attention_mask = runtime_inputs.next_attention_mask
            decode_past = runtime_inputs.decode_past

            if online_collection_enabled:
                stage_started = time.perf_counter()
                module_profiles = collect_module_profiles(
                    model=runtime_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    next_token=next_token,
                    next_attention_mask=next_attention_mask,
                    decode_past=decode_past,
                    prefill_estimates=prefill_estimates,
                    decode_estimates=decode_estimates,
                    warmup=args.warmup,
                    repeat=args.profile_repeat,
                    timing_mode=timing_modes,
                    scope_mode=scope_modes,
                    allowed_module_scopes=allowed_module_scopes,
                )

                gathered_module_profiles = gather_rank_objects(
                    {
                        "prefill": [
                            asdict(record) for record in module_profiles["prefill"]
                        ],
                        "decode_step": [
                            asdict(record) for record in module_profiles["decode_step"]
                        ],
                    },
                    execution,
                )
                collected_module_profiles = aggregate_module_profiles(
                    gathered_module_profiles
                )
                collected_module_profiles["prefill"] = sanitize_module_profiles(
                    prefill_estimates, collected_module_profiles["prefill"]
                )
                collected_module_profiles["decode_step"] = sanitize_module_profiles(
                    decode_estimates, collected_module_profiles["decode_step"]
                )

                if module_profile_mode == "hybrid":
                    aggregated_module_profiles = merge_module_profiles(
                        base=aggregated_module_profiles,
                        extra=collected_module_profiles,
                    )
                else:
                    aggregated_module_profiles = collected_module_profiles

                if execution.parallel_mode == "single":
                    prefill_base_summary = summarize_phase_with_module_substitution(
                        "prefill",
                        summary_prefill_estimates,
                        aggregated_module_profiles["prefill"],
                    )
                    decode_base_summary = summarize_phase_with_module_substitution(
                        "decode_step",
                        summary_decode_estimates,
                        aggregated_module_profiles["decode_step"],
                    )
                    phase_adjustments = collect_phase_adjustments(
                        model=runtime_model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        next_token=next_token,
                        next_attention_mask=next_attention_mask,
                        decode_past=decode_past,
                        prefill_estimate_ms=prefill_base_summary.end_to_end_time_ms,
                        decode_estimate_ms=decode_base_summary.end_to_end_time_ms,
                        warmup=args.warmup,
                        repeat=args.profile_repeat,
                    )

                if args.table_writeback and is_primary_rank(execution):
                    table_writeback_rows = append_module_profiles_to_table(
                        table_db_path=args.table_db_path,
                        context=table_context,
                        module_profiles=collected_module_profiles,
                        source="online_collection",
                    )
                    if execution.parallel_mode == "single":
                        table_writeback_rows += append_phase_adjustments_to_table(
                            table_db_path=args.table_db_path,
                            context=table_context,
                            phase_adjustments=phase_adjustments,
                            source="online_collection",
                        )
                estimate_stage_timings["module_profile_wall_time_s"] = (
                    time.perf_counter() - stage_started
                )

        module_profile_meta["table_writeback"]["rows_appended"] = table_writeback_rows
        stage_started = time.perf_counter()
        predicted_prefill_comm = build_predicted_comm(
            prefill_estimates,
            execution,
            aggregated_module_profiles["prefill"],
        )
        predicted_decode_comm = build_predicted_comm(
            decode_estimates,
            execution,
            aggregated_module_profiles["decode_step"],
        )

        prefill_summary = summarize_phase_with_module_substitution(
            "prefill",
            summary_prefill_estimates,
            aggregated_module_profiles["prefill"],
            graph_comm_time_ms=predicted_prefill_comm["predicted_total_ms"],
            phase_adjustment_time_ms=(
                phase_adjustments["prefill"].mean_ms
                if phase_adjustments["prefill"] is not None
                else 0.0
            ),
        )
        decode_summary = summarize_phase_with_module_substitution(
            "decode_step",
            summary_decode_estimates,
            aggregated_module_profiles["decode_step"],
            graph_comm_time_ms=predicted_decode_comm["predicted_total_ms"],
            phase_adjustment_time_ms=(
                phase_adjustments["decode_step"].mean_ms
                if phase_adjustments["decode_step"] is not None
                else 0.0
            ),
        )
        request_decode_steps_profile_ms: list[float] = []
        request_decode_estimation_policy = "prefill_plus_constant_decode_step"
        request_estimate_ms = (
            prefill_summary.end_to_end_time_ms
            + max(args.max_new_tokens - 1, 0) * decode_summary.end_to_end_time_ms
        )
        estimate_stage_timings["analytical_estimate_wall_time_s"] += (
            time.perf_counter() - stage_started
        )
        estimation_wall_time_s = time.perf_counter() - analysis_started

        estimate_snapshot = {
            "runtime_model": "torch_eager_v1",
            "mode": "inference",
            "model": {
                "id": model_id,
                "path": args.model_path,
                "prompt": args.prompt,
                "prompt_tokens": int(input_ids.shape[1]),
                "max_new_tokens": args.max_new_tokens,
                "dtype": args.dtype,
            },
            "execution": execution_report,
            "calibration": asdict(calibration),
            "analysis_timing": dict(estimate_stage_timings),
            "graph_cache": dict(graph_cache_record),
            "graph": graph_counts,
            "estimate": {
                "prefill": asdict(prefill_summary),
                "decode_step": asdict(decode_summary),
                "request_end_to_end_time_ms": request_estimate_ms,
                "request_decode_estimation_policy": request_decode_estimation_policy,
                "request_decode_profile_steps_ms": request_decode_steps_profile_ms,
            },
            "operator_compare": {
                "prefill": [],
                "decode_step": [],
                "summary": {
                    "matched_rows": 0,
                    "coverage_estimate_ms_pct": 0.0,
                    "status": ("unavailable" if execution.nnodes > 1 else "available"),
                    "reason": (
                        "distributed tp runtime does not emit stable operator-level matching yet"
                        if execution.nnodes > 1
                        else None
                    ),
                },
            },
            "rank_measurements": {"prefill": [], "decode_step": [], "request": []},
            "comm": {
                "prefill": {
                    "collectives": [],
                    "total_measured_ms": 0.0,
                    "predicted_collectives": predicted_prefill_comm[
                        "predicted_collectives"
                    ],
                    "predicted_total_ms": predicted_prefill_comm["predicted_total_ms"],
                },
                "decode_step": {
                    "collectives": [],
                    "total_measured_ms": 0.0,
                    "predicted_collectives": predicted_decode_comm[
                        "predicted_collectives"
                    ],
                    "predicted_total_ms": predicted_decode_comm["predicted_total_ms"],
                },
            },
            "phase_adjustment": {
                phase: (
                    asdict(phase_adjustments[phase])
                    if phase_adjustments[phase] is not None
                    else None
                )
                for phase in ("prefill", "decode_step")
            },
            "module_profile_meta": module_profile_meta,
        }
        if is_primary_rank(execution):
            write_dashboard_status(
                output_dir,
                {
                    "stage": "estimation_ready",
                    "report": estimate_snapshot,
                    "timings": {
                        "estimation_wall_time_s": estimation_wall_time_s,
                        "measurement_wall_time_s": None,
                        "predictor_total_wall_time_s": None,
                        **estimate_stage_timings,
                    },
                },
            )

        if args.estimate_only:
            zero_stats = {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "samples_ms": [],
            }
            estimate_only_report = {
                **estimate_snapshot,
                "module_profile": {
                    "prefill": [
                        asdict(record)
                        for record in aggregated_module_profiles["prefill"]
                    ],
                    "decode_step": [
                        asdict(record)
                        for record in aggregated_module_profiles["decode_step"]
                    ],
                },
                "phase_adjustment": {
                    phase: (
                        asdict(phase_adjustments[phase])
                        if phase_adjustments[phase] is not None
                        else None
                    )
                    for phase in ("prefill", "decode_step")
                },
                "measured": {
                    "prefill": dict(zero_stats),
                    "decode_step": dict(zero_stats),
                    "request": dict(zero_stats),
                },
                "profile": {
                    "status": "skipped",
                    "reason": "estimate-only run skips profiler collection",
                    "prefill_top_cuda_ops": [],
                    "decode_top_cuda_ops": [],
                    "prefill_per_rank": [],
                    "decode_per_rank": [],
                    "prefill_raw_events": [],
                    "decode_raw_events": [],
                },
                "comparison": {
                    "prefill_relative_error_pct": 0.0,
                    "decode_step_relative_error_pct": 0.0,
                    "request_relative_error_pct": 0.0,
                },
            }
            if not is_primary_rank(execution):
                return
            write_reports(output_dir, estimate_only_report)
            write_dashboard_status(
                output_dir,
                {
                    "stage": "measurement_ready",
                    "report": estimate_only_report,
                    "timings": {
                        "estimation_wall_time_s": estimation_wall_time_s,
                        "measurement_wall_time_s": 0.0,
                        "predictor_total_wall_time_s": estimation_wall_time_s,
                        **estimate_stage_timings,
                    },
                },
            )
            print(json.dumps(round_nested(estimate_only_report), indent=2))
            return

        def prefill_fn() -> None:
            with torch.no_grad():
                runtime_model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=True
                )

        def prepare_decode_inputs() -> dict[str, Any]:
            return {
                "next_token": next_token.clone(),
                "next_attention_mask": next_attention_mask.clone(),
                "decode_past": clone_past_key_values(decode_past),
            }

        def decode_fn() -> None:
            with torch.no_grad():
                runtime_model(
                    input_ids=next_token,
                    attention_mask=next_attention_mask,
                    past_key_values=decode_past,
                    use_cache=True,
                )

        def decode_benchmark_fn(state: dict[str, Any]) -> None:
            with torch.no_grad():
                runtime_model(
                    input_ids=state["next_token"],
                    attention_mask=state["next_attention_mask"],
                    past_key_values=state["decode_past"],
                    use_cache=True,
                )

        def request_fn() -> None:
            run_short_request(
                runtime_model, input_ids, attention_mask, args.max_new_tokens
            )

        prefill_measured, prefill_rank_measurements = distributed_cuda_wall_time_ms(
            prefill_fn, args.warmup, args.benchmark_repeat, execution
        )
        decode_measured, decode_rank_measurements = distributed_cuda_wall_time_ms(
            decode_benchmark_fn,
            args.warmup,
            args.benchmark_repeat,
            execution,
            prepare_fn=prepare_decode_inputs,
        )
        request_measured, request_rank_measurements = distributed_cuda_wall_time_ms(
            request_fn, args.warmup, args.benchmark_repeat, execution
        )
        if execution.nnodes > 1 and args.max_new_tokens > 1:
            profiled_decode_steps = min(max(args.max_new_tokens - 1, 0), 4)
            local_request_decode_steps = collect_decode_loop_step_times(
                runtime_model,
                input_ids,
                attention_mask,
                profiled_decode_steps,
            )
            gathered_request_decode_steps = gather_rank_objects(
                local_request_decode_steps, execution
            )
            request_decode_steps_profile_ms = [
                max(
                    float(rank_steps[index])
                    for rank_steps in gathered_request_decode_steps
                )
                for index in range(profiled_decode_steps)
            ]

        physical_device = execution.local_device
        if enable_operator_profiles:
            prefill_profile_local = profile_cuda_ops(
                runtime_model,
                prefill_fn,
                "prefill",
                args.warmup,
                execution.rank,
                execution.host_name,
                execution.node_rank,
                execution.local_rank,
                physical_device,
            )
            decode_profile_local = profile_cuda_ops(
                runtime_model,
                decode_fn,
                "decode_step",
                args.warmup,
                execution.rank,
                execution.host_name,
                execution.node_rank,
                execution.local_rank,
                physical_device,
            )
            prefill_profiles = gather_rank_objects(prefill_profile_local, execution)
            decode_profiles = gather_rank_objects(decode_profile_local, execution)
        else:
            prefill_profiles = []
            decode_profiles = []

        generated_ids = run_short_request(
            runtime_model, input_ids, attention_mask, args.max_new_tokens
        )
        generated_ids_all = gather_rank_objects(generated_ids, execution)
        root_generated_ids = list(generated_ids_all[0])
        generated_text = tokenizer.decode(root_generated_ids, skip_special_tokens=True)
        generated_tokens_consistent = all(
            list(item) == root_generated_ids for item in generated_ids_all
        )

        if not is_primary_rank(execution):
            return

        prefill_profile = (
            build_profile_report(prefill_profiles)
            if enable_operator_profiles
            else {"top_cuda_ops": [], "raw_events": [], "per_rank": []}
        )
        decode_profile = (
            build_profile_report(decode_profiles)
            if enable_operator_profiles
            else {"top_cuda_ops": [], "raw_events": [], "per_rank": []}
        )
        prefill_estimate_rows = build_estimate_compare_rows(
            prefill_estimates, execution
        )
        decode_estimate_rows = build_estimate_compare_rows(decode_estimates, execution)
        prefill_operator_compare = (
            build_operator_compare_rows(
                "prefill",
                prefill_estimate_rows,
                [item["measured_ops"] for item in prefill_profiles],
            )
            if enable_operator_profiles
            else []
        )
        decode_operator_compare = (
            build_operator_compare_rows(
                "decode_step",
                decode_estimate_rows,
                [item["measured_ops"] for item in decode_profiles],
            )
            if enable_operator_profiles
            else []
        )
        operator_compare_summary = compare_summary(
            prefill_operator_compare + decode_operator_compare
        )

        report = {
            "runtime_model": "torch_eager_v1",
            "mode": "inference",
            "model": {
                "id": model_id,
                "path": args.model_path,
                "prompt": args.prompt,
                "prompt_tokens": int(input_ids.shape[1]),
                "max_new_tokens": args.max_new_tokens,
                "dtype": args.dtype,
                "generated_token_ids": root_generated_ids,
                "generated_text": generated_text,
                "generated_tokens_consistent_across_ranks": generated_tokens_consistent,
            },
            "execution": execution_report,
            "calibration": asdict(calibration),
            "analysis_timing": dict(estimate_stage_timings),
            "graph_cache": dict(graph_cache_record),
            "graph": graph_counts,
            "estimate": {
                "prefill": asdict(prefill_summary),
                "decode_step": asdict(decode_summary),
                "request_end_to_end_time_ms": request_estimate_ms,
                "request_decode_estimation_policy": request_decode_estimation_policy,
                "request_decode_profile_steps_ms": request_decode_steps_profile_ms,
            },
            "module_profile": {
                "prefill": [
                    asdict(record) for record in aggregated_module_profiles["prefill"]
                ],
                "decode_step": [
                    asdict(record)
                    for record in aggregated_module_profiles["decode_step"]
                ],
            },
            "phase_adjustment": {
                phase: (
                    asdict(phase_adjustments[phase])
                    if phase_adjustments[phase] is not None
                    else None
                )
                for phase in ("prefill", "decode_step")
            },
            "measured": {
                "prefill": prefill_measured,
                "decode_step": decode_measured,
                "request": request_measured,
            },
            "profile": {
                "status": "available" if enable_operator_profiles else "unavailable",
                "reason": (
                    None
                    if enable_operator_profiles
                    else "cross-host tp runtime does not emit stable operator profiler tables yet; module profiles remain enabled"
                ),
                "prefill_top_cuda_ops": prefill_profile["top_cuda_ops"],
                "decode_top_cuda_ops": decode_profile["top_cuda_ops"],
                "prefill_per_rank": prefill_profile["per_rank"],
                "decode_per_rank": decode_profile["per_rank"],
                "prefill_raw_events": prefill_profile["raw_events"],
                "decode_raw_events": decode_profile["raw_events"],
            },
            "comparison": {
                "prefill_relative_error_pct": relative_error_pct(
                    prefill_summary.end_to_end_time_ms, prefill_measured["mean_ms"]
                ),
                "decode_step_relative_error_pct": relative_error_pct(
                    decode_summary.end_to_end_time_ms, decode_measured["mean_ms"]
                ),
                "request_relative_error_pct": relative_error_pct(
                    request_estimate_ms, request_measured["mean_ms"]
                ),
            },
            "operator_compare": {
                "prefill": prefill_operator_compare,
                "decode_step": decode_operator_compare,
                "summary": {
                    **operator_compare_summary,
                    "status": (
                        "available" if enable_operator_profiles else "unavailable"
                    ),
                    "reason": (
                        None
                        if enable_operator_profiles
                        else "distributed tp runtime does not emit stable operator-level matching yet"
                    ),
                },
            },
            "rank_measurements": {
                "prefill": prefill_rank_measurements,
                "decode_step": decode_rank_measurements,
                "request": request_rank_measurements,
            },
            "comm": {
                "prefill": {
                    **merge_comm_summaries([item["comm"] for item in prefill_profiles]),
                    "predicted_collectives": predicted_prefill_comm[
                        "predicted_collectives"
                    ],
                    "predicted_total_ms": predicted_prefill_comm["predicted_total_ms"],
                },
                "decode_step": {
                    **merge_comm_summaries([item["comm"] for item in decode_profiles]),
                    "predicted_collectives": predicted_decode_comm[
                        "predicted_collectives"
                    ],
                    "predicted_total_ms": predicted_decode_comm["predicted_total_ms"],
                },
            },
            "module_profile_meta": module_profile_meta,
        }

        predictor_total_wall_time_s = time.perf_counter() - analysis_started
        write_reports(output_dir, report)
        write_dashboard_status(
            output_dir,
            {
                "stage": "measurement_ready",
                "report": report,
                "timings": {
                    "estimation_wall_time_s": estimation_wall_time_s,
                    "measurement_wall_time_s": predictor_total_wall_time_s
                    - estimation_wall_time_s,
                    "predictor_total_wall_time_s": predictor_total_wall_time_s,
                    **estimate_stage_timings,
                },
            },
        )
        print(json.dumps(round_nested(report), indent=2))
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
