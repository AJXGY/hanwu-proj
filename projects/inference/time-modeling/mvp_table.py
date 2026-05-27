from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mvp_estimator import covered_estimates_for_scope
from mvp_graph import module_scope_key, normalize_module_scope_name
from mvp_model import stable_model_identifier
from mvp_types import ModuleProfileRecord, NodeEstimate, PhaseAdjustmentProfile

TABLE_SCHEMA_VERSION = "mvp_module_profile_table_v1"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def phase_timing_mode_map(tp_size: int, nnodes: int) -> dict[str, str]:
    prefill_mode = "wall_time" if tp_size > 1 else "cuda_event"
    decode_mode = "wall_time" if tp_size > 1 or nnodes > 1 else "cuda_event"
    return {"prefill": prefill_mode, "decode_step": decode_mode}


def phase_scope_mode_map(tp_size: int, nnodes: int) -> dict[str, str]:
    if tp_size == 1 and nnodes == 1:
        return {"prefill": "layer_plus_tail", "decode_step": "layer_plus_tail"}
    prefill_mode = "submodule"
    decode_mode = "layer" if nnodes > 1 else "submodule"
    return {"prefill": prefill_mode, "decode_step": decode_mode}


def profile_scope_from_estimate(module_scope: str, phase_scope_mode: str) -> str | None:
    scope = normalize_module_scope_name(module_scope)
    parts = scope.split(".")

    if phase_scope_mode in {"layer", "layer_plus_tail"}:
        if "layers" in parts:
            index = parts.index("layers")
            if index + 1 < len(parts) and parts[index + 1].isdigit():
                return ".".join(parts[: index + 2])
        if phase_scope_mode == "layer_plus_tail" and scope in {
            "model.norm",
            "model.lm_head",
        }:
            return scope
        return None

    for marker in (".self_attn", ".mlp"):
        if marker in scope:
            prefix, _, _ = scope.partition(marker)
            return f"{prefix}{marker}"
    if scope.endswith(("input_layernorm", "post_attention_layernorm")):
        return scope
    if scope in {"model.norm", "model.lm_head"}:
        return scope
    return None


def expected_profile_scopes(
    phase_estimates: list[NodeEstimate], phase_scope_mode: str
) -> list[str]:
    scopes = {
        scope
        for estimate in phase_estimates
        if (
            scope := profile_scope_from_estimate(
                estimate.module_scope, phase_scope_mode
            )
        )
    }
    return sorted(scopes, key=module_scope_key)


def build_table_context(
    model_id: str,
    dtype: str,
    prompt_tokens: int,
    execution: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "runtime_model": "torch_eager_v1",
        "model_id": model_id,
        "dtype": dtype,
        "prompt_tokens": int(prompt_tokens),
        "parallel_mode": execution["parallel_mode"],
        "tp_size": int(execution["tp_size"]),
        "world_size": int(execution["world_size"]),
        "nnodes": int(execution["nnodes"]),
        "interconnect": execution["interconnect"],
        "device_name": calibration["device_name"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _entry_matches(
    entry: dict[str, Any],
    context: dict[str, Any],
    phase: str,
    module_scope: str,
) -> bool:
    if entry.get("schema_version") != TABLE_SCHEMA_VERSION:
        return False
    if entry.get("record_type") != "module_profile":
        return False
    key = entry.get("key", {})
    required = {
        "runtime_model": context["runtime_model"],
        "model_id": context["model_id"],
        "dtype": context["dtype"],
        "parallel_mode": context["parallel_mode"],
        "tp_size": context["tp_size"],
        "world_size": context["world_size"],
        "nnodes": context["nnodes"],
        "interconnect": context["interconnect"],
        "device_name": context["device_name"],
        "phase": phase,
        "module_scope": module_scope,
    }
    if not all(
        key.get(field) == expected
        for field, expected in required.items()
        if field != "model_id"
    ):
        return False
    entry_model_id = stable_model_identifier(
        model_id=key.get("model_id"),
        model_path=key.get("model_path"),
    )
    return entry_model_id == context["model_id"]


def _phase_adjustment_entry_matches(
    entry: dict[str, Any], context: dict[str, Any], phase: str
) -> bool:
    if entry.get("schema_version") != TABLE_SCHEMA_VERSION:
        return False
    if entry.get("record_type") != "phase_adjustment_profile":
        return False
    key = entry.get("key", {})
    required = {
        "runtime_model": context["runtime_model"],
        "model_id": context["model_id"],
        "dtype": context["dtype"],
        "parallel_mode": context["parallel_mode"],
        "tp_size": context["tp_size"],
        "world_size": context["world_size"],
        "nnodes": context["nnodes"],
        "interconnect": context["interconnect"],
        "device_name": context["device_name"],
        "phase": phase,
    }
    if not all(
        key.get(field) == expected
        for field, expected in required.items()
        if field != "model_id"
    ):
        return False
    entry_model_id = stable_model_identifier(
        model_id=key.get("model_id"),
        model_path=key.get("model_path"),
    )
    return entry_model_id == context["model_id"]


def _record_from_entry(
    entry: dict[str, Any],
    phase_estimates: list[NodeEstimate],
    substitution_policy: str,
) -> ModuleProfileRecord | None:
    value = entry.get("value", {})
    scope = value.get("module_scope") or entry.get("key", {}).get("module_scope")
    if not isinstance(scope, str):
        return None
    covered = covered_estimates_for_scope(scope, phase_estimates)
    if not covered:
        return None
    samples = value.get("samples_ms")
    if not isinstance(samples, list) or not samples:
        samples = [float(value.get("mean_ms", 0.0))]
    return ModuleProfileRecord(
        module_scope=scope,
        module_kind=str(value.get("module_kind", scope.split(".")[-1])),
        phase=str(value.get("phase", entry.get("key", {}).get("phase", "unknown"))),
        covered_node_ids=[item.node_name for item in covered],
        covered_op_families=sorted({item.op_family for item in covered}),
        substitution_policy=substitution_policy,
        mean_ms=float(value.get("mean_ms", 0.0)),
        median_ms=float(value.get("median_ms", value.get("mean_ms", 0.0))),
        min_ms=float(value.get("min_ms", value.get("mean_ms", 0.0))),
        max_ms=float(value.get("max_ms", value.get("mean_ms", 0.0))),
        samples_ms=[float(sample) for sample in samples],
    )


def load_module_profiles_from_table(
    table_db_path: str,
    context: dict[str, Any],
    prefill_estimates: list[NodeEstimate],
    decode_estimates: list[NodeEstimate],
    scope_modes: dict[str, str],
) -> tuple[dict[str, list[ModuleProfileRecord]], dict[str, Any]]:
    path = Path(table_db_path).expanduser().resolve()
    entries = _read_jsonl(path)

    estimate_lookup = {
        "prefill": prefill_estimates,
        "decode_step": decode_estimates,
    }
    profiles: dict[str, list[ModuleProfileRecord]] = {
        "prefill": [],
        "decode_step": [],
    }
    stats: dict[str, Any] = {
        "status": "loaded" if entries else "empty",
        "table_db_path": str(path),
        "records_total": len(entries),
    }

    for phase in ("prefill", "decode_step"):
        expected_scopes = expected_profile_scopes(
            estimate_lookup[phase], scope_modes.get(phase, "submodule")
        )
        exact_hits = 0
        nearest_hits = 0
        phase_records: list[ModuleProfileRecord] = []
        for scope in expected_scopes:
            candidates = [
                item
                for item in entries
                if _entry_matches(item, context, phase=phase, module_scope=scope)
            ]
            if not candidates:
                continue

            prompt_tokens = int(context["prompt_tokens"])
            exact_candidates = [
                item
                for item in candidates
                if int(item.get("key", {}).get("prompt_tokens", -1)) == prompt_tokens
            ]
            if exact_candidates:
                selected = exact_candidates[-1]
                substitution_policy = "module_profile_table_lookup"
                exact_hits += 1
            else:
                selected = min(
                    candidates,
                    key=lambda item: abs(
                        int(item.get("key", {}).get("prompt_tokens", 0)) - prompt_tokens
                    ),
                )
                substitution_policy = "module_profile_table_nearest_prompt_tokens"
                nearest_hits += 1

            record = _record_from_entry(
                selected,
                phase_estimates=estimate_lookup[phase],
                substitution_policy=substitution_policy,
            )
            if record is not None:
                phase_records.append(record)

        profiles[phase] = sorted(
            phase_records, key=lambda item: module_scope_key(item.module_scope)
        )
        stats[phase] = {
            "expected_scope_count": len(expected_scopes),
            "loaded_scope_count": len(phase_records),
            "exact_prompt_hits": exact_hits,
            "nearest_prompt_hits": nearest_hits,
            "missing_scope_count": max(len(expected_scopes) - len(phase_records), 0),
        }

    return profiles, stats


def load_phase_adjustments_from_table(
    table_db_path: str, context: dict[str, Any]
) -> tuple[dict[str, PhaseAdjustmentProfile | None], dict[str, Any]]:
    path = Path(table_db_path).expanduser().resolve()
    entries = _read_jsonl(path)
    profiles: dict[str, PhaseAdjustmentProfile | None] = {
        "prefill": None,
        "decode_step": None,
    }
    stats: dict[str, Any] = {
        "status": "loaded" if entries else "empty",
        "table_db_path": str(path),
        "records_total": len(entries),
    }

    for phase in ("prefill", "decode_step"):
        candidates = [
            item
            for item in entries
            if _phase_adjustment_entry_matches(item, context, phase=phase)
        ]
        if not candidates:
            stats[phase] = {
                "loaded": False,
                "exact_prompt_hits": 0,
                "nearest_prompt_hits": 0,
            }
            continue

        prompt_tokens = int(context["prompt_tokens"])
        exact_candidates = [
            item
            for item in candidates
            if int(item.get("key", {}).get("prompt_tokens", -1)) == prompt_tokens
        ]
        if exact_candidates:
            selected = exact_candidates[-1]
            substitution_policy = "phase_adjustment_table_lookup"
            exact_hits = 1
            nearest_hits = 0
        else:
            selected = min(
                candidates,
                key=lambda item: abs(
                    int(item.get("key", {}).get("prompt_tokens", 0)) - prompt_tokens
                ),
            )
            substitution_policy = "phase_adjustment_table_nearest_prompt_tokens"
            exact_hits = 0
            nearest_hits = 1

        value = selected.get("value", {})
        samples = value.get("samples_ms")
        if not isinstance(samples, list) or not samples:
            samples = [float(value.get("mean_ms", 0.0))]
        profiles[phase] = PhaseAdjustmentProfile(
            phase=phase,
            mean_ms=float(value.get("mean_ms", 0.0)),
            median_ms=float(value.get("median_ms", value.get("mean_ms", 0.0))),
            min_ms=float(value.get("min_ms", value.get("mean_ms", 0.0))),
            max_ms=float(value.get("max_ms", value.get("mean_ms", 0.0))),
            samples_ms=[float(sample) for sample in samples],
        )
        stats[phase] = {
            "loaded": True,
            "exact_prompt_hits": exact_hits,
            "nearest_prompt_hits": nearest_hits,
            "substitution_policy": substitution_policy,
        }

    return profiles, stats


def missing_profile_scopes(
    phase_estimates: list[NodeEstimate],
    scope_mode: str,
    existing_records: list[ModuleProfileRecord],
) -> set[str]:
    expected = set(expected_profile_scopes(phase_estimates, scope_mode))
    loaded = {record.module_scope for record in existing_records}
    return expected - loaded


def merge_module_profiles(
    base: dict[str, list[ModuleProfileRecord]],
    extra: dict[str, list[ModuleProfileRecord]],
) -> dict[str, list[ModuleProfileRecord]]:
    merged: dict[str, list[ModuleProfileRecord]] = {"prefill": [], "decode_step": []}
    for phase in merged:
        by_scope: dict[str, ModuleProfileRecord] = {}
        for record in base.get(phase, []):
            by_scope[record.module_scope] = record
        for record in extra.get(phase, []):
            by_scope.setdefault(record.module_scope, record)
        merged[phase] = sorted(
            by_scope.values(), key=lambda item: module_scope_key(item.module_scope)
        )
    return merged


def append_module_profiles_to_table(
    table_db_path: str,
    context: dict[str, Any],
    module_profiles: dict[str, list[ModuleProfileRecord]],
    source: str,
) -> int:
    path = Path(table_db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for phase in ("prefill", "decode_step"):
        for record in module_profiles.get(phase, []):
            rows.append(
                {
                    "schema_version": TABLE_SCHEMA_VERSION,
                    "record_type": "module_profile",
                    "created_at": _iso_now(),
                    "key": {
                        **context,
                        "phase": phase,
                        "module_scope": record.module_scope,
                    },
                    "value": {
                        "module_scope": record.module_scope,
                        "module_kind": record.module_kind,
                        "phase": phase,
                        "mean_ms": record.mean_ms,
                        "median_ms": record.median_ms,
                        "min_ms": record.min_ms,
                        "max_ms": record.max_ms,
                        "samples_ms": list(record.samples_ms),
                    },
                    "provenance": {
                        "source": source,
                    },
                }
            )

    if not rows:
        return 0

    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(rows)


def append_phase_adjustments_to_table(
    table_db_path: str,
    context: dict[str, Any],
    phase_adjustments: dict[str, PhaseAdjustmentProfile | None],
    source: str,
) -> int:
    path = Path(table_db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for phase in ("prefill", "decode_step"):
        record = phase_adjustments.get(phase)
        if record is None:
            continue
        rows.append(
            {
                "schema_version": TABLE_SCHEMA_VERSION,
                "record_type": "phase_adjustment_profile",
                "created_at": _iso_now(),
                "key": {
                    **context,
                    "phase": phase,
                },
                "value": {
                    "phase": phase,
                    "mean_ms": record.mean_ms,
                    "median_ms": record.median_ms,
                    "min_ms": record.min_ms,
                    "max_ms": record.max_ms,
                    "samples_ms": list(record.samples_ms),
                },
                "provenance": {
                    "source": source,
                },
            }
        )

    if not rows:
        return 0

    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(rows)


def module_profiles_to_dict(
    module_profiles: dict[str, list[ModuleProfileRecord]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "prefill": [asdict(record) for record in module_profiles.get("prefill", [])],
        "decode_step": [
            asdict(record) for record in module_profiles.get("decode_step", [])
        ],
    }
