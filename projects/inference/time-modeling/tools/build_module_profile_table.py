from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mvp_model import stable_model_identifier

TABLE_SCHEMA_VERSION = "mvp_module_profile_table_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import report.json module profiles into table database"
    )
    parser.add_argument(
        "--reports-glob",
        default="validation_reports/**/report.json",
        help="Glob pattern to locate report.json files",
    )
    parser.add_argument(
        "--table-db-path",
        default="database/module_profile_table.jsonl",
        help="Output table DB jsonl path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output DB instead of append",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _build_key(report: dict[str, Any], phase: str, module_scope: str) -> dict[str, Any]:
    model = report.get("model", {})
    execution = report.get("execution", {})
    calibration = report.get("calibration", {})
    return {
        "runtime_model": report.get("runtime_model", "torch_eager_v1"),
        "model_id": stable_model_identifier(
            model_id=model.get("id"),
            model_path=model.get("path"),
        ),
        "dtype": model.get("dtype", ""),
        "prompt_tokens": int(model.get("prompt_tokens", 0) or 0),
        "parallel_mode": execution.get("parallel_mode", "single"),
        "tp_size": int(execution.get("tp_size", 1) or 1),
        "world_size": int(execution.get("world_size", 1) or 1),
        "nnodes": int(execution.get("nnodes", 1) or 1),
        "interconnect": execution.get("interconnect", "local"),
        "device_name": calibration.get("device_name", "unknown"),
        "phase": phase,
        "module_scope": module_scope,
    }


def _build_phase_adjustment_key(report: dict[str, Any], phase: str) -> dict[str, Any]:
    key = _build_key(report, phase=phase, module_scope="__phase_adjustment__")
    key.pop("module_scope", None)
    return key


def _row_created_at(report_path: Path, report: dict[str, Any]) -> str:
    metadata = report.get("module_profile_meta", {})
    created_at = metadata.get("report_created_at")
    if isinstance(created_at, str) and created_at:
        return created_at
    return datetime.fromtimestamp(
        report_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()


def _build_rows(report_path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    module_profile = report.get("module_profile")
    if not isinstance(module_profile, dict):
        return []

    rows: list[dict[str, Any]] = []
    created_at = _row_created_at(report_path, report)
    for phase in ("prefill", "decode_step"):
        records = module_profile.get(phase, [])
        if not isinstance(records, list):
            continue
        for item in records:
            if not isinstance(item, dict):
                continue
            module_scope = item.get("module_scope")
            if not isinstance(module_scope, str):
                continue
            row = {
                "schema_version": TABLE_SCHEMA_VERSION,
                "record_type": "module_profile",
                "created_at": created_at,
                "key": _build_key(report, phase=phase, module_scope=module_scope),
                "value": {
                    "module_scope": module_scope,
                    "module_kind": item.get("module_kind", module_scope.split(".")[-1]),
                    "phase": phase,
                    "mean_ms": float(item.get("mean_ms", 0.0)),
                    "median_ms": float(item.get("median_ms", item.get("mean_ms", 0.0))),
                    "min_ms": float(item.get("min_ms", item.get("mean_ms", 0.0))),
                    "max_ms": float(item.get("max_ms", item.get("mean_ms", 0.0))),
                    "samples_ms": [float(value) for value in item.get("samples_ms", [])]
                    or [float(item.get("mean_ms", 0.0))],
                },
                "provenance": {
                    "source": "report_import",
                    "report_path": str(report_path),
                },
            }
            rows.append(row)

    phase_adjustment = report.get("phase_adjustment")
    if isinstance(phase_adjustment, dict):
        for phase in ("prefill", "decode_step"):
            item = phase_adjustment.get(phase)
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "schema_version": TABLE_SCHEMA_VERSION,
                    "record_type": "phase_adjustment_profile",
                    "created_at": created_at,
                    "key": _build_phase_adjustment_key(report, phase=phase),
                    "value": {
                        "phase": phase,
                        "mean_ms": float(item.get("mean_ms", 0.0)),
                        "median_ms": float(
                            item.get("median_ms", item.get("mean_ms", 0.0))
                        ),
                        "min_ms": float(item.get("min_ms", item.get("mean_ms", 0.0))),
                        "max_ms": float(item.get("max_ms", item.get("mean_ms", 0.0))),
                        "samples_ms": [
                            float(value) for value in item.get("samples_ms", [])
                        ]
                        or [float(item.get("mean_ms", 0.0))],
                    },
                    "provenance": {
                        "source": "report_import",
                        "report_path": str(report_path),
                    },
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    report_paths = sorted(Path.cwd().glob(args.reports_glob))
    table_path = Path(args.table_db_path)
    if not table_path.is_absolute():
        table_path = Path.cwd() / table_path
    table_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and table_path.exists():
        table_path.unlink()

    imported_rows = 0
    imported_reports = 0
    with table_path.open("a", encoding="utf-8") as handle:
        for path in report_paths:
            report = _read_json(path)
            if report is None:
                continue
            rows = _build_rows(path, report)
            if not rows:
                continue
            imported_reports += 1
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                imported_rows += 1

    print(
        json.dumps(
            {
                "table_db_path": str(table_path),
                "reports_scanned": len(report_paths),
                "reports_imported": imported_reports,
                "rows_imported": imported_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
