from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvp_dashboard import (
    RunRecord,
    configure_dashboard_settings,
    dashboard_request_defaults,
    environment_state_payload,
    prepare_environment,
    run_job,
    stop_environment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and reuse the managed MVP runtime environment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--config",
            default=None,
            help="Dashboard environment config JSON path",
        )

    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare managed environment"
    )
    add_config_argument(prepare_parser)
    prepare_parser.add_argument(
        "--force", action="store_true", help="Recreate the managed environment"
    )

    status_parser = subparsers.add_parser(
        "status", help="Show managed environment status"
    )
    add_config_argument(status_parser)
    stop_parser = subparsers.add_parser("stop", help="Stop managed environment")
    add_config_argument(stop_parser)

    run_parser = subparsers.add_parser(
        "run", help="Run estimate/measurement using config defaults"
    )
    add_config_argument(run_parser)
    run_parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Recreate the managed environment before running",
    )
    run_parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Override config and run estimate-only",
    )
    run_parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip graph export for this run",
    )
    return parser


def print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_dashboard_settings(args.config)

    if args.command == "prepare":
        print_json(prepare_environment(force=bool(args.force)))
        return 0

    if args.command == "status":
        print_json(environment_state_payload())
        return 0

    if args.command == "stop":
        print_json(stop_environment())
        return 0

    environment = prepare_environment(force=bool(args.force_prepare))
    if not environment.get("ready"):
        print_json(environment)
        return 1

    payload = dashboard_request_defaults()
    if args.estimate_only:
        payload["estimate_only"] = True
    if args.no_graph:
        payload["generate_graph_viz"] = False
    record = RunRecord(
        run_id=f"cli{uuid.uuid4().hex[:8]}",
        request=payload,
        created_at=time.time(),
    )
    run_job(record)
    summary = {
        "status": record.status,
        "output_dir": record.output_dir,
        "return_code": record.return_code,
        "timings": record.timings,
        "output_policy": record.output_policy,
        "error": record.error,
    }
    print_json(summary)
    return 0 if record.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
