from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TP training report to dashboard summary")
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    execution = report.get("execution", {})
    measured = report.get("measured", {})
    estimate = report.get("estimate", {})
    now = time.time()
    summary = {
        "task": "training",
        "success": True,
        "started_at": now,
        "finished_at": now,
        "duration_seconds": 0.0,
        "model_path": (report.get("model") or {}).get("path"),
        "parallel_mode": execution.get("parallel_mode"),
        "world_size": execution.get("world_size"),
        "tp_size": execution.get("tp_size"),
        "physical_devices": execution.get("physical_devices", []),
        "microbatch_count": execution.get("microbatch_count"),
        "microbatch_size": execution.get("microbatch_size"),
        "sequence_length": execution.get("sequence_length"),
        "iterations": execution.get("benchmark_repeat"),
        "optimizer_type": execution.get("optimizer_type"),
        "gradient_checkpointing": execution.get("gradient_checkpointing"),
        "training_mode": execution.get("training_mode"),
        "mean_iteration_time_ms": measured.get("train_iteration_time_ms"),
        "estimated_iteration_time_ms": estimate.get("train_iteration_time_ms"),
        "report_path": str(report_path),
        "report_md_path": str(output_dir / "report.md"),
        "note": "TP training runtime validation report adapted for dashboard completion state.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
