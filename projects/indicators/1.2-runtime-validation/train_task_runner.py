from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cambricon TP training adaptation run wrapper"
    )
    parser.add_argument("--train-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="mlu:0")
    parser.add_argument("--physical-devices", default="0")
    parser.add_argument("--parallel-mode", choices=["single", "tp"], default="single")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer-type", choices=["adamw", "sgd"], default="sgd")
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument("--optimizer-foreach", action="store_true")
    parser.add_argument("--enable-gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--adapter-num-labels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_summary(report_path: Path, output_dir: Path) -> None:
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


def main() -> None:
    args = parse_args()
    train_repo = Path(args.train_repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tp_script = train_repo / "torch_train_tp_mvp.py"
    if not tp_script.exists():
        raise RuntimeError(f"Missing TP training script: {tp_script}")
    if args.parallel_mode == "tp":
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.tp_size),
            str(tp_script),
        ]
    else:
        command = [sys.executable, str(tp_script)]
    command.extend(
        [
            "--model-path",
            args.model_path,
            "--dtype",
            args.dtype,
            "--device",
            args.device,
            "--parallel-mode",
            args.parallel_mode,
            "--physical-devices",
            args.physical_devices,
            "--world-size",
            str(args.world_size),
            "--tp-size",
            str(args.tp_size),
            "--nproc-per-node",
            str(args.tp_size if args.parallel_mode == "tp" else 1),
            "--microbatch-count",
            str(args.microbatch_count),
            "--microbatch-size",
            str(args.microbatch_size),
            "--sequence-length",
            str(args.sequence_length),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--optimizer-type",
            args.optimizer_type,
            "--sgd-momentum",
            str(args.sgd_momentum),
            "--lora-rank",
            str(args.lora_rank),
            "--lora-alpha",
            str(args.lora_alpha),
            "--adapter-num-labels",
            str(args.adapter_num_labels),
            "--warmup",
            str(args.warmup),
            "--benchmark-repeat",
            str(args.iterations),
            "--profile-repeat",
            str(args.profile_repeat),
            "--estimate-mode",
            "online",
            "--train-config-path",
            str(train_repo / "configs" / "train_config.yaml"),
            "--profile-db-path",
            str(train_repo / "database" / "train_component_profile_tp.jsonl"),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(output_dir),
        ]
    )
    if args.enable_gradient_checkpointing:
        command.append("--enable-gradient-checkpointing")
    if args.optimizer_foreach:
        command.append("--optimizer-foreach")
    subprocess.run(command, check=True)
    write_summary(output_dir / "report.json", output_dir)


if __name__ == "__main__":
    main()
