from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cambricon inference adaptation run wrapper"
    )
    parser.add_argument("--infer-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="mlu:0")
    parser.add_argument("--parallel-mode", choices=["single", "tp"], default="single")
    parser.add_argument("--physical-devices", default="0")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeat", type=int, default=1)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--estimate-mode", choices=["online", "table", "hybrid"], default="hybrid")
    parser.add_argument("--table-db-path", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_repo = Path(args.infer_repo).expanduser().resolve()
    report_path = output_dir / "report.json"

    script_args = [
        str(infer_repo / "torch_infer_mvp.py"),
        "--model-path",
        args.model_path,
        "--prompt",
        args.prompt,
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
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(args.world_size),
        "--node-rank",
        "0",
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        "29500",
        "--interconnect",
        "ethernet",
        "--dist-timeout-minutes",
        "30",
        "--warmup",
        str(args.warmup),
        "--benchmark-repeat",
        str(args.benchmark_repeat),
        "--profile-repeat",
        str(args.profile_repeat),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output-dir",
        str(output_dir),
        "--estimate-mode",
        args.estimate_mode,
    ]
    if args.table_db_path.strip():
        script_args.extend(["--table-db-path", args.table_db_path.strip()])
        script_args.append("--table-writeback")
    if args.parallel_mode == "tp" and args.world_size > 1:
        command = [
            "python3",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.world_size),
            *script_args,
        ]
    else:
        command = ["python3", *script_args]

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(infer_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    finished = time.time()
    (output_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")

    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    success = (
        completed.returncode == 0
        and report is not None
        and not report.get("error")
        and bool((report.get("model") or {}).get("generated_text", "").strip())
    )

    summary = {
        "task": "inference",
        "success": success,
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": finished - started,
        "returncode": completed.returncode,
        "command": command,
        "model_path": args.model_path,
        "parallel_mode": args.parallel_mode,
        "physical_devices": args.physical_devices,
        "generated_text": (report.get("model") or {}).get("generated_text", "") if report else "",
        "generated_token_ids": (report.get("model") or {}).get("generated_token_ids", []) if report else [],
        "request_time_ms": (((report or {}).get("measured") or {}).get("request") or {}).get("mean_ms"),
        "prefill_time_ms": (((report or {}).get("measured") or {}).get("prefill") or {}).get("mean_ms"),
        "decode_time_ms": (((report or {}).get("measured") or {}).get("decode_step") or {}).get("mean_ms"),
        "error": None if success else "Inference run failed or produced empty generation.",
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
