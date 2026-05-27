from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
UI_DIR = ROOT / "ui"
DEFAULT_HOST = os.environ.get("RUNTEST_DASHBOARD_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("RUNTEST_DASHBOARD_PORT", "8242"))
DEFAULT_IMAGE = "cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310"
DEFAULT_MODEL_DIR = "/home/o_mabin/LLM/models/Llama-3.1-8B"
INFER_REPO = REPO_ROOT / "projects" / "inference" / "time-modeling"
TRAIN_REPO = REPO_ROOT / "projects" / "training" / "time-modeling"
DEFAULT_INFER_TABLE = "/deps/infer/database/module_profile_table_cambricon_mlu580.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "runs"


@dataclass
class RunRecord:
    run_id: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    command: list[str] = field(default_factory=list)
    command_text: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    output_dir: str = ""
    stdout: str = ""
    stderr: str = ""
    summary: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        reconcile_run(self)
        payload = asdict(self)
        payload["duration_seconds"] = (
            (self.finished_at or time.time()) - self.started_at
            if self.started_at
            else None
        )
        return payload


RUNS: dict[str, RunRecord] = {}
RUNS_LOCK = threading.Lock()


def reconcile_run(record: RunRecord) -> None:
    if not record.output_dir:
        return
    summary_path = Path(record.output_dir) / "summary.json"
    if not summary_path.exists():
        return
    try:
        record.summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if bool((record.summary or {}).get("success")):
        record.status = "completed"
        if record.returncode is None:
            record.returncode = 0
        if record.finished_at is None:
            record.finished_at = (record.summary or {}).get("finished_at") or time.time()
    elif record.status == "running":
        record.status = "failed"
        if record.finished_at is None:
            record.finished_at = time.time()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def default_payload() -> dict[str, Any]:
    return {
        "task": "inference",
        "scale": "single",
        "image": DEFAULT_IMAGE,
        "host_model_dir": DEFAULT_MODEL_DIR,
        "prompt": "alpha alpha alpha alpha alpha alpha alpha alpha",
        "max_new_tokens": 4,
        "dtype": "bf16",
        "warmup": 1,
        "benchmark_repeat": 1,
        "profile_repeat": 1,
        "physical_devices": "0",
        "world_size": 1,
        "tp_size": 1,
        "microbatch_count": 2,
        "microbatch_size": 1,
        "sequence_length": 32,
        "iterations": 3,
        "optimizer_type": "sgd",
        "enable_gradient_checkpointing": True,
        "optimizer_foreach": False,
        "output_root": str(DEFAULT_OUTPUT_ROOT),
    }


def infer_single_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update({"task": "inference", "scale": "single", "physical_devices": "0"})
    return payload


def infer_dual_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update(
        {
            "task": "inference",
            "scale": "dual",
            "physical_devices": "0,1",
        }
    )
    return payload


def train_single_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update(
        {
            "task": "training",
            "scale": "single",
            "physical_devices": "0",
            "world_size": 1,
            "tp_size": 1,
            "microbatch_count": 2,
            "sequence_length": 16,
            "iterations": 3,
            "enable_gradient_checkpointing": True,
        }
    )
    return payload


def train_dual_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update(
        {
            "task": "training",
            "scale": "dual",
            "physical_devices": "0,1",
            "world_size": 2,
            "tp_size": 2,
            "microbatch_count": 2,
            "sequence_length": 16,
            "iterations": 3,
            "enable_gradient_checkpointing": False,
        }
    )
    return payload


def normalize_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    payload = default_payload()
    payload.update(raw_payload or {})
    payload["task"] = str(payload["task"]).strip()
    payload["scale"] = str(payload["scale"]).strip()
    payload["image"] = str(payload["image"]).strip()
    payload["host_model_dir"] = str(payload["host_model_dir"]).strip()
    payload["prompt"] = str(payload["prompt"]).strip()
    payload["dtype"] = str(payload["dtype"]).strip()
    payload["physical_devices"] = str(payload["physical_devices"]).strip()
    payload["optimizer_type"] = str(payload["optimizer_type"]).strip()
    payload["output_root"] = str(payload["output_root"]).strip()
    payload["max_new_tokens"] = int(payload["max_new_tokens"])
    payload["warmup"] = int(payload["warmup"])
    payload["benchmark_repeat"] = int(payload["benchmark_repeat"])
    payload["profile_repeat"] = int(payload["profile_repeat"])
    payload["world_size"] = int(payload["world_size"])
    payload["tp_size"] = int(payload["tp_size"])
    payload["microbatch_count"] = int(payload["microbatch_count"])
    payload["microbatch_size"] = int(payload["microbatch_size"])
    payload["sequence_length"] = int(payload["sequence_length"])
    payload["iterations"] = int(payload["iterations"])
    payload["enable_gradient_checkpointing"] = bool(
        payload.get("enable_gradient_checkpointing")
    )
    payload["optimizer_foreach"] = bool(payload.get("optimizer_foreach"))
    return payload


def host_output_dir(run_id: str, payload: dict[str, Any]) -> Path:
    output_root = Path(payload["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / payload["task"]
    if task_dir.exists() and not os.access(task_dir, os.W_OK | os.X_OK):
        task_dir = output_root / f"{payload['task']}_local"
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir = task_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_runner_command(run_id: str, payload: dict[str, Any]) -> tuple[list[str], Path]:
    output_dir = host_output_dir(run_id, payload)
    workspace_output_dir = "/workspace/" + str(output_dir.relative_to(ROOT)).replace("\\", "/")
    common_prefix = [
        "source /torch/venv3/pytorch/bin/activate",
        "mkdir -p /workspace/reports/runs/inference /workspace/reports/runs/training",
    ]
    adapter_command: list[str] | None = None
    if payload["task"] == "inference":
        python_command = [
            "python3",
            "/workspace/infer_task_runner.py",
            "--infer-repo",
            "/deps/infer",
            "--model-path",
            "/model",
            "--prompt",
            payload["prompt"],
            "--max-new-tokens",
            str(payload["max_new_tokens"]),
            "--dtype",
            payload["dtype"],
            "--device",
            "mlu:0",
            "--parallel-mode",
            "tp" if payload["scale"] == "dual" else "single",
            "--physical-devices",
            payload["physical_devices"],
            "--world-size",
            "2" if payload["scale"] == "dual" else "1",
            "--tp-size",
            "2" if payload["scale"] == "dual" else "1",
            "--warmup",
            str(payload["warmup"]),
            "--benchmark-repeat",
            str(payload["benchmark_repeat"]),
            "--profile-repeat",
            str(payload["profile_repeat"]),
            "--estimate-mode",
            "hybrid",
            "--table-db-path",
            DEFAULT_INFER_TABLE,
            "--output-dir",
            workspace_output_dir,
        ]
    else:
        scale_is_dual = payload["scale"] == "dual"
        python_command = [
            "python3",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(payload["tp_size"]),
            "/deps/train/torch_train_tp_mvp.py",
        ] if scale_is_dual else [
            "python3",
            "/deps/train/torch_train_tp_mvp.py",
        ]
        python_command.extend(
            [
            "--model-path",
            "/model",
            "--dtype",
            payload["dtype"],
            "--device",
            "mlu:0",
            "--physical-devices",
            payload["physical_devices"],
            "--parallel-mode",
            "tp" if scale_is_dual else "single",
            "--world-size",
            str(payload["world_size"]),
            "--tp-size",
            str(payload["tp_size"]),
            "--nproc-per-node",
            str(payload["tp_size"] if scale_is_dual else 1),
            "--microbatch-count",
            str(payload["microbatch_count"]),
            "--microbatch-size",
            str(payload["microbatch_size"]),
            "--sequence-length",
            str(payload["sequence_length"]),
            "--optimizer-type",
            payload["optimizer_type"],
            "--warmup",
            str(payload["warmup"]),
            "--benchmark-repeat",
            str(payload["iterations"]),
            "--profile-repeat",
            str(payload["profile_repeat"]),
            "--estimate-mode",
            "online",
            "--train-config-path",
            "/deps/train/configs/train_config.yaml",
            "--profile-db-path",
            "/deps/train/database/train_component_profile_tp.jsonl",
            "--output-dir",
            workspace_output_dir,
            ]
        )
        if payload["enable_gradient_checkpointing"]:
            python_command.append("--enable-gradient-checkpointing")
        if payload["optimizer_foreach"]:
            python_command.append("--optimizer-foreach")
        adapter_command = [
            "python3",
            "/workspace/train_tp_summary_adapter.py",
            "--report-path",
            f"{workspace_output_dir}/report.json",
            "--output-dir",
            workspace_output_dir,
        ]
    inner_commands = common_prefix + [shell_join(python_command)]
    if adapter_command:
        inner_commands.append(shell_join(adapter_command))
    inner = " && ".join(inner_commands)
    command = [
        "docker",
        "run",
        "--rm",
        "--privileged",
        "--net=host",
        "--pid=host",
        "--ipc=host",
        "--cgroupns=host",
        "--shm-size",
        "64gb",
        "-e",
        "CAMBRICON_VISIBLE_DEVICES=all",
        "-e",
        "MLU_VISIBLE_DEVICE=all",
        "-e",
        "PYTORCH_MLU_ALLOC_CONF=expandable_segments:True",
        "-v",
        "/usr/bin/cnmon:/usr/bin/cnmon",
        "-v",
        "/sys/kernel/debug:/sys/kernel/debug",
        "-v",
        f"{ROOT}:/workspace",
        "-v",
        f"{INFER_REPO}:/deps/infer",
        "-v",
        f"{TRAIN_REPO}:/deps/train",
        "-v",
        f"{payload['host_model_dir']}:/model:ro",
        "-v",
        "/data:/data",
        payload["image"],
        "bash",
        "-lc",
        inner,
    ]
    return command, output_dir


def load_saved_runs() -> None:
    output_root = DEFAULT_OUTPUT_ROOT
    if not output_root.exists():
        return
    for summary_path in sorted(output_root.rglob("summary.json")):
        output_dir = summary_path.parent
        run_id = output_dir.name
        with RUNS_LOCK:
            if run_id in RUNS:
                continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload = default_payload()
        payload["task"] = summary.get("task", output_dir.parent.name)
        payload["scale"] = "dual" if len(summary.get("physical_devices", [])) >= 2 else "single"
        record = RunRecord(
            run_id=run_id,
            payload=payload,
            status="completed" if summary.get("success") else "failed",
            created_at=summary_path.stat().st_mtime,
            started_at=summary.get("started_at"),
            finished_at=summary.get("finished_at"),
            returncode=0 if summary.get("success") else 1,
            output_dir=str(output_dir),
            summary=summary,
        )
        with RUNS_LOCK:
            RUNS[run_id] = record


def run_job(record: RunRecord) -> None:
    try:
        command, output_dir = build_runner_command(record.run_id, record.payload)
        record.command = command
        record.command_text = shell_join(command)
        record.output_dir = str(output_dir)
        record.status = "running"
        record.started_at = time.time()
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
        record.stdout = stdout
        record.stderr = stderr
        record.returncode = process.returncode
        record.finished_at = time.time()
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            record.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if process.returncode == 0 and record.summary and record.summary.get("success"):
            record.status = "completed"
        else:
            record.status = "failed"
            record.error = "Task run failed or summary.json did not indicate success."
    except Exception as exc:
        record.finished_at = time.time()
        record.status = "failed"
        record.error = str(exc)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/defaults":
            self.write_json(
                {
                    "defaults": default_payload(),
                    "presets": {
                        "infer_single": infer_single_preset(),
                        "infer_dual": infer_dual_preset(),
                        "train_single": train_single_preset(),
                        "train_dual": train_dual_preset(),
                    },
                }
            )
            return
        if parsed.path == "/api/runs":
            with RUNS_LOCK:
                runs = [
                    item.to_dict()
                    for item in sorted(
                        RUNS.values(), key=lambda record: record.created_at, reverse=True
                    )
                ]
            self.write_json({"runs": runs})
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                record = RUNS.get(run_id)
            if record is None:
                self.write_json({"error": "Run not found"}, status=404)
                return
            self.write_json(record.to_dict())
            return
        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/runs":
            self.write_json({"error": "Not found"}, status=404)
            return
        content_length = int(self.headers.get("Content-Length") or 0)
        payload = {}
        if content_length:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        normalized = normalize_payload(payload)
        run_id = uuid.uuid4().hex[:8]
        record = RunRecord(run_id=run_id, payload=normalized)
        with RUNS_LOCK:
            RUNS[run_id] = record
        thread = threading.Thread(target=run_job, args=(record,), daemon=True)
        thread.start()
        self.write_json({"run_id": run_id, "status": record.status}, status=HTTPStatus.ACCEPTED)


def main() -> None:
    load_saved_runs()
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), DashboardHandler)
    print(f"Run-test dashboard listening on http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
