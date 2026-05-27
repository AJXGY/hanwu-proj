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
UI_DIR = ROOT / "ui"
DEFAULT_HOST = os.environ.get("TRAIN_DASHBOARD_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("TRAIN_DASHBOARD_PORT", "8234"))
DEFAULT_IMAGE = "cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310"
DEFAULT_MODEL_DIR = "/home/o_mabin/LLM/models/Llama-3.1-8B"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "dashboard_runs"
DEFAULT_TP_SINGLE_PROFILE_DB = "/workspace/database/train_component_profile_cambricon_mlu580_tp_single.jsonl"
DEFAULT_TP2_PROFILE_DB = "/workspace/database/train_component_profile_cambricon_mlu580_tp2.jsonl"


@dataclass
class RunRecord:
    run_id: str
    payload: dict[str, Any]
    created_at: float
    status: str = "queued"
    command: list[str] = field(default_factory=list)
    command_text: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    output_dir: str = ""
    report_path: str = ""
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_seconds"] = (
            (self.finished_at or time.time()) - self.started_at
            if self.started_at
            else None
        )
        return payload


RUNS: dict[str, RunRecord] = {}
RUNS_LOCK = threading.Lock()


def default_payload() -> dict[str, Any]:
    return {
        "image": DEFAULT_IMAGE,
        "host_model_dir": DEFAULT_MODEL_DIR,
        "model_dir": "/model",
        "parallel_mode": "tp",
        "world_size": 2,
        "tp_size": 2,
        "microbatch_count": 1,
        "microbatch_size": 1,
        "sequence_length": 8,
        "optimizer_type": "sgd",
        "estimate_mode": "online",
        "warmup": 1,
        "benchmark_repeat": 3,
        "profile_repeat": 3,
        "physical_devices": "0,1",
        "profile_db_path": DEFAULT_TP2_PROFILE_DB,
        "enable_gradient_checkpointing": True,
        "optimizer_foreach": False,
        "write_profile_db": True,
        "output_root": str(DEFAULT_OUTPUT_ROOT),
    }


def single_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update(
        {
            "parallel_mode": "single",
            "world_size": 1,
            "tp_size": 1,
            "physical_devices": "0",
            "profile_db_path": DEFAULT_TP_SINGLE_PROFILE_DB,
        }
    )
    return payload


def tp2_preset() -> dict[str, Any]:
    return default_payload()


def normalize_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    payload = default_payload()
    payload.update(raw_payload or {})
    payload["parallel_mode"] = str(payload.get("parallel_mode") or "tp").strip()
    payload["world_size"] = int(payload["world_size"])
    payload["tp_size"] = int(payload["tp_size"])
    payload["microbatch_count"] = int(payload["microbatch_count"])
    payload["microbatch_size"] = int(payload["microbatch_size"])
    payload["sequence_length"] = int(payload["sequence_length"])
    payload["warmup"] = int(payload["warmup"])
    payload["benchmark_repeat"] = int(payload["benchmark_repeat"])
    payload["profile_repeat"] = int(payload["profile_repeat"])
    payload["host_model_dir"] = str(payload["host_model_dir"]).strip()
    payload["model_dir"] = str(payload.get("model_dir") or "/model").strip() or "/model"
    payload["physical_devices"] = str(payload.get("physical_devices") or "").strip()
    payload["image"] = str(payload["image"]).strip()
    payload["optimizer_type"] = str(payload["optimizer_type"]).strip()
    payload["estimate_mode"] = str(payload["estimate_mode"]).strip()
    payload["profile_db_path"] = str(payload["profile_db_path"]).strip()
    payload["output_root"] = str(payload["output_root"]).strip()
    payload["enable_gradient_checkpointing"] = bool(
        payload.get("enable_gradient_checkpointing")
    )
    payload["optimizer_foreach"] = bool(payload.get("optimizer_foreach"))
    payload["write_profile_db"] = bool(payload.get("write_profile_db"))
    if payload["parallel_mode"] == "single":
        payload["world_size"] = 1
        payload["tp_size"] = 1
        if not payload["physical_devices"]:
            payload["physical_devices"] = "0"
        if payload["profile_db_path"] == DEFAULT_TP2_PROFILE_DB:
            payload["profile_db_path"] = DEFAULT_TP_SINGLE_PROFILE_DB
    else:
        payload["parallel_mode"] = "tp"
        payload["world_size"] = max(payload["world_size"], payload["tp_size"])
        if not payload["physical_devices"]:
            payload["physical_devices"] = "0,1"
        if payload["profile_db_path"] == DEFAULT_TP_SINGLE_PROFILE_DB:
            payload["profile_db_path"] = DEFAULT_TP2_PROFILE_DB
    return payload


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_inner_python_command(payload: dict[str, Any], run_id: str) -> tuple[list[str], Path]:
    host_output_root = Path(payload["output_root"]).expanduser().resolve()
    host_output_root.mkdir(parents=True, exist_ok=True)
    host_output_dir = host_output_root / run_id
    host_output_dir.mkdir(parents=True, exist_ok=True)
    workspace_output_dir = f"/workspace/reports/dashboard_runs/{run_id}"
    training_args = [
        "python",
        "/workspace/torch_train_tp_mvp.py",
        "--model-path",
        payload["model_dir"],
        "--dtype",
        "bf16",
        "--device",
        "mlu:0",
        "--parallel-mode",
        payload["parallel_mode"],
        "--world-size",
        str(payload["world_size"]),
        "--tp-size",
        str(payload["tp_size"]),
        "--nproc-per-node",
        str(payload["world_size"]),
        "--microbatch-count",
        str(payload["microbatch_count"]),
        "--microbatch-size",
        str(payload["microbatch_size"]),
        "--sequence-length",
        str(payload["sequence_length"]),
        "--optimizer-type",
        payload["optimizer_type"],
        "--physical-devices",
        payload["physical_devices"],
        "--estimate-mode",
        payload["estimate_mode"],
        "--profile-db-path",
        payload["profile_db_path"],
        "--warmup",
        str(payload["warmup"]),
        "--benchmark-repeat",
        str(payload["benchmark_repeat"]),
        "--profile-repeat",
        str(payload["profile_repeat"]),
        "--output-dir",
        workspace_output_dir,
    ]
    if payload["parallel_mode"] == "tp" and payload["tp_size"] > 1:
        args = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(payload["world_size"]),
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            "29501",
            *training_args[1:],
        ]
    else:
        args = training_args
    if payload["enable_gradient_checkpointing"]:
        args.append("--enable-gradient-checkpointing")
    if payload["optimizer_foreach"]:
        args.append("--optimizer-foreach")
    if payload["write_profile_db"]:
        args.append("--write-profile-db")
    return args, host_output_dir


def build_docker_command(payload: dict[str, Any], run_id: str) -> tuple[list[str], Path, Path]:
    python_args, host_output_dir = build_inner_python_command(payload, run_id)
    inner = " && ".join(
        [
            "source /torch/venv3/pytorch/bin/activate",
            "export PYTHONPATH=/workspace/src",
            "mkdir -p /workspace/database /workspace/reports/dashboard_runs",
            shell_join(python_args),
        ]
    )
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
        f"{payload['host_model_dir']}:{payload['model_dir']}:ro",
        "-v",
        "/data:/data",
        payload["image"],
        "bash",
        "-lc",
        inner,
    ]
    report_path = host_output_dir / "report.json"
    return command, host_output_dir, report_path


def load_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_training_job(record: RunRecord) -> None:
    try:
        command, host_output_dir, report_path = build_docker_command(record.payload, record.run_id)
        record.command = command
        record.command_text = shell_join(command)
        record.output_dir = str(host_output_dir)
        record.report_path = str(report_path)
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
        record.report = load_report(report_path)
        if process.returncode == 0 and record.report is not None:
            record.status = "succeeded"
        else:
            record.status = "failed"
            if record.report is None and not record.error:
                record.error = "Run finished without producing report.json"
    except Exception as exc:  # noqa: BLE001
        record.status = "failed"
        record.finished_at = time.time()
        record.error = str(exc)


def start_run(payload: dict[str, Any]) -> RunRecord:
    normalized = normalize_payload(payload)
    run_id = uuid.uuid4().hex[:8]
    record = RunRecord(run_id=run_id, payload=normalized, created_at=time.time())
    with RUNS_LOCK:
        RUNS[run_id] = record
    thread = threading.Thread(target=run_training_job, args=(record,), daemon=True)
    thread.start()
    return record


def run_list_payload() -> list[dict[str, Any]]:
    with RUNS_LOCK:
        records = list(RUNS.values())
    records.sort(key=lambda item: item.created_at, reverse=True)
    return [record.to_dict() for record in records]


def run_detail_payload(run_id: str) -> dict[str, Any] | None:
    with RUNS_LOCK:
        record = RUNS.get(run_id)
    return None if record is None else record.to_dict()


def existing_report_list() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for report_path in sorted((ROOT / "reports").glob("**/report.json"), reverse=True):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        execution = report.get("execution", {})
        comparison = report.get("comparison", {})
        reports.append(
            {
                "path": str(report_path),
                "parallel_mode": execution.get("parallel_mode"),
                "tp_size": execution.get("tp_size"),
                "microbatch_count": execution.get("microbatch_count"),
                "optimizer_type": execution.get("optimizer_type"),
                "gradient_checkpointing": execution.get("gradient_checkpointing"),
                "error_pct": comparison.get(
                    "train_iteration_relative_error_pct",
                    report.get("error_pct"),
                ),
                "mtime": report_path.stat().st_mtime,
            }
        )
    reports.sort(key=lambda item: item["mtime"], reverse=True)
    return reports[:40]


def json_response(handler: SimpleHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class TrainDashboardHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        if parsed.path == "/" or parsed.path == "":
            return str(UI_DIR / "train_dashboard.html")
        if parsed.path.startswith("/ui/"):
            return str(UI_DIR / parsed.path.removeprefix("/ui/"))
        return super().translate_path(path)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/defaults":
            json_response(
                self,
                {
                    "defaults": default_payload(),
                    "presets": {
                        "single": single_preset(),
                        "tp2": tp2_preset(),
                    },
                },
            )
            return
        if parsed.path == "/api/runs":
            json_response(self, {"runs": run_list_payload()})
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.rsplit("/", 1)[-1]
            payload = run_detail_payload(run_id)
            if payload is None:
                json_response(self, {"error": "run not found"}, status=404)
                return
            json_response(self, payload)
            return
        if parsed.path == "/api/reports":
            json_response(self, {"reports": existing_report_list()})
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            json_response(self, {"error": "not found"}, status=404)
            return
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b"{}"
        payload = json.loads(body.decode("utf-8"))
        record = start_run(payload)
        json_response(self, record.to_dict(), status=HTTPStatus.ACCEPTED)


def main() -> None:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), TrainDashboardHandler)
    print(f"Train dashboard listening on http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
