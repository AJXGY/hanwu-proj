from __future__ import annotations

import json
import mimetypes
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
DEFAULT_HOST = os.environ.get("MODEL_DASHBOARD_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MODEL_DASHBOARD_PORT", "8253"))
DEFAULT_IMAGE = "cambricon-base/pytorch:v25.10.0-torch2.7.1-torchmlu1.29.1-ubuntu22.04-py310"
DEFAULT_MODEL_DIR = "/home/o_mabin/LLM/models/Llama-3.1-8B"
INFER_REPO = REPO_ROOT / "projects" / "inference" / "time-modeling"
TRAIN_REPO = REPO_ROOT / "projects" / "training" / "time-modeling"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "models"


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
        payload["artifacts"] = list_artifacts(self.output_dir)
        return payload


RUNS: dict[str, RunRecord] = {}
RUNS_LOCK = threading.Lock()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def list_artifacts(output_dir: str) -> list[dict[str, str]]:
    if not output_dir:
        return []
    root = Path(output_dir)
    if not root.exists():
        return []
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".svg", ".html", ".md", ".txt", ".log"}:
            rel = path.relative_to(root).as_posix()
            artifacts.append(
                {
                    "name": rel,
                    "path": rel,
                    "kind": path.suffix.lower().lstrip("."),
                }
            )
    return artifacts


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
            record.finished_at = time.time()
    elif record.status == "running":
        record.status = "failed"
        if record.finished_at is None:
            record.finished_at = time.time()


def default_payload() -> dict[str, Any]:
    return {
        "task": "inference",
        "scale": "single",
        "image": DEFAULT_IMAGE,
        "host_model_dir": DEFAULT_MODEL_DIR,
        "prompt": "alpha alpha alpha alpha alpha alpha alpha alpha",
        "dtype": "bf16",
        "physical_devices": "0",
        "world_size": 1,
        "tp_size": 1,
        "microbatch_count": 2,
        "microbatch_size": 1,
        "sequence_length": 16,
        "output_root": str(DEFAULT_OUTPUT_ROOT),
    }


def inference_single_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update({"task": "inference", "scale": "single", "physical_devices": "0", "world_size": 1, "tp_size": 1})
    return payload


def inference_dual_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update({"task": "inference", "scale": "dual", "physical_devices": "0,1", "world_size": 2, "tp_size": 2})
    return payload


def training_single_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update({"task": "training", "scale": "single", "physical_devices": "0", "world_size": 1, "tp_size": 1, "microbatch_count": 2})
    return payload


def training_dual_preset() -> dict[str, Any]:
    payload = default_payload()
    payload.update({"task": "training", "scale": "dual", "physical_devices": "0,1", "world_size": 2, "tp_size": 2, "microbatch_count": 2})
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
    payload["world_size"] = int(payload["world_size"])
    payload["tp_size"] = int(payload["tp_size"])
    payload["microbatch_count"] = int(payload["microbatch_count"])
    payload["microbatch_size"] = int(payload["microbatch_size"])
    payload["sequence_length"] = int(payload["sequence_length"])
    payload["output_root"] = str(payload["output_root"]).strip()
    return payload


def host_output_dir(run_id: str, payload: dict[str, Any]) -> Path:
    output_root = Path(payload["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / payload["task"]
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir = task_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_command(run_id: str, payload: dict[str, Any]) -> tuple[list[str], Path]:
    output_dir = host_output_dir(run_id, payload)
    workspace_output_dir = "/workspace/" + str(output_dir.relative_to(ROOT)).replace("\\", "/")
    if payload["task"] == "inference":
        python_command = [
            "python3",
            "/workspace/infer_model_builder.py",
            "--infer-repo",
            "/deps/infer",
            "--model-path",
            "/model",
            "--prompt",
            payload["prompt"],
            "--dtype",
            payload["dtype"],
            "--device",
            "mlu:0",
            "--parallel-mode",
            "tp" if payload["scale"] == "dual" else "single",
            "--physical-devices",
            payload["physical_devices"],
            "--world-size",
            str(payload["world_size"]),
            "--tp-size",
            str(payload["tp_size"]),
            "--microbatch-count",
            str(payload["microbatch_count"]),
            "--output-dir",
            workspace_output_dir,
        ]
    else:
        python_command = [
            "python3",
            "/workspace/train_model_builder.py",
            "--train-repo",
            "/deps/train",
            "--model-path",
            "/model",
            "--parallel-mode",
            "tp" if payload["scale"] == "dual" else "single",
            "--world-size",
            str(payload["world_size"]),
            "--tp-size",
            str(payload["tp_size"]),
            "--microbatch-count",
            str(payload["microbatch_count"]),
            "--microbatch-size",
            str(payload["microbatch_size"]),
            "--sequence-length",
            str(payload["sequence_length"]),
            "--physical-devices",
            payload["physical_devices"],
            "--output-dir",
            workspace_output_dir,
        ]
    inner = " && ".join(
        [
            "source /torch/venv3/pytorch/bin/activate",
            shell_join(python_command),
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
        task = output_dir.parent.name
        payload = default_payload()
        payload["task"] = task
        payload["scale"] = "dual" if len(summary.get("physical_devices", [])) >= 2 else "single"
        record = RunRecord(
            run_id=run_id,
            payload=payload,
            status="completed" if summary.get("success") else "failed",
            created_at=summary_path.stat().st_mtime,
            finished_at=summary_path.stat().st_mtime,
            returncode=0 if summary.get("success") else 1,
            output_dir=str(output_dir),
            summary=summary,
        )
        with RUNS_LOCK:
            RUNS[run_id] = record


def run_job(record: RunRecord) -> None:
    try:
        command, output_dir = build_command(record.run_id, record.payload)
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
            record.error = "Model structure build failed."
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

    def serve_artifact(self, record: RunRecord, relative_path: str) -> None:
        artifact_root = Path(record.output_dir).resolve()
        target = (artifact_root / relative_path).resolve()
        if artifact_root not in target.parents and artifact_root != target:
            self.write_json({"error": "Invalid artifact path"}, status=400)
            return
        if not target.exists() or not target.is_file():
            self.write_json({"error": "Artifact not found"}, status=404)
            return
        content = target.read_bytes()
        mime_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/defaults":
            self.write_json(
                {
                    "defaults": default_payload(),
                    "presets": {
                        "infer_single": inference_single_preset(),
                        "infer_dual": inference_dual_preset(),
                        "train_single": training_single_preset(),
                        "train_dual": training_dual_preset(),
                    },
                }
            )
            return
        if parsed.path == "/api/runs":
            with RUNS_LOCK:
                runs = [
                    item.to_dict()
                    for item in sorted(RUNS.values(), key=lambda record: record.created_at, reverse=True)
                ]
            self.write_json({"runs": runs})
            return
        if parsed.path.startswith("/api/runs/"):
            parts = parsed.path.split("/")
            if len(parts) >= 5 and parts[4] == "artifacts":
                run_id = parts[3]
                relative_path = "/".join(parts[5:])
                with RUNS_LOCK:
                    record = RUNS.get(run_id)
                if record is None:
                    self.write_json({"error": "Run not found"}, status=404)
                    return
                self.serve_artifact(record, relative_path)
                return
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
    print(f"Model-structure dashboard listening on http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
