from __future__ import annotations

import json
import mimetypes
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
DEFAULT_LOCAL_PYTHON = Path("/home/o_zhanghui/miniconda3/envs/llama_4gpu/bin/python")
DEFAULT_DASHBOARD_CONFIG = ROOT / "configs" / "dashboard_env.json"
DEFAULT_OUTPUT_ROOT = ROOT / "dashboard_runs"
DOCUMENTATION_SUFFIXES = {".md", ".rst", ".txt"}
DOCUMENTATION_DIRS = {"docs", "notes"}
ACCELERATOR_VISIBLE_DEVICE_ENVS = {
    "cuda": ["CUDA_VISIBLE_DEVICES"],
    "mlu": [],
}


DASHBOARD_SETTINGS: dict[str, Any] | None = None
ENVIRONMENT_THREAD: threading.Thread | None = None
ENVIRONMENT_LOCK = threading.Lock()
ENVIRONMENT_STATE: dict[str, Any] = {
    "status": "unprepared",
    "ready": False,
    "runner": "",
    "config_path": str(DEFAULT_DASHBOARD_CONFIG),
    "container_name": None,
    "prepared_at": None,
    "last_error": None,
    "details": {},
}


def default_environment_config() -> dict[str, Any]:
    return {
        "runner": "docker_run_image",
        "image_name": "ubuntu2204-torch26-py:latest",
        "python_bin": default_local_python_bin(),
        "container_name": "mvp-dashboard-env",
        "gpu_binding": "all",
        "docker_run_args": [],
        "network_mode": "host",
        "ipc_mode": "host",
        "project_mounts": [str(ROOT.parent.resolve()), "/tmp"],
        "docker_env": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        "auto_prepare": True,
    }


def load_dashboard_config(config_path: str | Path | None = None) -> dict[str, Any]:
    target = Path(
        config_path
        or os.environ.get("MVP_DASHBOARD_CONFIG")
        or DEFAULT_DASHBOARD_CONFIG
    ).expanduser()
    raw_config: dict[str, Any] = {}
    if target.exists():
        raw_config = json.loads(target.read_text(encoding="utf-8"))

    environment = default_environment_config()
    raw_environment = raw_config.get("environment") or {}
    if isinstance(raw_environment, dict):
        environment.update(
            {
                key: value
                for key, value in raw_environment.items()
                if key != "docker_env"
            }
        )
        docker_env = dict(default_environment_config()["docker_env"])
        docker_env.update(raw_environment.get("docker_env") or {})
        environment["docker_env"] = docker_env
    docker_run_args = raw_environment.get("docker_run_args")
    if isinstance(docker_run_args, list):
        environment["docker_run_args"] = [str(item) for item in docker_run_args]
    environment["python_bin"] = str(
        environment.get("python_bin") or default_local_python_bin()
    )
    environment["runner"] = str(environment.get("runner") or "docker_run_image")
    environment["image_name"] = str(
        environment.get("image_name") or "ubuntu2204-torch26-py:latest"
    )
    environment["container_name"] = str(
        environment.get("container_name") or "mvp-dashboard-env"
    )
    environment["accelerator_kind"] = str(
        environment.get("accelerator_kind") or detect_local_accelerator_kind()
    )
    environment["project_mounts"] = [
        str(Path(path).expanduser())
        for path in environment.get("project_mounts", [])
        if str(path).strip()
    ]

    defaults = default_request()
    raw_defaults = raw_config.get("request_defaults") or {}
    if isinstance(raw_defaults, dict):
        defaults.update(raw_defaults)
    defaults["runner"] = environment["runner"]
    defaults["image_name"] = environment["image_name"]
    defaults["python_bin"] = environment["python_bin"]
    defaults = normalize_request_payload(defaults, managed_environment=environment)
    env_output_root = os.environ.get("MVP_DASHBOARD_OUTPUT_ROOT", "").strip()
    output_root = (
        Path(env_output_root).expanduser().resolve()
        if env_output_root
        else DEFAULT_OUTPUT_ROOT.resolve()
    )

    return {
        "config_path": str(target.resolve()),
        "environment": environment,
        "defaults": defaults,
        "output_root": str(output_root),
    }


def configure_dashboard_settings(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    global DASHBOARD_SETTINGS
    DASHBOARD_SETTINGS = load_dashboard_config(config_path)
    detected = detect_environment_state(DASHBOARD_SETTINGS["environment"])
    with ENVIRONMENT_LOCK:
        ENVIRONMENT_STATE.update(
            {
                "status": detected["status"],
                "ready": detected["ready"],
                "runner": DASHBOARD_SETTINGS["environment"]["runner"],
                "config_path": DASHBOARD_SETTINGS["config_path"],
                "container_name": DASHBOARD_SETTINGS["environment"].get(
                    "container_name"
                ),
                "prepared_at": detected["prepared_at"],
                "last_error": detected["last_error"],
                "details": detected["details"],
            }
        )
    load_saved_runs()
    return DASHBOARD_SETTINGS


def dashboard_settings() -> dict[str, Any]:
    return DASHBOARD_SETTINGS or configure_dashboard_settings()


def dashboard_environment_config() -> dict[str, Any]:
    return dict(dashboard_settings()["environment"])


def dashboard_request_defaults() -> dict[str, Any]:
    return dict(dashboard_settings()["defaults"])


def detect_environment_state(environment: dict[str, Any]) -> dict[str, Any]:
    details = {
        "auto_prepare": bool(environment.get("auto_prepare", True)),
        "project_mounts": list(environment.get("project_mounts", [])),
    }
    if environment["runner"] == "local_python":
        python_bin = str(environment.get("python_bin") or default_local_python_bin())
        ready = (
            Path(python_bin).exists()
            if Path(python_bin).is_absolute()
            else shutil.which(python_bin) is not None
        )
        details["python_bin"] = python_bin
        return {
            "status": "ready" if ready else "failed",
            "ready": ready,
            "prepared_at": None,
            "last_error": None if ready else f"Python not found: {python_bin}",
            "details": details,
        }

    container_name = str(environment.get("container_name") or "")
    details["container_name"] = container_name
    details["image_name"] = environment.get("image_name")
    details["gpu_binding"] = environment.get("gpu_binding", "all")
    if container_name and docker_container_running(container_name):
        return {
            "status": "ready",
            "ready": True,
            "prepared_at": None,
            "last_error": None,
            "details": details,
        }
    if container_name and docker_container_exists(container_name):
        return {
            "status": "stopped",
            "ready": False,
            "prepared_at": None,
            "last_error": None,
            "details": details,
        }
    return {
        "status": "unprepared",
        "ready": False,
        "prepared_at": None,
        "last_error": None,
        "details": details,
    }


def documentation_path(path_text: str) -> bool:
    path = Path(path_text)
    if path.name in {"AGENTS.md", "CLAUDE.md"}:
        return True
    if any(part in DOCUMENTATION_DIRS for part in path.parts):
        return True
    return path.suffix.lower() in DOCUMENTATION_SUFFIXES


def git_output_context() -> dict[str, Any]:
    env_output_root = os.environ.get("MVP_DASHBOARD_OUTPUT_ROOT", "").strip()
    configured_root = (
        Path(env_output_root).expanduser()
        if env_output_root
        else DEFAULT_OUTPUT_ROOT
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    commit_short = commit.stdout.strip() if commit.returncode == 0 else ""
    dirty_code_paths: list[str] = []
    if status.returncode == 0:
        for line in status.stdout.splitlines():
            entry = line[3:].strip()
            if not entry:
                continue
            candidates = [part.strip() for part in entry.split("->")]
            if any(not documentation_path(candidate) for candidate in candidates):
                dirty_code_paths.extend(candidates)
    code_clean = bool(commit_short) and not dirty_code_paths
    if code_clean and commit_short:
        base_dir = configured_root / commit_short
    else:
        base_dir = configured_root / "dirty_worktree"
    return {
        "commit_short": commit_short or None,
        "code_clean": code_clean,
        "dirty_code_paths": dirty_code_paths,
        "base_dir": str(base_dir),
    }


def resolve_run_output_dir(
    run_id: str, created_at: float
) -> tuple[Path, dict[str, Any]]:
    output_context = git_output_context()
    base_dir = Path(output_context["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(created_at))
    output_dir = base_dir / f"{timestamp}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, output_context


def environment_state_payload() -> dict[str, Any]:
    with ENVIRONMENT_LOCK:
        return {
            "status": ENVIRONMENT_STATE["status"],
            "ready": ENVIRONMENT_STATE["ready"],
            "runner": ENVIRONMENT_STATE["runner"],
            "config_path": ENVIRONMENT_STATE["config_path"],
            "container_name": ENVIRONMENT_STATE["container_name"],
            "prepared_at": ENVIRONMENT_STATE["prepared_at"],
            "last_error": ENVIRONMENT_STATE["last_error"],
            "details": dict(ENVIRONMENT_STATE["details"]),
        }


def set_environment_state(status: str, **updates: Any) -> dict[str, Any]:
    with ENVIRONMENT_LOCK:
        ENVIRONMENT_STATE["status"] = status
        ENVIRONMENT_STATE["ready"] = status == "ready"
        ENVIRONMENT_STATE.update(updates)
        return {
            "status": ENVIRONMENT_STATE["status"],
            "ready": ENVIRONMENT_STATE["ready"],
            "runner": ENVIRONMENT_STATE["runner"],
            "config_path": ENVIRONMENT_STATE["config_path"],
            "container_name": ENVIRONMENT_STATE["container_name"],
            "prepared_at": ENVIRONMENT_STATE["prepared_at"],
            "last_error": ENVIRONMENT_STATE["last_error"],
            "details": dict(ENVIRONMENT_STATE["details"]),
        }


def docker_mount_args_for_environment(environment: dict[str, Any]) -> list[str]:
    mounts = [(str(ROOT.resolve()), "/workspace")]
    seen = {mounts[0]}
    for raw_path in environment.get("project_mounts", []):
        path = Path(str(raw_path)).expanduser()
        if not path.exists():
            continue
        pair = (str(path.resolve()), str(path.resolve()))
        if pair in seen:
            continue
        mounts.append(pair)
        seen.add(pair)
    args: list[str] = []
    for host_path, container_path in mounts:
        args.extend(["-v", f"{host_path}:{container_path}"])
    return args


def docker_mount_args_for_remote_environment(environment: dict[str, Any]) -> list[str]:
    mounts = [(str(ROOT.resolve()), "/workspace")]
    seen = {mounts[0]}
    for raw_path in environment.get("project_mounts", []):
        text = str(raw_path).strip()
        if not text:
            continue
        expanded = str(Path(text).expanduser())
        pair = (expanded, expanded)
        if pair in seen:
            continue
        mounts.append(pair)
        seen.add(pair)
    args: list[str] = []
    for host_path, container_path in mounts:
        args.extend(["-v", f"{host_path}:{container_path}"])
    return args


def is_multi_host_request(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("parallel_mode", "single")) == "tp"
        and int(payload.get("nnodes", 1)) > 1
    )


def remote_host_from_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("remote_host", "") or "").strip()


def remote_ssh_port_from_payload(payload: dict[str, Any]) -> int:
    try:
        return int(payload.get("remote_ssh_port", 22) or 22)
    except (TypeError, ValueError):
        return 22


def multi_host_automation_enabled(payload: dict[str, Any]) -> bool:
    return is_multi_host_request(payload) and bool(remote_host_from_payload(payload))


def run_ssh_command(
    host: str, port: int, command: str, check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            host,
            "bash",
            "-lc",
            command,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=check,
    )


def ensure_remote_multi_host_supported(payload: dict[str, Any]) -> None:
    if not is_multi_host_request(payload):
        return
    if int(payload.get("node_rank", 0)) != 0:
        raise ValueError("Dashboard multi-host launcher expects local node_rank=0")
    if int(payload.get("nnodes", 1)) != 2:
        raise ValueError(
            "Dashboard multi-host launcher currently supports exactly 2 nodes"
        )
    runner = str(payload.get("runner", "") or "")
    if runner not in {"docker_run_image", "local_python"}:
        raise ValueError(f"Unsupported runner for multi-host automation: {runner}")
    if not remote_host_from_payload(payload):
        raise ValueError(
            "Multi-host automation requires remote_host (SSH target for node_rank=1)"
        )


def remote_container_name_for_payload(
    payload: dict[str, Any], environment: dict[str, Any]
) -> str:
    configured = str(payload.get("remote_container_name", "") or "").strip()
    if configured:
        return configured
    base = str(environment.get("container_name") or "mvp-dashboard-env")
    return f"{base}-remote"


def prepare_remote_docker_environment(
    environment: dict[str, Any], payload: dict[str, Any], force: bool = False
) -> dict[str, Any]:
    ensure_remote_multi_host_supported(payload)
    host = remote_host_from_payload(payload)
    port = remote_ssh_port_from_payload(payload)
    container_name = remote_container_name_for_payload(payload, environment)
    remote_devices = str(payload.get("remote_physical_devices", "") or "").strip()
    gpu_binding = (
        f"device={remote_devices}"
        if remote_devices
        else str(environment.get("gpu_binding", "all"))
    )

    if force:
        rm_force = " ".join(
            shell_quote(part) for part in ["docker", "rm", "-f", container_name]
        )
        run_ssh_command(host, port, f"{rm_force} >/dev/null 2>&1 || true", check=False)

    check_running = " ".join(
        shell_quote(part)
        for part in ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
    )
    running = run_ssh_command(host, port, check_running, check=False)
    if running.returncode == 0 and running.stdout.strip() == "true":
        return {
            "host": host,
            "ssh_port": port,
            "runner": "docker_run_image",
            "container_name": container_name,
            "image_name": environment["image_name"],
            "gpu_binding": gpu_binding,
        }

    check_exists = " ".join(
        shell_quote(part) for part in ["docker", "inspect", container_name]
    )
    exists = run_ssh_command(host, port, check_exists, check=False)
    if exists.returncode == 0:
        start_command = " ".join(
            shell_quote(part) for part in ["docker", "start", container_name]
        )
        run_ssh_command(host, port, start_command, check=True)
    else:
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
        ]
        if gpu_binding:
            command.extend(["--gpus", gpu_binding])
        command.extend(str(item) for item in environment.get("docker_run_args", []))
        if environment.get("network_mode"):
            command.extend(["--network", str(environment["network_mode"])])
        if environment.get("ipc_mode"):
            command.extend(["--ipc", str(environment["ipc_mode"])])
        command.extend(["-w", "/workspace"])
        for key, value in sorted((environment.get("docker_env") or {}).items()):
            command.extend(["-e", f"{key}={value}"])
        command.extend(docker_mount_args_for_remote_environment(environment))
        command.extend(
            [
                environment["image_name"],
                "bash",
                "-lc",
                "trap 'exit 0' TERM INT; while true; do sleep 3600; done",
            ]
        )
        remote_command = " ".join(shell_quote(part) for part in command)
        run_ssh_command(host, port, remote_command, check=True)

    verify = " ".join(
        shell_quote(part)
        for part in [
            "docker",
            "exec",
            container_name,
            "bash",
            "-lc",
            "python3 --version",
        ]
    )
    run_ssh_command(host, port, verify, check=True)
    return {
        "host": host,
        "ssh_port": port,
        "runner": "docker_run_image",
        "container_name": container_name,
        "image_name": environment["image_name"],
        "gpu_binding": gpu_binding,
    }


def prepare_remote_local_python_environment(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_remote_multi_host_supported(payload)
    host = remote_host_from_payload(payload)
    port = remote_ssh_port_from_payload(payload)
    python_bin = str(
        payload.get("remote_python_bin") or payload.get("python_bin") or ""
    ).strip()
    if not python_bin:
        python_bin = default_local_python_bin()
    verify_command = " && ".join(
        [
            " ".join(
                shell_quote(part)
                for part in ["test", "-f", str(ROOT / "torch_infer_mvp.py")]
            ),
            " ".join(shell_quote(part) for part in [python_bin, "--version"]),
        ]
    )
    completed = run_ssh_command(host, port, verify_command, check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "verification command failed"
        raise RuntimeError(
            f"Remote local_python verification failed on {host}:{port}: {detail}"
        )
    return {
        "host": host,
        "ssh_port": port,
        "runner": "local_python",
        "python_bin": python_bin,
    }


def prepare_remote_environment(
    environment: dict[str, Any], payload: dict[str, Any], force: bool = False
) -> dict[str, Any]:
    runner = str(payload.get("runner", "") or "")
    if runner == "docker_run_image":
        return prepare_remote_docker_environment(environment, payload, force=force)
    if runner == "local_python":
        return prepare_remote_local_python_environment(payload)
    raise ValueError(f"Unsupported runner for remote prepare: {runner}")


def docker_container_running(container_name: str) -> bool:
    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return inspect.returncode == 0 and inspect.stdout.strip() == "true"


def docker_container_exists(container_name: str) -> bool:
    inspect = subprocess.run(
        ["docker", "inspect", container_name],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return inspect.returncode == 0


def prepare_docker_environment(
    environment: dict[str, Any], force: bool = False
) -> dict[str, Any]:
    container_name = str(environment["container_name"])
    if force and docker_container_exists(container_name):
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    if docker_container_running(container_name):
        return {
            "container_name": container_name,
            "image_name": environment["image_name"],
            "gpu_binding": environment.get("gpu_binding", "all"),
        }

    if docker_container_exists(container_name):
        subprocess.run(
            ["docker", "start", container_name],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
        ]
        gpu_binding = str(environment.get("gpu_binding", "") or "").strip()
        if gpu_binding:
            command.extend(["--gpus", gpu_binding])
        command.extend(str(item) for item in environment.get("docker_run_args", []))
        if environment.get("network_mode"):
            command.extend(["--network", str(environment["network_mode"])])
        if environment.get("ipc_mode"):
            command.extend(["--ipc", str(environment["ipc_mode"])])
        command.extend(["-w", "/workspace"])
        for key, value in sorted((environment.get("docker_env") or {}).items()):
            command.extend(["-e", f"{key}={value}"])
        command.extend(docker_mount_args_for_environment(environment))
        command.extend(
            [
                environment["image_name"],
                "bash",
                "-lc",
                "trap 'exit 0' TERM INT; while true; do sleep 3600; done",
            ]
        )
        subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True, check=True
        )

    subprocess.run(
        ["docker", "exec", container_name, "bash", "-lc", "python3 --version"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "container_name": container_name,
        "image_name": environment["image_name"],
        "gpu_binding": environment.get("gpu_binding", "all"),
    }


def prepare_environment(
    force: bool = False, prepare_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    needs_remote_prepare = bool(
        prepare_payload and multi_host_automation_enabled(prepare_payload)
    )
    current = environment_state_payload()
    if current["ready"] and not force and not needs_remote_prepare:
        return current

    environment = dashboard_environment_config()
    set_environment_state(
        "preparing",
        runner=environment["runner"],
        config_path=dashboard_settings()["config_path"],
        container_name=environment.get("container_name"),
        prepared_at=None,
        last_error=None,
        details={
            "auto_prepare": bool(environment.get("auto_prepare", True)),
            "project_mounts": list(environment.get("project_mounts", [])),
        },
    )
    try:
        if environment["runner"] == "local_python":
            python_bin = str(
                environment.get("python_bin") or default_local_python_bin()
            )
            if Path(python_bin).is_absolute():
                if not Path(python_bin).exists():
                    raise FileNotFoundError(f"Python not found: {python_bin}")
            elif shutil.which(python_bin) is None:
                raise FileNotFoundError(f"Python not found in PATH: {python_bin}")
            details = {
                "python_bin": python_bin,
                "auto_prepare": bool(environment.get("auto_prepare", True)),
            }
            if prepare_payload and multi_host_automation_enabled(prepare_payload):
                ensure_remote_multi_host_supported(prepare_payload)
                remote = prepare_remote_environment(
                    environment, prepare_payload, force=force
                )
                details["remote"] = remote
            return set_environment_state(
                "ready",
                prepared_at=time.time(),
                last_error=None,
                details=details,
            )

        details = prepare_docker_environment(environment, force=force)
        details["project_mounts"] = list(environment.get("project_mounts", []))
        details["auto_prepare"] = bool(environment.get("auto_prepare", True))
        if prepare_payload and multi_host_automation_enabled(prepare_payload):
            remote = prepare_remote_environment(
                environment, prepare_payload, force=force
            )
            details["remote"] = remote
        return set_environment_state(
            "ready",
            prepared_at=time.time(),
            last_error=None,
            details=details,
        )
    except Exception as exc:  # pragma: no cover
        return set_environment_state("failed", prepared_at=None, last_error=str(exc))


def prepare_environment_async(
    force: bool = False, prepare_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    global ENVIRONMENT_THREAD
    needs_remote_prepare = bool(
        prepare_payload and multi_host_automation_enabled(prepare_payload)
    )
    current = environment_state_payload()
    if current["status"] == "preparing":
        return current
    if current["ready"] and not force and not needs_remote_prepare:
        return current
    environment = dashboard_environment_config()
    set_environment_state(
        "preparing",
        runner=environment["runner"],
        config_path=dashboard_settings()["config_path"],
        container_name=environment.get("container_name"),
        prepared_at=None,
        last_error=None,
    )
    thread = threading.Thread(
        target=prepare_environment,
        kwargs={"force": force, "prepare_payload": prepare_payload},
        daemon=True,
    )
    ENVIRONMENT_THREAD = thread
    thread.start()
    return environment_state_payload()


def stop_environment() -> dict[str, Any]:
    environment = dashboard_environment_config()
    if environment["runner"] == "docker_run_image":
        subprocess.run(
            ["docker", "rm", "-f", str(environment["container_name"])],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    return set_environment_state("stopped", prepared_at=None, last_error=None)


@dataclass
class RunRecord:
    run_id: str
    request: dict[str, Any]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    command: list[str] = field(default_factory=list)
    graph_command: list[str] = field(default_factory=list)
    output_dir: str = ""
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    report: dict[str, Any] | None = None
    error: str | None = None
    graph: dict[str, Any] | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    output_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        reconcile_run_artifacts(self)
        return {
            "run_id": self.run_id,
            "request": self.request,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "command": self.command,
            "graph_command": self.graph_command,
            "output_dir": self.output_dir,
            "stdout": self.stdout[-12000:],
            "stderr": self.stderr[-12000:],
            "return_code": self.return_code,
            "report": self.report,
            "error": self.error,
            "graph": self.graph,
            "timings": self.timings,
            "output_policy": self.output_policy,
        }


RUNS: dict[str, RunRecord] = {}
RUNS_LOCK = threading.Lock()


def reconcile_run_artifacts(record: RunRecord) -> None:
    if not record.output_dir:
        return
    output_dir = Path(record.output_dir)
    if not output_dir.exists():
        return

    report_path = output_dir / "report.json"
    if record.report is None and report_path.exists():
        try:
            record.report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    graph_dir = output_dir / "graph_viz"
    summary_path = graph_dir / "summary.json"
    if record.graph is None and summary_path.exists():
        try:
            record.graph = graph_payload(record, graph_dir)
        except Exception:
            pass

    if record.graph is not None and record.status in {
        "queued",
        "running",
        "estimate_ready",
        "measurement_ready",
        "rendering_graph",
        "failed",
    }:
        record.status = "completed"
        if record.finished_at is None:
            record.finished_at = time.time()


def infer_run_id_from_output_dir(output_dir: Path) -> str:
    suffix = output_dir.name.rsplit("_", 1)[-1]
    if len(suffix) == 8:
        return suffix
    return output_dir.name


def request_from_saved_report(report: dict[str, Any]) -> dict[str, Any]:
    model = report.get("model") or {}
    execution = report.get("execution") or {}
    calibration = report.get("calibration") or {}
    accelerator_kind = str(calibration.get("accelerator_kind") or "cuda")
    physical_devices = execution.get("physical_devices") or []
    return {
        "runner": dashboard_environment_config().get("runner", "docker_run_image"),
        "image_name": dashboard_environment_config().get("image_name", ""),
        "python_bin": dashboard_environment_config().get("python_bin", "python3"),
        "model_path": model.get("path", ""),
        "prompt": model.get("prompt", ""),
        "max_new_tokens": int(model.get("max_new_tokens") or 1),
        "dtype": model.get("dtype", "bf16"),
        "parallel_mode": execution.get("parallel_mode", "single"),
        "physical_devices": ",".join(str(item) for item in physical_devices),
        "world_size": int(execution.get("world_size") or 1),
        "tp_size": int(execution.get("tp_size") or 1),
        "nnodes": int(execution.get("nnodes") or 1),
        "nproc_per_node": int(
            execution.get("nproc_per_node") or execution.get("world_size") or 1
        ),
        "interconnect": execution.get("interconnect", "ethernet"),
        "collective_bandwidth_gbps": execution.get("collective_bandwidth_gbps"),
        "collective_latency_ms": execution.get("collective_latency_ms"),
        "device": default_device_for_accelerator(accelerator_kind),
        "estimate_only": False,
        "estimate_mode": report.get("module_profile_meta", {})
        .get("estimate_mode_effective", "hybrid"),
        "table_db_path": report.get("module_profile_meta", {})
        .get("table_db_path", "database/module_profile_table_cambricon_mlu580.jsonl"),
        "table_writeback": True,
        "warmup": 1,
        "benchmark_repeat": 1,
        "profile_repeat": 1,
        "generate_graph_viz": True,
    }


def load_saved_runs() -> None:
    output_root = Path(dashboard_settings()["output_root"])
    if not output_root.exists():
        return
    for report_path in sorted(output_root.rglob("report.json")):
        output_dir = report_path.parent
        run_id = infer_run_id_from_output_dir(output_dir)
        with RUNS_LOCK:
            if run_id in RUNS:
                continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        status_path = output_dir / "dashboard_status.json"
        timings: dict[str, Any] = {}
        status = "completed" if (output_dir / "graph_viz" / "summary.json").exists() else "measurement_ready"
        if status_path.exists():
            try:
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
                timings.update(status_payload.get("timings") or {})
                stage = status_payload.get("stage")
                if stage == "estimation_ready":
                    status = "estimate_ready"
                elif stage == "measurement_ready":
                    status = "measurement_ready"
            except json.JSONDecodeError:
                pass
        record = RunRecord(
            run_id=run_id,
            request=request_from_saved_report(report),
            status=status,
            created_at=report_path.stat().st_mtime,
            started_at=None,
            finished_at=report_path.stat().st_mtime,
            output_dir=str(output_dir),
            report=report,
            timings=timings,
        )
        reconcile_run_artifacts(record)
        with RUNS_LOCK:
            RUNS[run_id] = record


def local_model_paths() -> list[str]:
    candidate_dirs = [
        ROOT / "downloads",
        ROOT / "smoke_models",
        ROOT.parent / "0320proj" / "downloads",
    ]
    models: list[str] = []
    seen: set[str] = set()
    for downloads_dir in candidate_dirs:
        if not downloads_dir.exists():
            continue
        for child in sorted(downloads_dir.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                resolved = str(child.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    models.append(resolved)
    return models


def parse_device_csv(raw_value: Any) -> list[int]:
    devices: list[int] = []
    seen: set[int] = set()
    for part in str(raw_value or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            device = int(text)
        except ValueError:
            continue
        if device in seen:
            continue
        seen.add(device)
        devices.append(device)
    return devices


def local_gpu_inventory() -> list[dict[str, Any]]:
    kind = detect_local_accelerator_kind()
    if kind == "mlu":
        return detect_cnmon_inventory()
    return detect_nvidia_inventory()


def default_single_host_devices() -> str:
    gpu_indices = [gpu["index"] for gpu in local_gpu_inventory()]
    if len(gpu_indices) >= 4 and 2 in gpu_indices and 3 in gpu_indices:
        return "2,3"
    if len(gpu_indices) >= 2:
        return ",".join(str(device) for device in gpu_indices[:2])
    if gpu_indices:
        return str(gpu_indices[0])
    return "0"


def default_local_python_bin() -> str:
    try:
        if DEFAULT_LOCAL_PYTHON.exists():
            return str(DEFAULT_LOCAL_PYTHON)
    except PermissionError:
        pass
    return "python3"


def accelerator_label(kind: str) -> str:
    return "MLU" if kind == "mlu" else "GPU"


def default_device_for_accelerator(kind: str) -> str:
    return f"{kind}:0"


def payload_accelerator_kind(payload: dict[str, Any] | None = None) -> str:
    device = str((payload or {}).get("device", "") or "").strip().lower()
    if device.startswith("mlu"):
        return "mlu"
    if device.startswith("cuda"):
        return "cuda"
    environment = dashboard_environment_config()
    configured = str(environment.get("accelerator_kind", "") or "").strip().lower()
    if configured in {"cuda", "mlu"}:
        return configured
    return detect_local_accelerator_kind()


def detect_cnmon_inventory() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["cnmon", "info"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in completed.stdout.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("Card "):
            if current is not None:
                devices.append(current)
            index_text = stripped.split(" ", 1)[1].strip()
            try:
                index = int(index_text)
            except ValueError:
                index = len(devices)
            current = {"index": index, "name": f"MLU {index}"}
            continue
        if current is None or ":" not in line:
            continue
        key, _, value = stripped.partition(":")
        if key.strip() == "Product Name":
            name = value.strip()
            if name:
                current["name"] = name
    if current is not None:
        devices.append(current)
    return devices


def detect_local_accelerator_kind() -> str:
    if detect_cnmon_inventory():
        return "mlu"
    if detect_nvidia_inventory():
        return "cuda"
    return "cuda"


def detect_nvidia_inventory() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        index_text, _, name = text.partition(",")
        try:
            index = int(index_text.strip())
        except ValueError:
            continue
        gpus.append({"index": index, "name": name.strip() or f"GPU {index}"})
    return gpus


def script_python_bin(payload: dict[str, Any]) -> str:
    if payload.get("runner") == "local_python":
        return str(payload.get("python_bin") or default_local_python_bin())
    return "python3"


def default_request() -> dict[str, Any]:
    accelerator_kind = detect_local_accelerator_kind()
    models = local_model_paths()
    preferred = next(
        (path for path in models if path.endswith("Meta-Llama-3.1-8B")),
        models[0] if models else str(ROOT / "downloads" / "Meta-Llama-3.1-8B"),
    )
    prompt = " ".join(["alpha"] * 64)
    physical_devices = default_single_host_devices()
    device_count = max(len(parse_device_csv(physical_devices)), 1)
    return {
        "runner": "docker_run_image",
        "image_name": "ubuntu2204-torch26-py:latest",
        "python_bin": default_local_python_bin(),
        "model_path": preferred,
        "prompt": prompt,
        "max_new_tokens": 12,
        "dtype": "bf16",
        "parallel_mode": "tp" if device_count > 1 else "single",
        "physical_devices": physical_devices,
        "world_size": device_count,
        "tp_size": device_count,
        "nnodes": 1,
        "nproc_per_node": device_count,
        "node_rank": 0,
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "remote_host": "",
        "remote_ssh_port": 22,
        "remote_physical_devices": "",
        "remote_python_bin": "",
        "remote_model_path": "",
        "remote_container_name": "",
        "interconnect": "ethernet",
        "collective_bandwidth_gbps": "",
        "collective_latency_ms": "",
        "dist_timeout_minutes": 30,
        "device": default_device_for_accelerator(accelerator_kind),
        "estimate_only": False,
        "estimate_mode": "online",
        "table_db_path": "database/module_profile_table.jsonl",
        "table_writeback": False,
        "warmup": 2,
        "benchmark_repeat": 5,
        "profile_repeat": 10,
        "generate_graph_viz": True,
    }


def read_json(request_handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(request_handler.headers.get("Content-Length", "0"))
    raw = request_handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


def write_json(
    request_handler: SimpleHTTPRequestHandler,
    payload: dict[str, Any],
    status: int = 200,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    request_handler.send_response(status)
    request_handler.send_header("Content-Type", "application/json; charset=utf-8")
    request_handler.send_header("Content-Length", str(len(body)))
    request_handler.end_headers()
    request_handler.wfile.write(body)


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def workspace_path(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return str(path)
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(ROOT)
    except ValueError:
        return str(candidate)
    return str(Path("/workspace") / relative)


def model_mount_args(model_path: str) -> list[str]:
    resolved = str(Path(model_path).resolve())
    if resolved.startswith(str(ROOT.resolve())):
        return []
    return ["-v", f"{resolved}:{resolved}:ro"]


def resolved_table_db_path(table_db_path: str) -> Path:
    path = Path(table_db_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def table_db_mount_args(payload: dict[str, Any]) -> list[str]:
    if not payload.get("table_writeback"):
        return []
    db_parent = resolved_table_db_path(payload["table_db_path"]).parent
    container_parent = workspace_path(db_parent)
    return ["-v", f"{db_parent}:{container_parent}"]


def graph_cache_mount_args() -> list[str]:
    host_cache_dir = ROOT / ".graph_cache"
    container_cache_dir = "/workspace/.graph_cache"
    return ["-v", f"{host_cache_dir}:{container_cache_dir}"]


def build_script_command(
    script_path: Path,
    payload: dict[str, Any],
    output_dir: Path,
    include_parallel: bool = False,
) -> list[str]:
    command = [
        script_python_bin(payload),
        str(script_path),
        "--model-path",
        payload["model_path"],
        "--prompt",
        payload["prompt"],
        "--dtype",
        payload["dtype"],
        "--device",
        payload["device"],
        "--warmup",
        str(payload["warmup"]),
        "--profile-repeat",
        str(payload["profile_repeat"]),
        "--output-dir",
        str(output_dir),
    ]
    if include_parallel:
        command[0:2] = []
        command = [
            *(
                [
                    script_python_bin(payload),
                    "-m",
                    "torch.distributed.run",
                    *(
                        [
                            "--nnodes",
                            str(payload.get("nnodes", 1)),
                            "--nproc_per_node",
                            str(payload.get("nproc_per_node", payload["world_size"])),
                            "--node_rank",
                            str(payload.get("node_rank", 0)),
                            "--master_addr",
                            payload.get("master_addr", "127.0.0.1"),
                            "--master_port",
                            str(payload.get("master_port", 29500)),
                        ]
                        if int(payload.get("nnodes", 1)) > 1
                        else [
                            "--standalone",
                            "--nproc_per_node",
                            str(payload["world_size"]),
                        ]
                    ),
                    str(script_path),
                ]
                if payload.get("parallel_mode") == "tp"
                else [script_python_bin(payload), str(script_path)]
            ),
            "--model-path",
            payload["model_path"],
            "--prompt",
            payload["prompt"],
            "--dtype",
            payload["dtype"],
            "--device",
            payload.get("device", "cuda:0"),
            "--parallel-mode",
            payload.get("parallel_mode", "single"),
            "--physical-devices",
            payload.get("physical_devices", ""),
            "--world-size",
            str(payload.get("world_size", 1)),
            "--tp-size",
            str(payload.get("tp_size", 1)),
            "--nnodes",
            str(payload.get("nnodes", 1)),
            "--nproc-per-node",
            str(payload.get("nproc_per_node", 1)),
            "--node-rank",
            str(payload.get("node_rank", 0)),
            "--master-addr",
            payload.get("master_addr", "127.0.0.1"),
            "--master-port",
            str(payload.get("master_port", 29500)),
            "--interconnect",
            payload.get("interconnect", "ethernet"),
            "--dist-timeout-minutes",
            str(payload.get("dist_timeout_minutes", 30)),
            "--warmup",
            str(payload["warmup"]),
            "--benchmark-repeat",
            str(payload["benchmark_repeat"]),
            "--profile-repeat",
            str(payload["profile_repeat"]),
            "--output-dir",
            str(output_dir),
        ]
        if payload.get("collective_bandwidth_gbps") is not None:
            command.extend(
                [
                    "--collective-bandwidth-gbps",
                    str(payload["collective_bandwidth_gbps"]),
                ]
            )
        if payload.get("collective_latency_ms") is not None:
            command.extend(
                [
                    "--collective-latency-ms",
                    str(payload["collective_latency_ms"]),
                ]
            )
        if payload.get("estimate_mode"):
            command.extend(["--estimate-mode", str(payload["estimate_mode"])])
        if payload.get("table_db_path"):
            command.extend(
                [
                    "--table-db-path",
                    str(resolved_table_db_path(payload["table_db_path"])),
                ]
            )
        if payload.get("table_writeback"):
            command.append("--table-writeback")
        if payload.get("estimate_only"):
            command.append("--estimate-only")
    return command


def build_predictor_command(payload: dict[str, Any], output_dir: Path) -> list[str]:
    command = build_script_command(
        ROOT / "torch_infer_mvp.py", payload, output_dir, include_parallel=True
    )
    output_index = command.index("--output-dir")
    command[output_index:output_index] = [
        "--max-new-tokens",
        str(payload["max_new_tokens"]),
    ]
    return command


def build_graph_command(payload: dict[str, Any], output_dir: Path) -> list[str]:
    graph_payload = dict(payload)
    graph_payload["device"] = default_device_for_accelerator(
        payload_accelerator_kind(payload)
    )
    return build_script_command(
        ROOT / "export_graph_viz.py", graph_payload, output_dir, include_parallel=False
    )


def dockerized_command(
    command: list[str], container_output_dir: str | None = None
) -> list[str]:
    output_index = command.index("--output-dir")
    converted = []
    for index, part in enumerate(command):
        if container_output_dir is not None and index == output_index + 1:
            converted.append(container_output_dir)
        elif isinstance(part, str) and Path(part).expanduser().is_absolute():
            converted.append(workspace_path(part))
        else:
            converted.append(part)
    return converted


def docker_runtime_command(payload: dict[str, Any], inner_command: list[str]) -> str:
    remote_command = " ".join(shell_quote(part) for part in inner_command)
    enable_docker_mps = os.environ.get("MVP_DASHBOARD_DOCKER_MPS", "0") == "1"
    if requested_gpu_binding(payload) is None or not enable_docker_mps:
        return remote_command
    mps_dir = "/tmp/mvp_dashboard_mps"
    # Reused containers can inherit a stale MPS daemon from a previous task.
    # Restart it per run so each exec gets a clean CUDA client state.
    return " && ".join(
        [
            f"mkdir -p {shell_quote(mps_dir)}",
            f"export CUDA_MPS_PIPE_DIRECTORY={shell_quote(mps_dir)}",
            f"export CUDA_MPS_LOG_DIRECTORY={shell_quote(mps_dir)}",
            "echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true",
            "nvidia-cuda-mps-control -d >/dev/null 2>&1",
            remote_command,
        ]
    )


def requested_gpu_binding(payload: dict[str, Any]) -> str | None:
    if payload_accelerator_kind(payload) != "cuda":
        return None
    if payload.get("parallel_mode") == "tp":
        devices = str(payload.get("physical_devices", "")).strip()
        if not devices:
            return None
        # Docker expects the device selector to stay quoted for multi-GPU requests.
        return f'"device={devices}"'
    device = str(payload.get("device", "cuda:0"))
    if device.startswith("cuda:"):
        return f"device={device.split(':', 1)[1]}"
    return None


def command_env(payload: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    accelerator_kind = payload_accelerator_kind(payload)
    if payload.get("runner") == "local_python":
        for key in [
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "CONDA_EXE",
            "_CE_CONDA",
            "_CE_M",
            "CUDA_MPS_PIPE_DIRECTORY",
            "CUDA_MPS_LOG_DIRECTORY",
        ]:
            env.pop(key, None)
        python_path = Path(str(payload.get("python_bin") or default_local_python_bin()))
        if python_path.parent.name == "bin":
            env["VIRTUAL_ENV"] = str(python_path.parent.parent)
            env["PATH"] = f"{python_path.parent}:{env.get('PATH', '')}".rstrip(":")
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("CUDA_VISIBLE_DEVICES", None)
    devices = str(payload.get("physical_devices", "")).strip()
    if payload.get("runner") == "local_python":
        return env
    visible_envs = ACCELERATOR_VISIBLE_DEVICE_ENVS.get(accelerator_kind, [])
    if payload.get("parallel_mode") == "tp":
        if devices:
            for env_name in visible_envs:
                env[env_name] = devices
    elif devices:
        value = devices.split(",", 1)[0].strip()
        for env_name in visible_envs:
            env[env_name] = value
    return env


def docker_exec_env_args(payload: dict[str, Any]) -> list[str]:
    accelerator_kind = payload_accelerator_kind(payload)
    devices = str(payload.get("physical_devices", "")).strip()
    env_args: list[str] = []
    visible_envs = ACCELERATOR_VISIBLE_DEVICE_ENVS.get(accelerator_kind, [])
    if payload.get("parallel_mode") == "tp":
        if devices:
            for env_name in visible_envs:
                env_args.extend(["-e", f"{env_name}={devices}"])
    elif devices:
        value = devices.split(",", 1)[0].strip()
        for env_name in visible_envs:
            env_args.extend(["-e", f"{env_name}={value}"])
    return env_args


def local_runtime_command(
    payload: dict[str, Any], command: list[str], host_output_dir: Path
) -> list[str]:
    if not str(payload.get("device", "cuda:0")).startswith(
        "cuda"
    ) and not requested_gpu_binding(payload):
        return command
    mps_dir = host_output_dir / ".local_mps"
    remote_command = " ".join(shell_quote(part) for part in command)
    return [
        "bash",
        "-lc",
        " && ".join(
            [
                f"mkdir -p {shell_quote(str(mps_dir))}",
                f"export CUDA_MPS_PIPE_DIRECTORY={shell_quote(str(mps_dir))}",
                f"export CUDA_MPS_LOG_DIRECTORY={shell_quote(str(mps_dir))}",
                "nvidia-cuda-mps-control -d",
                "trap 'echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true' EXIT",
                remote_command,
            ]
        ),
    ]


def parse_optional_float(value: Any) -> float | None:
    text = str(value or "").strip().lower()
    if text in {"", "none", "null", "nan"}:
        return None
    return float(text)


def normalize_request_payload(
    payload: dict[str, Any], managed_environment: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = dict(payload)
    environment = managed_environment or dashboard_environment_config()
    payload["runner"] = str(
        environment.get("runner")
        or payload.get("runner", "docker_run_image")
        or "docker_run_image"
    )
    payload["python_bin"] = str(
        environment.get("python_bin")
        or payload.get("python_bin", default_local_python_bin())
        or default_local_python_bin()
    )
    payload["image_name"] = str(
        environment.get("image_name")
        or payload.get("image_name", "ubuntu2204-torch26-py:latest")
        or "ubuntu2204-torch26-py:latest"
    )
    payload["parallel_mode"] = payload.get("parallel_mode", "single")
    payload["world_size"] = int(payload.get("world_size", 1) or 1)
    payload["tp_size"] = int(payload.get("tp_size", 1) or 1)
    payload["nnodes"] = int(payload.get("nnodes", 1) or 1)
    payload["nproc_per_node"] = int(payload.get("nproc_per_node", 1) or 1)
    payload["node_rank"] = int(payload.get("node_rank", 0) or 0)
    payload["master_addr"] = str(payload.get("master_addr", "127.0.0.1"))
    payload["master_port"] = int(payload.get("master_port", 29500) or 29500)
    payload["remote_host"] = str(payload.get("remote_host", "") or "").strip()
    payload["remote_ssh_port"] = int(payload.get("remote_ssh_port", 22) or 22)
    payload["remote_container_name"] = str(
        payload.get("remote_container_name", "") or ""
    ).strip()
    payload["remote_python_bin"] = str(
        payload.get("remote_python_bin", "") or ""
    ).strip()
    payload["remote_model_path"] = str(
        payload.get("remote_model_path", "") or ""
    ).strip()
    payload["interconnect"] = str(payload.get("interconnect", "ethernet") or "ethernet")
    payload["dist_timeout_minutes"] = int(payload.get("dist_timeout_minutes", 30) or 30)
    payload["estimate_mode"] = str(payload.get("estimate_mode", "online") or "online")
    payload["table_db_path"] = str(
        payload.get("table_db_path", "database/module_profile_table.jsonl")
        or "database/module_profile_table.jsonl"
    )
    payload["warmup"] = int(payload.get("warmup", 2) or 2)
    payload["benchmark_repeat"] = int(payload.get("benchmark_repeat", 5) or 5)
    payload["profile_repeat"] = int(payload.get("profile_repeat", 10) or 10)
    normalized_devices = parse_device_csv(payload.get("physical_devices", ""))
    payload["physical_devices"] = ",".join(str(device) for device in normalized_devices)
    remote_devices = parse_device_csv(payload.get("remote_physical_devices", ""))
    payload["remote_physical_devices"] = ",".join(
        str(device) for device in remote_devices
    )
    payload["collective_bandwidth_gbps"] = parse_optional_float(
        payload.get("collective_bandwidth_gbps", "")
    )
    payload["collective_latency_ms"] = parse_optional_float(
        payload.get("collective_latency_ms", "")
    )
    if payload["runner"] == "local_python" and payload["python_bin"] in {
        "python3",
        "python",
    }:
        payload["python_bin"] = default_local_python_bin()
    if payload["parallel_mode"] != "tp":
        if not payload["physical_devices"]:
            payload["physical_devices"] = default_single_host_devices().split(",", 1)[0]
        payload["world_size"] = 1
        payload["tp_size"] = 1
        payload["nnodes"] = 1
        payload["nproc_per_node"] = 1
        payload["node_rank"] = 0
    else:
        if payload["nnodes"] > 1:
            payload["world_size"] = payload["nnodes"] * payload["nproc_per_node"]
            if payload["tp_size"] <= 1:
                payload["tp_size"] = payload["world_size"]
            if not payload["physical_devices"]:
                payload["physical_devices"] = "0"
            if not payload["remote_physical_devices"]:
                payload["remote_physical_devices"] = payload["physical_devices"]
            if (
                payload.get("runner") == "local_python"
                and not payload["remote_python_bin"]
            ):
                payload["remote_python_bin"] = payload["python_bin"]
        else:
            if not payload["physical_devices"]:
                payload["physical_devices"] = default_single_host_devices()
            single_host_devices = parse_device_csv(payload["physical_devices"])
            if not single_host_devices:
                single_host_devices = [0]
                payload["physical_devices"] = "0"
            local_device_count = len(single_host_devices)
            payload["world_size"] = local_device_count
            payload["tp_size"] = local_device_count
            payload["nnodes"] = 1
            payload["nproc_per_node"] = local_device_count
            payload["node_rank"] = 0
            payload["master_addr"] = "127.0.0.1"
    return payload


def run_local_command(
    record: RunRecord,
    payload: dict[str, Any],
    command: list[str],
    host_output_dir: Path,
    header: str,
) -> subprocess.CompletedProcess[str]:
    wrapped_command = local_runtime_command(payload, command, host_output_dir)
    completed = subprocess.run(
        wrapped_command,
        cwd=str(ROOT),
        env=command_env(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    append_process_output(record, completed, header)
    return completed


def run_docker_command(
    record: RunRecord,
    payload: dict[str, Any],
    command: list[str],
    host_output_dir: Path,
    header: str,
) -> subprocess.CompletedProcess[str]:
    environment = environment_state_payload()
    if not environment["ready"]:
        raise RuntimeError(
            "Managed environment is not ready; prepare it before running"
        )
    container_name = str(environment.get("container_name") or "")
    if not container_name:
        raise RuntimeError("Managed docker environment is missing a container name")
    host_output_dir.mkdir(parents=True, exist_ok=True)
    inner_command = dockerized_command(command)
    remote_command = docker_runtime_command(payload, inner_command)
    exec_command = [
        "docker",
        "exec",
        *docker_exec_env_args(payload),
        container_name,
        "bash",
        "-lc",
        remote_command,
    ]
    process = subprocess.Popen(
        exec_command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while process.poll() is None:
        sync_dashboard_status(record, host_output_dir)
        time.sleep(2)
    stdout, stderr = process.communicate()
    completed = subprocess.CompletedProcess(
        args=command,
        returncode=int(process.returncode or 0),
        stdout=stdout,
        stderr=stderr,
    )
    append_process_output(record, completed, header)
    sync_dashboard_status(record, host_output_dir)
    return completed


def execute_command(
    record: RunRecord,
    payload: dict[str, Any],
    command: list[str],
    host_output_dir: Path,
    header: str,
) -> subprocess.CompletedProcess[str]:
    runner = payload["runner"]
    if runner == "local_python":
        return run_local_command(record, payload, command, host_output_dir, header)
    if runner == "docker_run_image":
        return run_docker_command(record, payload, command, host_output_dir, header)
    raise ValueError(f"Unsupported runner: {runner}")


def apply_dashboard_status(record: RunRecord, payload: dict[str, Any]) -> None:
    report = payload.get("report")
    if report is not None:
        record.report = report
    timings = payload.get("timings")
    if timings:
        record.timings.update(timings)
    stage = payload.get("stage")
    if stage == "estimation_ready":
        record.status = "estimate_ready"
    elif stage == "measurement_ready":
        record.status = "measurement_ready"


def sync_dashboard_status(record: RunRecord, host_output_dir: Path) -> None:
    status_path = host_output_dir / "dashboard_status.json"
    if not status_path.exists():
        return
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    apply_dashboard_status(record, payload)


def append_process_output(
    record: RunRecord, completed: subprocess.CompletedProcess[str], header: str
) -> None:
    sections = [header]
    if completed.stdout:
        sections.append(completed.stdout)
    if completed.stderr:
        sections.append(completed.stderr)
    text = "\n".join(sections).strip() + "\n"
    if completed.returncode == 0:
        record.stdout += text
    else:
        record.stderr += text


def graph_payload(record: RunRecord, graph_dir: Path) -> dict[str, Any]:
    summary_path = graph_dir / "summary.json"
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assets = []
    for path in sorted(graph_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".svg", ".html", ".json", ".txt"}:
            assets.append(
                {
                    "name": path.name,
                    "path": f"/api/runs/{record.run_id}/artifacts/graph_viz/{path.name}",
                    "kind": path.suffix.lower().lstrip("."),
                }
            )
    return {
        "status": "completed" if summary is not None else "failed",
        "output_dir": str(graph_dir),
        "summary": summary,
        "assets": assets,
        "error": None if summary is not None else "Graph export missing summary.json",
    }


@dataclass
class RemoteRankProcess:
    host: str
    ssh_port: int
    runner: str
    container_name: str | None
    process: subprocess.Popen[str]


def remote_rank_payload(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    remote = dict(payload)
    remote["node_rank"] = 1
    remote_devices = str(payload.get("remote_physical_devices", "") or "").strip()
    if remote_devices:
        remote["physical_devices"] = remote_devices
    remote_model_path = str(payload.get("remote_model_path", "") or "").strip()
    if remote_model_path:
        remote["model_path"] = remote_model_path
    remote_output_dir = str(payload.get("remote_output_dir", "") or "").strip()
    if not remote_output_dir:
        remote_output_dir = f"{output_dir}_remote_rank1"
    remote["remote_output_dir"] = remote_output_dir
    return normalize_request_payload(remote)


def start_remote_rank_process(
    record: RunRecord, payload: dict[str, Any], output_dir: Path
) -> RemoteRankProcess | None:
    if not multi_host_automation_enabled(payload):
        return None
    ensure_remote_multi_host_supported(payload)
    environment = dashboard_environment_config()
    remote_details = prepare_remote_environment(environment, payload, force=False)
    remote_payload = remote_rank_payload(payload, output_dir)
    remote_runner = str(remote_details.get("runner") or payload.get("runner") or "")
    if remote_runner == "local_python":
        remote_python = str(
            remote_payload.get("remote_python_bin")
            or remote_payload.get("python_bin")
            or default_local_python_bin()
        )
        remote_payload["python_bin"] = remote_python
    remote_command = build_predictor_command(
        remote_payload, Path(remote_payload["remote_output_dir"])
    )
    if remote_runner == "docker_run_image":
        remote_inner_command = dockerized_command(remote_command)
        remote_runtime_command = docker_runtime_command(
            remote_payload, remote_inner_command
        )
        exec_command = [
            "docker",
            "exec",
            *docker_exec_env_args(remote_payload),
            str(remote_details["container_name"]),
            "bash",
            "-lc",
            remote_runtime_command,
        ]
    elif remote_runner == "local_python":
        exec_command = local_runtime_command(
            remote_payload,
            remote_command,
            Path(str(remote_payload["remote_output_dir"])),
        )
    else:
        raise ValueError(f"Unsupported remote runner: {remote_runner}")
    ssh_command = " ".join(shell_quote(part) for part in exec_command)
    process = subprocess.Popen(
        [
            "ssh",
            "-p",
            str(remote_details["ssh_port"]),
            "-o",
            "BatchMode=yes",
            remote_details["host"],
            "bash",
            "-lc",
            ssh_command,
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return RemoteRankProcess(
        host=remote_details["host"],
        ssh_port=int(remote_details["ssh_port"]),
        runner=remote_runner,
        container_name=str(remote_details.get("container_name") or "") or None,
        process=process,
    )


def finalize_remote_rank_process(
    record: RunRecord,
    remote: RemoteRankProcess,
    payload: dict[str, Any],
) -> int:
    timeout_seconds = max(
        int(payload.get("dist_timeout_minutes", 30) or 30) * 60 + 180,
        300,
    )
    timed_out = False
    try:
        stdout, stderr = remote.process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        remote.process.kill()
        stdout, stderr = remote.process.communicate()
    return_code = int(remote.process.returncode or 0)
    if timed_out:
        return_code = 124
        if remote.runner == "docker_run_image" and remote.container_name:
            kill_command = " ".join(
                shell_quote(part)
                for part in [
                    "docker",
                    "exec",
                    remote.container_name,
                    "bash",
                    "-lc",
                    "pkill -f torch_infer_mvp.py >/dev/null 2>&1 || true",
                ]
            )
            run_ssh_command(remote.host, remote.ssh_port, kill_command, check=False)
        else:
            run_ssh_command(
                remote.host,
                remote.ssh_port,
                "pkill -f torch_infer_mvp.py >/dev/null 2>&1 || true",
                check=False,
            )
        stderr = (stderr or "") + "\nremote rank timed out while waiting for completion"
    completed = subprocess.CompletedProcess(
        args=["ssh", remote.host],
        returncode=return_code,
        stdout=stdout,
        stderr=stderr,
    )
    append_process_output(record, completed, "== remote rank1 torch_infer_mvp ==")
    return return_code


def run_job(record: RunRecord) -> None:
    output_dir, output_policy = resolve_run_output_dir(record.run_id, record.created_at)
    graph_dir = output_dir / "graph_viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    record.output_dir = str(output_dir)
    record.output_policy = output_policy
    predictor_command = build_predictor_command(record.request, output_dir)
    record.command = predictor_command
    record.status = "running"
    record.started_at = time.time()
    record.timings = {
        "estimation_wall_time_s": None,
        "measurement_wall_time_s": None,
        "predictor_total_wall_time_s": None,
        "predictor_wall_time_s": None,
        "calibration_wall_time_s": None,
        "model_load_wall_time_s": None,
        "graph_extract_wall_time_s": None,
        "runtime_inputs_wall_time_s": None,
        "torch_export_wall_time_s": None,
        "graph_cache_load_wall_time_s": None,
        "graph_cache_write_wall_time_s": None,
        "table_lookup_wall_time_s": None,
        "runtime_prepare_wall_time_s": None,
        "module_profile_wall_time_s": None,
        "analytical_estimate_wall_time_s": None,
        "graph_export_wall_time_s": None,
        "total_wall_time_s": None,
    }
    try:
        predictor_started = time.time()
        remote_rank = start_remote_rank_process(record, record.request, output_dir)
        remote_return_code: int | None = None
        try:
            completed = execute_command(
                record,
                record.request,
                predictor_command,
                output_dir,
                "== torch_infer_mvp ==",
            )
        finally:
            if remote_rank is not None:
                remote_return_code = finalize_remote_rank_process(
                    record, remote_rank, record.request
                )
        if (
            remote_rank is not None
            and remote_return_code is not None
            and completed.returncode == 0
            and remote_return_code != 0
        ):
            completed = subprocess.CompletedProcess(
                args=completed.args,
                returncode=remote_return_code,
                stdout=completed.stdout,
                stderr=(completed.stderr or "")
                + "\nremote rank1 failed; see remote logs above",
            )
        record.timings["predictor_wall_time_s"] = time.time() - predictor_started
        record.return_code = completed.returncode
        report_path = output_dir / "report.json"
        if completed.returncode != 0 or not report_path.exists():
            record.status = "failed"
            record.error = "Run failed before report.json was produced."
            return

        record.report = json.loads(report_path.read_text(encoding="utf-8"))
        record.status = "rendering_graph"

        if record.request.get("generate_graph_viz", True):
            graph_command = build_graph_command(record.request, graph_dir)
            record.graph_command = graph_command
            graph_started = time.time()
            graph_completed = execute_command(
                record,
                record.request,
                graph_command,
                graph_dir,
                "== export_graph_viz ==",
            )
            record.timings["graph_export_wall_time_s"] = time.time() - graph_started
            record.graph = graph_payload(record, graph_dir)
            if graph_completed.returncode != 0:
                record.graph["status"] = "failed"
                record.graph["error"] = "Graph export command failed."
        else:
            record.graph = {
                "status": "disabled",
                "output_dir": "",
                "summary": None,
                "assets": [],
                "error": None,
            }
        record.status = "completed"
    except Exception as exc:  # pragma: no cover
        record.status = "failed"
        record.error = str(exc)
    finally:
        record.finished_at = time.time()
        if record.started_at is not None:
            record.timings["total_wall_time_s"] = record.finished_at - record.started_at


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def serve_artifact(self, record: RunRecord, relative_path: str) -> None:
        artifact_root = Path(record.output_dir).resolve()
        target = (artifact_root / relative_path).resolve()
        if artifact_root not in target.parents and artifact_root != target:
            write_json(self, {"error": "Invalid artifact path"}, status=400)
            return
        if not target.exists() or not target.is_file():
            write_json(self, {"error": "Artifact not found"}, status=404)
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
        if parsed.path == "/api/config":
            settings = dashboard_settings()
            write_json(
                self,
                {
                    "defaults": dashboard_request_defaults(),
                    "models": local_model_paths(),
                    "local_gpus": local_gpu_inventory(),
                    "accelerator_kind": detect_local_accelerator_kind(),
                    "accelerator_label": accelerator_label(
                        detect_local_accelerator_kind()
                    ),
                    "cwd": str(ROOT),
                    "environment": environment_state_payload(),
                    "environment_config_path": settings["config_path"],
                    "environment_locked": True,
                },
            )
            return
        if parsed.path == "/api/environment":
            write_json(self, environment_state_payload())
            return
        if parsed.path == "/api/runs":
            with RUNS_LOCK:
                runs = [
                    item.to_dict()
                    for item in sorted(
                        RUNS.values(), key=lambda rec: rec.created_at, reverse=True
                    )
                ]
            write_json(self, {"runs": runs})
            return
        if parsed.path.startswith("/api/runs/"):
            parts = parsed.path.split("/")
            if len(parts) >= 5 and parts[4] == "artifacts":
                run_id = parts[3]
                relative_path = "/".join(parts[5:])
                with RUNS_LOCK:
                    record = RUNS.get(run_id)
                if record is None:
                    write_json(self, {"error": "Run not found"}, status=404)
                    return
                self.serve_artifact(record, relative_path)
                return
            run_id = parsed.path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                record = RUNS.get(run_id)
            if record is None:
                write_json(self, {"error": "Run not found"}, status=404)
                return
            write_json(self, record.to_dict())
            return
        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/environment/prepare":
            payload = dashboard_request_defaults()
            payload.update(read_json(self))
            try:
                payload = normalize_request_payload(payload)
            except Exception as exc:
                write_json(self, {"error": str(exc)}, status=400)
                return
            write_json(
                self,
                prepare_environment_async(force=False, prepare_payload=payload),
                status=HTTPStatus.ACCEPTED,
            )
            return
        if parsed.path == "/api/environment/stop":
            write_json(self, stop_environment())
            return
        if parsed.path != "/api/runs":
            write_json(self, {"error": "Not found"}, status=404)
            return
        environment = environment_state_payload()
        if not environment["ready"]:
            write_json(
                self,
                {
                    "error": "Environment is not ready; prepare it before starting a run."
                },
                status=409,
            )
            return
        payload = dashboard_request_defaults()
        payload.update(read_json(self))
        try:
            payload = normalize_request_payload(payload)
        except Exception as exc:
            write_json(self, {"error": str(exc)}, status=400)
            return
        if is_multi_host_request(payload):
            try:
                ensure_remote_multi_host_supported(payload)
            except Exception as exc:
                write_json(self, {"error": str(exc)}, status=400)
                return
        payload["generate_graph_viz"] = bool(payload.get("generate_graph_viz", True))
        run_id = uuid.uuid4().hex[:8]
        record = RunRecord(run_id=run_id, request=payload)
        with RUNS_LOCK:
            RUNS[run_id] = record
        thread = threading.Thread(target=run_job, args=(record,), daemon=True)
        thread.start()
        write_json(
            self,
            {"run_id": run_id, "status": record.status},
            status=HTTPStatus.ACCEPTED,
        )


def main() -> None:
    settings = configure_dashboard_settings()
    if settings["environment"].get("auto_prepare", True):
        prepare_environment_async(force=False)
    host = os.environ.get("MVP_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("MVP_DASHBOARD_PORT", "8123"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Torch MVP dashboard listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
