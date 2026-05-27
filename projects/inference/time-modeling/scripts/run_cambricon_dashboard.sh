#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MVP_DASHBOARD_CONFIG="${MVP_DASHBOARD_CONFIG:-$ROOT/configs/dashboard_env.json}"
export MVP_DASHBOARD_HOST="${MVP_DASHBOARD_HOST:-127.0.0.1}"
export MVP_DASHBOARD_PORT="${MVP_DASHBOARD_PORT:-8123}"
export MVP_DASHBOARD_OUTPUT_ROOT="${MVP_DASHBOARD_OUTPUT_ROOT:-$ROOT/dashboard_runs}"
export MVP_DASHBOARD_ENV_CONTAINER="${MVP_DASHBOARD_ENV_CONTAINER:-mvp-dashboard-env-mlu}"

find_listen_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
    return
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' || true
    return
  fi
  ss -ltnp 2>/dev/null | sed -n "s/.*:${port} .*pid=\([0-9]\+\).*/\1/p" | sort -u
}

cleanup_port() {
  local port="$1"
  mapfile -t pids < <(find_listen_pids "$port")
  if ((${#pids[@]} == 0)); then
    return
  fi
  echo "Port ${port} is busy, stopping: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 1
  mapfile -t pids < <(find_listen_pids "$port")
  if ((${#pids[@]} > 0)); then
    echo "Force stopping remaining processes on port ${port}: ${pids[*]}"
    kill -9 "${pids[@]}" 2>/dev/null || true
    sleep 1
  fi
}

cleanup_container_dashboard() {
  local container_name="$1"
  if ! command -v docker >/dev/null 2>&1; then
    return
  fi
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -Fxq "$container_name"; then
    return
  fi
  echo "Stopping stale dashboard processes in container: ${container_name}"
  docker exec "$container_name" bash -lc "pkill -f 'python3 mvp_dashboard.py' || pkill -f 'mvp_dashboard.py' || true" >/dev/null 2>&1 || true
  sleep 1
}

cleanup_container_dashboard "$MVP_DASHBOARD_ENV_CONTAINER"
cleanup_port "$MVP_DASHBOARD_PORT"

cd "$ROOT"
echo "Starting inference dashboard: http://${MVP_DASHBOARD_HOST}:${MVP_DASHBOARD_PORT}"
echo "  config: $MVP_DASHBOARD_CONFIG"
echo "  output root: $MVP_DASHBOARD_OUTPUT_ROOT"
python3 mvp_dashboard.py
