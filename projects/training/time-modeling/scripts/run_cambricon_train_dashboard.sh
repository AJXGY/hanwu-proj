#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRAIN_DASHBOARD_HOST="${TRAIN_DASHBOARD_HOST:-127.0.0.1}"
export TRAIN_DASHBOARD_PORT="${TRAIN_DASHBOARD_PORT:-8234}"

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

cleanup_port "$TRAIN_DASHBOARD_PORT"

cd "$ROOT"
echo "Starting training dashboard: http://${TRAIN_DASHBOARD_HOST}:${TRAIN_DASHBOARD_PORT}"
python3 train_dashboard.py
