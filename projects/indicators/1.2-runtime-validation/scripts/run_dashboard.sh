#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${RUNTEST_DASHBOARD_PORT:-8242}"

find_listen_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
    return
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$1" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' || true
    return
  fi
  ss -ltnp 2>/dev/null | sed -n "s/.*:$1 .*pid=\([0-9]\+\).*/\1/p" | sort -u
}

cleanup_port() {
  mapfile -t pids < <(find_listen_pids "$1")
  if ((${#pids[@]} == 0)); then
    return
  fi
  echo "Port $1 is busy, stopping: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 1
  mapfile -t pids < <(find_listen_pids "$1")
  if ((${#pids[@]} > 0)); then
    echo "Force stopping remaining processes: ${pids[*]}"
    kill -9 "${pids[@]}" 2>/dev/null || true
    sleep 1
  fi
}

cleanup_port "$PORT"
cd "$ROOT"
echo "Starting indicator 1.2 dashboard: http://127.0.0.1:${PORT}"
python3 dashboard.py
