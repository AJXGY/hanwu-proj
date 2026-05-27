#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/projects/operators/communication"
exec bash scripts/run_comm_response_time_tool_validation.sh "$@"
