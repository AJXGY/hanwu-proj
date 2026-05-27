#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/projects/indicators/1.2-runtime-validation"
exec bash scripts/run_dashboard.sh "$@"
