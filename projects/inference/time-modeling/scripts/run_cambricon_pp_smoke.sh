#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "PP inference smoke is disabled in hanwu-proj."
echo "Running the standard TP two-card inference validation instead."
exec bash "$ROOT/scripts/run_cambricon_infer_smoke.sh" "$@"
