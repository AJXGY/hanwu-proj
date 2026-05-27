#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/projects/operators/compute"
exec bash scripts/run_compute_op_space_validation.sh "$@"
