#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf tmp tmp_* artifacts 2>/dev/null || true
rm -f ./*.aux ./*.fdb_latexmk ./*.fls ./*.log ./*.synctex.gz ./*.xdv 2>/dev/null || true
