#!/usr/bin/env bash
set -euo pipefail

# Resolve everything relative to this script's own location, not the
# caller's cwd, so it works no matter where it's invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
ENGINE_SRC="$SCRIPT_DIR/project_1_pairs/engine.cpp"
ENGINE_BIN="$SCRIPT_DIR/project_1_pairs/engine"

echo "[1/3] Checking execution engine..."
if [ ! -f "$ENGINE_BIN" ] || [ "$ENGINE_SRC" -nt "$ENGINE_BIN" ]; then
    echo "Building engine (source is newer than binary, or binary missing)..."
    clang++ -std=c++17 "$ENGINE_SRC" -o "$ENGINE_BIN" -lsqlite3
else
    echo "Engine binary is up to date, skipping rebuild."
fi

echo "[2/3] Running Statistical Calculation Engine..."
"$VENV_PYTHON" "$SCRIPT_DIR/project_1_pairs/zscore_calculator.py"

echo "[3/3] Launching Interactive Control Interface..."
"$VENV_PYTHON" -m streamlit run "$SCRIPT_DIR/project_1_pairs/dashboard.py"
