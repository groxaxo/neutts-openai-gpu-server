#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export NEUTTS_VENV="${NEUTTS_VENV:-/home/op/neutts/.venv}"
export NEUTTS_UI_BASE_URL="${NEUTTS_UI_BASE_URL:-http://127.0.0.1:12435}"
export NEUTTS_UI_HOST="${NEUTTS_UI_HOST:-127.0.0.1}"
export NEUTTS_UI_PORT="${NEUTTS_UI_PORT:-7860}"
export NEUTTS_UI_SHARE="${NEUTTS_UI_SHARE:-0}"

args=(--base-url "$NEUTTS_UI_BASE_URL" --host "$NEUTTS_UI_HOST" --port "$NEUTTS_UI_PORT")
if [ "${NEUTTS_UI_SHARE:-0}" = "1" ]; then
  args+=(--share)
fi

exec "$NEUTTS_VENV/bin/python" "$REPO_DIR/gradio_app.py" "${args[@]}"
