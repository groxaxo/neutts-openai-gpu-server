#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export NEUTTS_ROOT="${NEUTTS_ROOT:-/home/op/neutts}"
export NEUTTS_VENV="${NEUTTS_VENV:-$NEUTTS_ROOT/.venv}"
export NEUTTS_VOICES_DIR="${NEUTTS_VOICES_DIR:-$NEUTTS_ROOT/samples}"
export NEUTTS_HOST="${NEUTTS_HOST:-127.0.0.1}"
export NEUTTS_PORT="${NEUTTS_PORT:-12436}"
export NEUTTS_BACKBONE_DEVICE="${NEUTTS_BACKBONE_DEVICE:-cpu}"
export NEUTTS_CODEC_DEVICE="${NEUTTS_CODEC_DEVICE:-cpu}"
export PYTHONPATH="$REPO_DIR:$NEUTTS_ROOT:${PYTHONPATH:-}"

exec env -u LD_LIBRARY_PATH "$NEUTTS_VENV/bin/python" "$REPO_DIR/server.py"
