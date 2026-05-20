#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="${NEUTTS_LOG_FILE:-/tmp/neutts_gpu_server.log}"

if pgrep -f "$REPO_DIR/server.py" >/dev/null; then
  echo "NeuTTS GPU server already running:"
  pgrep -af "$REPO_DIR/server.py"
  exit 0
fi

setsid -f "$REPO_DIR/scripts/launch_gpu_local.sh" > "$LOG_FILE" 2>&1 < /dev/null
sleep 1

if pgrep -f "$REPO_DIR/server.py" >/dev/null; then
  echo "Started NeuTTS GPU server:"
  pgrep -af "$REPO_DIR/server.py"
  echo "Log: $LOG_FILE"
else
  echo "Failed to start NeuTTS GPU server. Log: $LOG_FILE" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi
