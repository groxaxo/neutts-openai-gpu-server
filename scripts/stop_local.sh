#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mapfile -t pids < <(pgrep -f "$REPO_DIR/server.py" || true)
if [ "${#pids[@]}" -eq 0 ]; then
  echo "No NeuTTS server process found for $REPO_DIR"
  exit 0
fi

printf 'Stopping PIDs: %s\n' "${pids[*]}"
kill "${pids[@]}"
