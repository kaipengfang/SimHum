#!/usr/bin/env bash
# start_image_server.sh — Start RealSense image server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[start_image_server] Project directory: ${PROJECT_DIR}"

cd "${PROJECT_DIR}"
python3 -m camera.server --terminal-mode
