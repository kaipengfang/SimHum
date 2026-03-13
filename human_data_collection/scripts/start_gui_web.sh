#!/usr/bin/env bash
# start_gui_web.sh — Start Web GUI (FastAPI server version)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[start_gui_web] Project directory: ${PROJECT_DIR}"

# Check Python dependencies
for pkg in fastapi uvicorn; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        echo "[start_gui_web] Installing $pkg..."
        pip install "$pkg"
    fi
done

cd "${PROJECT_DIR}"

# Check if port 8000 is in use
PORT=8000
PORT_PIDS=$(lsof -ti ":${PORT}" 2>/dev/null || true)
if [ -n "${PORT_PIDS}" ]; then
    echo ""
    echo "[start_gui_web] Warning: Port ${PORT} is already in use by:"
    for pid in ${PORT_PIDS}; do
        echo "    PID ${pid}: $(ps -p "${pid}" -o comm= 2>/dev/null || echo 'unknown process')"
    done
    echo ""
    read -r -p "[start_gui_web] Force kill these processes? [y/N] " answer
    case "${answer}" in
        [yY]|[yY][eE][sS])
            for pid in ${PORT_PIDS}; do
                kill -9 "${pid}" 2>/dev/null || true
            done
            sleep 1
            echo "[start_gui_web] Port ${PORT} released"
            ;;
        *)
            echo "[start_gui_web] Startup cancelled"
            exit 1
            ;;
    esac
fi

python3 gui_web_server.py &
SERVER_PID=$!

# Open browser in --app mode (standalone app window, no tab bar, closing doesn't affect other windows)
BROWSER_PID=""
open_browser() {
    local URL="$1"
    for browser in google-chrome google-chrome-stable chromium-browser chromium; do
        if command -v "$browser" &>/dev/null; then
            "$browser" --app="$URL" 2>/dev/null &
            BROWSER_PID=$!
            echo "[start_gui_web] Using $browser --app mode (PID: ${BROWSER_PID})"
            return
        fi
    done
    # Firefox does not support --app, fall back to --new-window
    if command -v firefox &>/dev/null; then
        firefox --new-window "$URL" 2>/dev/null &
        BROWSER_PID=$!
        echo "[start_gui_web] Using firefox --new-window (PID: ${BROWSER_PID})"
        return
    fi
    xdg-open "$URL"
    echo "[start_gui_web] Using xdg-open (browser must be closed manually on exit)"
}

# On Ctrl+C / script exit: close app window + kill server + release port
cleanup() {
    echo ""
    echo "[start_gui_web] Shutting down..."

    # 1. Close browser app window
    if [ -n "${BROWSER_PID}" ]; then
        kill "${BROWSER_PID}" 2>/dev/null || true
    fi

    # 2. Stop server
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true

    # 3. Fallback: ensure port is released
    fuser -k 8000/tcp 2>/dev/null || true
    echo "[start_gui_web] Exited"
    exit 0
}
trap cleanup INT TERM

sleep 2

echo "[start_gui_web] Opening browser..."
open_browser "http://localhost:8000"

wait "${SERVER_PID}"
