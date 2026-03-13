#!/usr/bin/env bash
# cleanup.sh — Clean up all leftover processes, ports, and shared memory
set -e

echo "Cleaning up port 5555 (image server)..."
lsof -ti :5555 | xargs kill -9 2>/dev/null && echo "  Port 5555 cleaned" || echo "  No leftover processes on port 5555"

echo "Cleaning up port 8000 (web server)..."
lsof -ti :8000 | xargs kill -9 2>/dev/null && echo "  Port 8000 cleaned" || echo "  No leftover processes on port 8000"

echo "Cleaning up port 8012 (OpenTeleVision vuer server)..."
lsof -ti :8012 | xargs kill -9 2>/dev/null && echo "  Port 8012 cleaned" || echo "  No leftover processes on port 8012"

echo "Cleaning up shared memory integrated_capture..."
python3 -c "
from multiprocessing import shared_memory
try:
    shm = shared_memory.SharedMemory(name='integrated_capture')
    shm.close(); shm.unlink()
    print('  Shared memory integrated_capture cleaned')
except Exception:
    print('  Shared memory integrated_capture does not exist')
"
