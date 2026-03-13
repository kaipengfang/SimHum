#!/usr/bin/env python3
"""
Integrated Motion and Image Capture System (Web Server)
Uses FastAPI to provide HTTP API, frontend accessed via browser
"""
import sys
import os
import logging

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from common.font_utils import configure_matplotlib_fonts
from common.matrix_utils import extract_xyz_from_matrix, matrix_to_xyz_quaternion
from common.data_buffer import RealTimeDataBuffer
from common.episode_manager import EpisodeManager
from server.server_api import ServerAPI
from server.capture_api import CaptureAPI
from server.recording_api import RecordingAPI
from server.data_api import DataAPI

configure_matplotlib_fonts()

import time as _time
_SERVER_START_TIME = _time.time()  # Unique identifier per startup


class AppAPI(ServerAPI, CaptureAPI, RecordingAPI, DataAPI):
    """
    Merge all API mixins as the backend API implementation
    """

    def __init__(self):
        # ── Shared state (all API mixins access via self) ──────────
        self.image_server_process = None
        self.image_server_thread = None
        self.capture_system = None
        self.capture_thread = None

        self.is_image_server_running = False
        self.is_capture_running = False
        self.server_connection_tested = False
        self.server_port = 5555
        self.is_shutting_down = False

        self.data_buffer = RealTimeDataBuffer(window_seconds=90, fps=30)
        self.episode_manager = EpisodeManager(initial_episode=0)

        # Recording state machine: 'waiting' | 'countdown' | 'recording'
        self.recording_state = 'waiting'
        self.is_recording_episode = False
        self.is_countdown_active = False
        self.countdown_value = 0

        self._log_queue = []
        self._log_lock = __import__('threading').Lock()

        # Clean up leftover processes on startup
        self.cleanup_existing_processes()
        # Async check for existing server
        __import__('threading').Thread(target=self._init_check_server, daemon=True).start()

    def _init_check_server(self):
        """Delayed check for existing server after startup"""
        import time
        time.sleep(0.8)
        self.connect_to_existing_server()

    def shutdown(self):
        """Clean up resources on window close"""
        self.is_shutting_down = True
        self.is_capture_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.is_image_server_running:
            self._stop_image_server()
        return {'ok': True}

    # ── Override DataAPI._push_log for thread safety ────────────────
    def _push_log(self, message, level='info'):
        import datetime
        entry = {
            'time': datetime.datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'message': message,
        }
        with self._log_lock:
            self._log_queue.append(entry)

    def _flush_log_queue(self):
        with self._log_lock:
            logs = list(self._log_queue)
            self._log_queue.clear()
        return logs


# SSE client queue list
_sse_clients: list[asyncio.Queue] = []


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # startup: attach filter after uvicorn dictConfig completes
    for _name in ('uvicorn.error', 'uvicorn', 'asyncio'):
        logging.getLogger(_name).addFilter(_BrokenPipeFilter())
    logging.getLogger('uvicorn.access').addFilter(_PollingFilter())
    loop = asyncio.get_event_loop()
    _orig = loop.get_exception_handler() or loop.default_exception_handler
    def _exc_handler(loop, context):
        if isinstance(context.get('exception'), _SUPPRESS_EXCS):
            return
        _orig(context)
    loop.set_exception_handler(_exc_handler)

    yield

    # shutdown: push close signal to all connected pages
    for q in list(_sse_clients):
        await q.put('shutdown')
    if _sse_clients:
        await asyncio.sleep(0.5)  # Wait for clients to receive the message


# Create FastAPI app
app = FastAPI(title="Integrated Data Acquisition System v1.0", lifespan=_lifespan)

# CORS middleware (allow browser cross-origin access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API instance
api_instance = AppAPI()


# ── API Routes ──────────────────────────────────────────

@app.post("/api/toggle_server")
def toggle_server():
    return api_instance.toggle_server()

@app.post("/api/connect_to_existing_server")
def connect_to_existing_server():
    return api_instance.connect_to_existing_server()

@app.post("/api/check_server_connection")
def check_server_connection():
    return api_instance.check_server_connection()

@app.get("/api/get_server_status")
def get_server_status():
    return api_instance.get_server_status()

@app.post("/api/start_capture")
def start_capture(path: str = '', description: str = 'test'):
    return api_instance.start_capture(path, description)

@app.post("/api/stop_capture")
def stop_capture():
    return api_instance.stop_capture()

@app.get("/api/browse_path")
def browse_path():
    # Web version does not support native file dialog, return hint
    return {'ok': True, 'path': None, 'message': 'Please enter path manually in web version'}

@app.post("/api/set_episode")
def set_episode(value: int = Body(..., embed=True)):
    return api_instance.set_episode(value)

@app.post("/api/start_recording")
def start_recording():
    return api_instance.start_recording()

@app.post("/api/stop_recording")
def stop_recording():
    return api_instance.stop_recording()

@app.post("/api/drop_recording")
def drop_recording():
    return api_instance.drop_recording()

@app.get("/api/get_full_status")
def get_full_status():
    result = api_instance.get_full_status()
    result['server_start_time'] = _SERVER_START_TIME
    return result

@app.get("/api/get_realtime_data")
def get_realtime_data():
    return api_instance.get_realtime_data()

@app.get("/api/get_image_frame")
def get_image_frame():
    return api_instance.get_image_frame()

@app.post("/api/shutdown")
async def shutdown():
    result = api_instance.shutdown()
    # Delay server shutdown
    import asyncio
    asyncio.create_task(_delayed_shutdown())
    return result

async def _delayed_shutdown():
    """Shut down server after 1-second delay"""
    import asyncio
    await asyncio.sleep(1)
    os._exit(0)


# ── Static File Serving ──────────────────────────────────────

# Mount static file directories
app.mount("/css", StaticFiles(directory=os.path.join(current_dir, "web/css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(current_dir, "web/js")), name="js")

@app.get("/api/events")
async def sse_events():
    """SSE long connection for pushing events (e.g. shutdown) from server to page"""
    queue: asyncio.Queue = asyncio.Queue()
    _sse_clients.append(queue)

    async def generate():
        try:
            while True:
                msg = await queue.get()
                yield f"data: {msg}\n\n"
                if msg == 'shutdown':
                    break
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _sse_clients:
                _sse_clients.remove(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def root():
    """Return main page HTML"""
    return FileResponse(os.path.join(current_dir, "web/index.html"))


_SUPPRESS_EXCS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

class _BrokenPipeFilter(logging.Filter):
    def filter(self, record):
        if record.exc_info and record.exc_info[0] is not None:
            if issubclass(record.exc_info[0], _SUPPRESS_EXCS):
                return False
        msg = record.getMessage()
        return 'Broken pipe' not in msg and 'BrokenPipeError' not in msg


_POLLING_PATHS = {'/api/get_full_status', '/api/get_realtime_data', '/api/get_image_frame'}

class _PollingFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(path in msg for path in _POLLING_PATHS)


def main():
    """Start FastAPI server"""

    print("=" * 60)
    print("🚀 Integrated Data Acquisition System v1.0")
    print("=" * 60)
    print(f"📡 Server starting at: http://localhost:8000")
    print(f"📂 Working directory: {current_dir}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == '__main__':
    main()
