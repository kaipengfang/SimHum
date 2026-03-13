"""
Data Query API
Provides real-time image frame, xyz data, and aggregated status query interfaces
"""
import base64
import time
import subprocess
import numpy as np
import cv2

_ADB_CACHE_TTL = 5.0  # seconds


class DataAPI:
    """Real-time data query interface, called by JS polling"""

    def get_image_frame(self):
        """Return current image frame as base64 JPEG string"""
        try:
            if not self.capture_system:
                return {'ok': False, 'data': None}
            handler = getattr(self.capture_system, 'image_handler', None)
            if handler is None:
                return {'ok': False, 'data': None}
            img = handler.get_current_image()
            if img is None:
                return {'ok': False, 'data': None}
            # Take left half only when width exceeds 1280 (head camera)
            if len(img.shape) == 3 and img.shape[1] > 1280:
                img = img[:, :1280, :]
            img_copy = img.copy()
            # Scale down to reduce transfer size
            h, w = img_copy.shape[:2]
            if w > 640:
                scale = 640 / w
                img_copy = cv2.resize(img_copy, (640, int(h * scale)))
            img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode('utf-8')
            return {'ok': True, 'data': b64}
        except Exception:
            return {'ok': False, 'data': None}

    def get_realtime_data(self):
        """Return latest hand xyz data (for chart polling)"""
        try:
            arrays = self.data_buffer.get_data_arrays()
            if len(arrays['timestamps']) == 0:
                return {'ok': True, 'data': None}
            return {
                'ok': True,
                'data': {
                    'timestamps': arrays['timestamps'],
                    'left_x': arrays['left_x'],
                    'left_y': arrays['left_y'],
                    'left_z': arrays['left_z'],
                    'right_x': arrays['right_x'],
                    'right_y': arrays['right_y'],
                    'right_z': arrays['right_z'],
                }
            }
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def get_full_status(self):
        """Aggregated status query (1-second polling)"""
        try:
            # Capture system status
            frame_count = 0
            if self.capture_system:
                status = self.capture_system.get_status()
                frame_count = status.get('data_count', 0) if status else 0

            # ADB status (check every 5 seconds to avoid high-frequency calls disturbing ADB daemon)
            adb_connected = self._get_adb_status()

            if self.server_connection_tested:
                server_status = 'connected'
            elif self.is_image_server_running:
                server_status = 'connecting'
            else:
                server_status = 'disconnected'

            return {
                'ok': True,
                'server_connected': self.server_connection_tested,
                'server_status': server_status,
                'capture_running': self.is_capture_running,
                'recording_state': self.recording_state,
                'is_recording': self.is_recording_episode,
                'is_countdown': self.is_countdown_active,
                'countdown_value': self.countdown_value,
                'episode': self.episode_manager.get_current_episode(),
                'frame_count': frame_count,
                'adb_connected': adb_connected,
                'logs': self._flush_log_queue(),
            }
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def _flush_log_queue(self):
        """Return and clear the pending log list"""
        if not hasattr(self, '_log_queue'):
            self._log_queue = []
        logs = list(self._log_queue)
        self._log_queue.clear()
        return logs

    def _get_adb_status(self) -> bool:
        """Check ADB connection status, result cached for 5 seconds to avoid disturbing ADB daemon"""
        now = time.time()
        if not hasattr(self, '_adb_last_check'):
            self._adb_last_check = 0.0
            self._adb_cached = False
        if now - self._adb_last_check < _ADB_CACHE_TTL:
            return self._adb_cached
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=2)
            self._adb_cached = '\tdevice' in result.stdout
        except Exception:
            self._adb_cached = False
        self._adb_last_check = now
        return self._adb_cached

    def _push_log(self, message, level='info'):
        """Add log entry to the pending push queue (called by internal methods)"""
        if not hasattr(self, '_log_queue'):
            self._log_queue = []
        import datetime
        self._log_queue.append({
            'time': datetime.datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'message': message,
        })
