"""
Real-time Data Buffer Module
Manages a 90-second sliding window of xyz coordinate data
"""
import threading
from collections import deque


class RealTimeDataBuffer:
    """Real-time data buffer managing a 90-second sliding window of xyz coordinate data"""

    def __init__(self, window_seconds=90, fps=30):
        self.window_seconds = window_seconds
        self.fps = fps
        self.max_samples = window_seconds * fps

        # Use deque as circular buffer
        self.timestamps = deque(maxlen=self.max_samples)
        self.left_x = deque(maxlen=self.max_samples)
        self.left_y = deque(maxlen=self.max_samples)
        self.left_z = deque(maxlen=self.max_samples)
        self.right_x = deque(maxlen=self.max_samples)
        self.right_y = deque(maxlen=self.max_samples)
        self.right_z = deque(maxlen=self.max_samples)

        self.lock = threading.Lock()

    def add_data(self, timestamp, left_xyz, right_xyz):
        """Add a new xyz data point"""
        with self.lock:
            self.timestamps.append(timestamp)
            self.left_x.append(left_xyz[0])
            self.left_y.append(left_xyz[1])
            self.left_z.append(left_xyz[2])
            self.right_x.append(right_xyz[0])
            self.right_y.append(right_xyz[1])
            self.right_z.append(right_xyz[2])

    def get_data_arrays(self):
        """Get all data arrays (for plotting)"""
        with self.lock:
            return {
                'timestamps': list(self.timestamps),
                'left_x': list(self.left_x),
                'left_y': list(self.left_y),
                'left_z': list(self.left_z),
                'right_x': list(self.right_x),
                'right_y': list(self.right_y),
                'right_z': list(self.right_z)
            }

    def clear(self):
        """Clear all data"""
        with self.lock:
            self.timestamps.clear()
            self.left_x.clear()
            self.left_y.clear()
            self.left_z.clear()
            self.right_x.clear()
            self.right_y.clear()
            self.right_z.clear()
