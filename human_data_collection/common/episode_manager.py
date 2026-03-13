"""
Episode Manager Module
Manage episode numbering and synchronization
"""
from typing import Callable, List


class EpisodeManager:
    """Episode number manager"""

    def __init__(self, initial_episode=0):
        self._current_episode = initial_episode
        self._listeners: List[Callable[[int], None]] = []

    def on_episode_changed(self, callback: Callable[[int], None]):
        """Register episode change callback"""
        self._listeners.append(callback)

    def _notify(self):
        for cb in self._listeners:
            cb(self._current_episode)

    def get_current_episode(self):
        """Get current episode number"""
        return self._current_episode

    def set_episode(self, episode):
        """Set episode number"""
        if episode != self._current_episode:
            self._current_episode = episode
            self._notify()

    def increment(self):
        """Increment episode number"""
        self._current_episode += 1
        self._notify()

    def reset(self):
        """Reset episode counter"""
        self._current_episode = 0
        self._notify()

    def sync_to_capture_system(self, capture_system):
        """Sync episode number to capture system"""
        if capture_system is not None:
            capture_system.episode = self._current_episode
