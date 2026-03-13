"""
Recording Control API
Handles countdown, recording start/stop/discard
"""
import threading
import time


class RecordingAPI:
    """Recording control interface"""

    def start_recording(self):
        """Trigger recording (with 3-second countdown)"""
        try:
            if not self.is_capture_running:
                return {'ok': False, 'error': 'Capture system not running'}
            if self.recording_state == 'recording':
                return self.stop_recording()
            if self.recording_state == 'countdown':
                return {'ok': False, 'error': 'Countdown already active'}

            self._start_recording_countdown()
            return {'ok': True, 'state': 'countdown'}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def stop_recording(self):
        """Stop current recording"""
        try:
            if not self.is_recording_episode:
                return {'ok': False, 'error': 'Not recording'}

            self.recording_state = 'waiting'
            self.is_recording_episode = False

            quality_ok = True
            if self.capture_system:
                try:
                    if (hasattr(self.capture_system, 'is_recording') and
                            self.capture_system.is_recording()):
                        if hasattr(self.capture_system, 'manual_stop_recording'):
                            result, quality_result = self.capture_system.manual_stop_recording()
                            if quality_result and not quality_result.get('is_valid', True):
                                quality_ok = False
                                self._push_log(
                                    "Episode discarded: hands remained still too long",
                                    'error'
                                )
                            else:
                                # Recording successful, increment episode
                                self.episode_manager.increment()
                                self.episode_manager.sync_to_capture_system(self.capture_system)
                                self._push_log(
                                    f"Episode saved. Next: {self.episode_manager.get_current_episode()}",
                                    'success'
                                )
                except Exception as e:
                    self._push_log(f"Error stopping recording: {e}", 'error')

            return {
                'ok': True,
                'quality_ok': quality_ok,
                'episode': self.episode_manager.get_current_episode(),
            }
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def drop_recording(self):
        """Discard current recording"""
        try:
            if self.is_countdown_active:
                self._stop_countdown()
                self._push_log("Countdown cancelled", 'info')
                return {'ok': True, 'dropped': False}

            if not self.is_recording_episode:
                return {'ok': False, 'error': 'Not recording'}

            self.recording_state = 'waiting'
            self.is_recording_episode = False

            if self.capture_system:
                try:
                    if hasattr(self.capture_system, 'manual_drop_recording'):
                        self.capture_system.manual_drop_recording()
                    elif hasattr(self.capture_system, 'manual_stop_recording'):
                        self.capture_system.manual_stop_recording()
                except Exception:
                    pass

            episode = self.episode_manager.get_current_episode()
            self._push_log(f"Episode {episode} discarded", 'warning')
            return {'ok': True, 'dropped': True, 'episode': episode}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def get_recording_state(self):
        """Return current recording state machine status"""
        return {
            'ok': True,
            'state': self.recording_state,
            'is_recording': self.is_recording_episode,
            'is_countdown': self.is_countdown_active,
            'countdown_value': self.countdown_value,
            'episode': self.episode_manager.get_current_episode(),
        }

    # ── Private Methods ──────────────────────────────────────────

    def _start_recording_countdown(self):
        """Start 3-second countdown thread"""
        self.countdown_value = 3
        self.is_countdown_active = True
        self.recording_state = 'countdown'

        def _tick():
            for i in range(3, 0, -1):
                if not self.is_countdown_active:
                    return
                self.countdown_value = i
                time.sleep(1.0)
            if self.is_countdown_active:
                self._stop_countdown()
                self._start_actual_recording()

        threading.Thread(target=_tick, daemon=True).start()

    def _stop_countdown(self):
        """Cancel countdown"""
        self.is_countdown_active = False
        self.countdown_value = 0
        if self.recording_state == 'countdown':
            self.recording_state = 'waiting'

    def _start_actual_recording(self):
        """Start recording after countdown finishes"""
        try:
            self.recording_state = 'recording'
            self.is_recording_episode = True

            if self.capture_system:
                if hasattr(self.capture_system, 'manual_start_recording'):
                    self.capture_system.manual_start_recording()
                elif hasattr(self.capture_system, 'start_recording'):
                    self.capture_system.start_recording()
                else:
                    self.capture_system._should_start_recording = True

            ep = self.episode_manager.get_current_episode()
            self._push_log(f"Recording Episode {ep}...", 'record')
        except Exception as e:
            self._push_log(f"Error starting recording: {e}", 'error')
            self.recording_state = 'waiting'
            self.is_recording_episode = False
