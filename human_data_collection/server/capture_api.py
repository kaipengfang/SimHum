"""
Capture System Management API
Handles capture system startup, shutdown, and status queries
"""
import os
import threading
import time
import datetime


class CaptureAPI:
    """Capture system management interface"""

    def start_capture(self, save_path='', description='test'):
        """Start capture system, enter keyboard control waiting state"""
        try:
            from capture import IntegratedActionImageCapture

            if not save_path:
                save_path = 'data/integrated_recordings'

            abs_path = os.path.abspath(save_path)
            try:
                os.makedirs(abs_path, exist_ok=True)
            except Exception:
                abs_path = os.path.abspath('./recordings')
                os.makedirs(abs_path, exist_ok=True)
                save_path = './recordings'

            self._push_log(f"Starting capture system, save path: {abs_path}", 'info')

            args = {
                'des': f"gui_capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'no_record': False,
                'description': description,
                'path': save_path,
                'image_server': 'localhost',
                'image_port': 5555,
            }

            self.capture_system = IntegratedActionImageCapture(
                description=description,
                args=args,
                freq=30,
                path=save_path,
                image_server_address='localhost',
                image_server_port=5555,
            )

            self.capture_system.set_gui_log_callback(
                lambda msg: self._push_log(msg, 'info')
            )
            self.episode_manager.sync_to_capture_system(self.capture_system)

            def run_capture():
                try:
                    self.is_capture_running = True
                    frame_count = 0
                    start_time = time.time()
                    while self.is_capture_running:
                        success = self.capture_system.step()
                        if success:
                            frame_count += 1
                            self._update_realtime_data()
                        # Frame rate control: wait until the next frame's scheduled time
                        next_frame_time = start_time + frame_count / 30.0
                        sleep_time = next_frame_time - time.time()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                except Exception as e:
                    self._push_log(f"Capture thread error: {e}", 'error')
                finally:
                    self.is_capture_running = False

            self.capture_thread = threading.Thread(target=run_capture, daemon=True)
            self.capture_thread.start()

            self.recording_state = 'waiting'
            self.is_recording_episode = False
            self.episode_manager.reset()
            self.data_buffer.clear()

            self._push_log("Capture system started, waiting for keyboard control", 'success')
            return {'ok': True}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._push_log(f"Failed to start capture: {e}", 'error')
            return {'ok': False, 'error': str(e)}

    def stop_capture(self):
        """Stop capture system; heavy cleanup runs in background thread, returns immediately"""
        try:
            if self.is_countdown_active:
                self._stop_countdown()

            self.is_capture_running = False
            self.recording_state = 'waiting'
            self.is_recording_episode = False

            if self.capture_system:
                try:
                    if (hasattr(self.capture_system, 'is_recording') and
                            self.capture_system.is_recording()):
                        if hasattr(self.capture_system, 'manual_stop_recording'):
                            self.capture_system.manual_stop_recording()
                except Exception:
                    pass

            # Move heavy cleanup to background thread to avoid blocking HTTP response
            capture_system = self.capture_system
            capture_thread = self.capture_thread
            self.capture_system = None
            self.capture_thread = None

            def _cleanup_async():
                if capture_thread and capture_thread.is_alive():
                    capture_thread.join(timeout=3.0)
                if capture_system:
                    try:
                        capture_system.cleanup()
                    except Exception:
                        pass

            threading.Thread(target=_cleanup_async, daemon=True).start()

            self.data_buffer.clear()
            self._push_log("Capture system stopped", 'info')
            return {'ok': True}
        except Exception as e:
            self._push_log(f"Failed to stop capture: {e}", 'error')
            return {'ok': False, 'error': str(e)}

    def browse_path(self):
        """Open system file dialog (pywebview native support)"""
        try:
            import webview
            dirs = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
            if dirs:
                return {'ok': True, 'path': dirs[0]}
            return {'ok': True, 'path': None}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def set_episode(self, value):
        """Manually set episode number"""
        try:
            n = int(value)
            if n >= 0:
                self.episode_manager.set_episode(n)
                self.episode_manager.sync_to_capture_system(self.capture_system)
            return {'ok': True, 'episode': self.episode_manager.get_current_episode()}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    # ── Private Methods ──────────────────────────────────────────

    def _update_realtime_data(self):
        """Pull latest xyz from capture system and write to data_buffer"""
        try:
            if not self.capture_system:
                return
            if hasattr(self.capture_system, 'hand_data_lock'):
                with self.capture_system.hand_data_lock:
                    left_mat = self.capture_system.current_left_wrist_mat
                    right_mat = self.capture_system.current_right_wrist_mat
                    head_mat = self.capture_system.current_head_mat
            else:
                left_mat = getattr(self.capture_system, 'current_left_wrist_mat', None)
                right_mat = getattr(self.capture_system, 'current_right_wrist_mat', None)
                head_mat = getattr(self.capture_system, 'current_head_mat', None)

            if left_mat is None or right_mat is None or head_mat is None:
                return
            if not all(m.shape == (4, 4) for m in [left_mat, right_mat, head_mat]):
                return

            head_xyz = head_mat[:3, 3]
            left_xyz = head_xyz + left_mat[:3, 3]
            right_xyz = head_xyz + right_mat[:3, 3]

            now = time.time()
            if not hasattr(self, '_last_data_ts') or now - self._last_data_ts >= 0.033:
                self.data_buffer.add_data(now, left_xyz, right_xyz)
                self._last_data_ts = now
        except Exception:
            pass

