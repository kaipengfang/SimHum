"""
Image Server Management API
Handles server startup, shutdown, and connection detection
"""
import threading
import subprocess
import zmq


class ServerAPI:
    """Image server management interface"""

    def toggle_server(self):
        """Toggle server state (connect/disconnect/start/stop)"""
        try:
            if not self.is_image_server_running and not self.server_connection_tested:
                return self._start_image_server()
            elif self.server_connection_tested:
                return self._disconnect_server()
            else:
                return self._stop_image_server()
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def connect_to_existing_server(self):
        """Try to connect to an existing image server"""
        try:
            connected = self._test_zmq_connection()
            if connected:
                self.server_connection_tested = True
                self.is_image_server_running = False  # Not started by us
                self._push_log("Successfully connected to the existing image server", 'success')
                return {'ok': True, 'connected': True, 'status': 'Connected'}
            else:
                self._push_log("No existing image server found", 'info')
                return {'ok': True, 'connected': False, 'status': 'Server Not Found'}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def get_server_status(self):
        """Return server connection status"""
        if self.server_connection_tested:
            return {'ok': True, 'status': 'connected', 'text': 'Connected'}
        elif self.is_image_server_running:
            return {'ok': True, 'status': 'connecting', 'text': 'Connecting...'}
        else:
            return {'ok': True, 'status': 'disconnected', 'text': 'Not connected'}

    def check_server_connection(self):
        """Actively test connection and update status (called by JS)"""
        try:
            connected = self._test_zmq_connection()
            if connected:
                self.server_connection_tested = True
                self._push_log("Image server connection successful", 'success')
                return {'ok': True, 'connected': True}
            else:
                self.server_connection_tested = False
                self._push_log("Image server connection failed", 'error')
                return {'ok': True, 'connected': False}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    # ── Private Methods ──────────────────────────────────────────

    def _start_image_server(self):
        """Start image server thread"""
        try:
            import camera.server as _cam_server
            from camera.server import ImageServer
            self._push_log("Starting the image server...", 'info')

            _cam_server.TERMINAL_MODE = True

            config = {
                'fps': 30,
                'head_camera_type': 'opencv',
                'head_camera_image_shape': [480, 640],
                'head_camera_id_numbers': [0],  # <- your /dev/video* number
            }

            def run_server():
                try:
                    server = ImageServer(config, port=5555, Unit_Test=False)
                    self.image_server_process = server
                    self.is_image_server_running = True
                    self._push_log("Image server started successfully", 'success')
                    server.send_process()
                except Exception as e:
                    if not self.is_shutting_down:
                        self._push_log(f"Image server error: {e}", 'error')
                    self.is_image_server_running = False

            self.image_server_thread = threading.Thread(target=run_server, daemon=True)
            self.image_server_thread.start()

            checker = threading.Thread(target=self._auto_connect_checker, daemon=True)
            checker.start()

            return {'ok': True, 'status': 'starting'}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def _stop_image_server(self):
        """Stop image server"""
        try:
            self.is_image_server_running = False

            # Notify send_process to exit loop first, then wait for thread to finish
            if self.image_server_process:
                if hasattr(self.image_server_process, 'stop'):
                    self.image_server_process.stop()
                self.image_server_process = None

            if self.image_server_thread and self.image_server_thread.is_alive():
                self.image_server_thread.join(timeout=3.0)
                if self.image_server_thread.is_alive():
                    self._push_log("Image server thread did not exit in time, force killing port", 'warning')

            self._kill_processes_on_port(self.server_port)
            self.server_connection_tested = False
            self._push_log("Image server has been stopped", 'info')
            return {'ok': True}
        except Exception as e:
            self._push_log(f"Failed to stop image server: {e}", 'error')
            return {'ok': False, 'error': str(e)}

    def _disconnect_server(self):
        """Disconnect from server (without stopping the server process)"""
        self.server_connection_tested = False
        # If the server was started by us, stop it too
        if self.is_image_server_running:
            return self._stop_image_server()
        self._push_log("Server connection has been disconnected", 'info')
        return {'ok': True}

    def _auto_connect_checker(self):
        """After server startup, poll ZMQ until connection succeeds (max 30 seconds)"""
        import time
        for _ in range(30):
            time.sleep(1)  # Wait first, then check — avoid exiting before server thread is ready
            if self.is_shutting_down:
                return
            if self._test_zmq_connection():
                self.server_connection_tested = True
                self._push_log("Image server connection confirmed", 'success')
                return
        self._push_log("Image server connection timeout, please check the camera", 'warning')

    def _test_zmq_connection(self):
        """Test server connection via ZMQ"""
        try:
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.LINGER, 0)  # Prevent context.term() from blocking forever
            socket.connect("tcp://localhost:5555")
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            try:
                socket.recv()
                return True
            except zmq.Again:
                return False
            finally:
                socket.close()
                context.term()
        except Exception:
            return False

    def _kill_processes_on_port(self, port):
        """Kill processes occupying a port"""
        try:
            result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True, timeout=5)
            for pid in result.stdout.strip().split('\n'):
                if pid.strip():
                    try:
                        subprocess.run(['kill', '-9', pid.strip()], timeout=5)
                    except Exception:
                        pass
        except Exception:
            pass

    def cleanup_existing_processes(self):
        """Clean up leftover processes on startup"""
        try:
            # Clean up ZMQ image server port
            if self._is_port_occupied(self.server_port):
                self._kill_processes_on_port(self.server_port)
            # Clean up OpenTeleVision vuer port
            if self._is_port_occupied(8012):
                self._kill_processes_on_port(8012)
            self._kill_image_server_processes()
            # Clean up leftover shared memory to prevent OOM or port conflicts
            self._cleanup_shared_memory()
        except Exception:
            pass

    def _cleanup_shared_memory(self):
        """Clean up leftover shared memory segments"""
        try:
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(name="integrated_capture")
            shm.close()
            shm.unlink()
        except Exception:
            pass  # Ignore if not exists

    def _is_port_occupied(self, port):
        """Check if a port is occupied"""
        try:
            result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True, timeout=5)
            return f':{port}' in result.stdout
        except Exception:
            return False

    def _kill_image_server_processes(self):
        """Clean up leftover image server processes"""
        import os
        current_pid = os.getpid()
        for pattern in ['camera/server.py', 'camera.server', 'ImageServer']:
            try:
                result = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True, timeout=5)
                for pid in result.stdout.strip().split('\n'):
                    pid = pid.strip()
                    if pid and pid.isdigit() and int(pid) != current_pid:
                        try:
                            subprocess.run(['kill', '-9', pid], timeout=5)
                        except Exception:
                            pass
            except Exception:
                pass
