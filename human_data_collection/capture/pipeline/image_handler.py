"""
Image Handler Module
Image server connection and image processing
"""
import zmq
import cv2
import numpy as np
import threading
from multiprocessing import shared_memory
from camera.client import ImageClient


class ImageHandler:
    """Handle image server connection and image processing"""

    def __init__(self, config):
        """
        Initialize image handler

        Args:
            config: CaptureConfig instance
        """
        self.config = config
        self.image_server_address = config.image_server_address
        self.image_server_port = config.image_server_port

        # Image buffer
        self.current_head_image = None
        self.image_lock = threading.Lock()

        # Image client setup
        self.img_client_shape = (480, 640, 3)
        self.img_client_shm = None
        self.img_client_array = None
        self.image_client = None
        self.image_thread = None

    def test_connection(self):
        """Test image server connection using ZeroMQ"""
        try:
            print(f"Testing ZeroMQ connection to {self.image_server_address}:{self.image_server_port}")
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.connect(f"tcp://{self.image_server_address}:{self.image_server_port}")
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout

            # Try to receive a message to test connection
            try:
                message = socket.recv()
                print(f"✅ ZeroMQ connection successful, received message of {len(message)} bytes")
                result = True
            except zmq.Again:
                print("⚠️  ZeroMQ connection timeout - server may not be sending data")
                result = False
            except Exception as e:
                print(f"❌ ZeroMQ connection error: {e}")
                result = False

            socket.close()
            context.term()
            return result
        except Exception as e:
            print(f"Connection test error: {e}")
            return False

    def setup(self):
        """Set up image client (head camera only)"""
        # Create shared memory for image client
        try:
            self.img_client_shm = shared_memory.SharedMemory(
                name="img_client_shm",
                create=True,
                size=np.prod(self.img_client_shape) * np.uint8().itemsize
            )
        except FileExistsError:
            existing_shm = shared_memory.SharedMemory(name="img_client_shm")
            existing_shm.unlink()
            self.img_client_shm = shared_memory.SharedMemory(
                name="img_client_shm",
                create=True,
                size=np.prod(self.img_client_shape) * np.uint8().itemsize
            )

        self.img_client_array = np.ndarray(
            self.img_client_shape,
            dtype=np.uint8,
            buffer=self.img_client_shm.buf
        )

        # Create image client
        self.image_client = ImageClient(
            tv_img_shape=self.img_client_shape,
            tv_img_shm_name=self.img_client_shm.name,
            image_show=False,
            server_address=self.image_server_address,
            port=self.image_server_port,
            Unit_Test=False
        )

        # Start image receive thread
        self.image_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.image_thread.start()

    def _receive_loop(self):
        """Image receive loop (runs in separate thread)"""
        try:
            self.image_client.receive_process()
        except Exception as e:
            print(f"Image receive loop error: {e}")

    def get_current_image(self):
        """
        Get current head camera image

        Returns:
            np.ndarray or None: Current image, or None if not available
        """
        with self.image_lock:
            if self.img_client_array is not None:
                # Copy image from shared memory
                image = self.img_client_array.copy()
                # Convert BGR to RGB if needed
                self.current_head_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return self.current_head_image
            return None

    def cleanup(self):
        """Clean up resources"""
        # 1. Stop receive thread (close socket to unblock recv)
        if self.image_client is not None:
            try:
                self.image_client.stop()
            except Exception:
                pass
        if self.image_thread is not None and self.image_thread.is_alive():
            self.image_thread.join(timeout=2.0)

        # 2. Release shared memory
        if self.img_client_shm is not None:
            try:
                self.img_client_shm.close()
                self.img_client_shm.unlink()
            except Exception as e:
                print(f"Error cleaning up image handler: {e}")
