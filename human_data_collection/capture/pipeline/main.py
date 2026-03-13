"""
Main Capture Module
Main class integrating all functionality
"""
import time
import numpy as np
from multiprocessing import shared_memory
from opentv.TeleVision import OpenTeleVision
from ..utils import check_adb_setup
from .config import CaptureConfig
from .image_handler import ImageHandler
from .hand_tracker import HandTracker
from .recording_controller import RecordingController


class IntegratedActionImageCapture:
    """Integrated action and image capture system"""

    def __init__(
        self,
        description,
        args,
        freq=30,
        path="data/integrated_recordings",
        root_path=None,
        image_server_address="localhost",
        image_server_port=5555
    ):
        """
        Initialize integrated capture system

        Args:
            description: Task description
            args: Arguments dictionary
            freq: Capture frequency (Hz)
            path: Base save path
            root_path: Root path (deprecated)
            image_server_address: Image server IP address
            image_server_port: Image server port
        """
        # Initialize configuration
        self.config = CaptureConfig(
            description=description,
            args=args,
            freq=freq,
            path=path,
            image_server_address=image_server_address,
            image_server_port=image_server_port
        )

        # Check ADB setup
        check_adb_setup()

        # Initialize shared memory for TeleVision
        img_shape = self.config.get_image_shape()
        try:
            self.shm = shared_memory.SharedMemory(
                name="integrated_capture",
                create=True,
                size=np.prod(img_shape) * np.uint8().itemsize
            )
        except FileExistsError:
            existing_shm = shared_memory.SharedMemory(name="integrated_capture")
            existing_shm.unlink()
            self.shm = shared_memory.SharedMemory(
                name="integrated_capture",
                create=True,
                size=np.prod(img_shape) * np.uint8().itemsize
            )

        # Initialize OpenTeleVision
        self.tv = OpenTeleVision(
            self.config.resolution_cropped,
            self.shm.name,
            cert_file=None,
            key_file=None
        )

        # Initialize components
        self.image_handler = ImageHandler(self.config)
        self.hand_tracker = HandTracker(self.config, self.tv)
        self.recording_controller = RecordingController(self.config)

        # Test image server connection
        if not self.image_handler.test_connection():
            print(f"⚠️  Warning: Cannot connect to image server {image_server_address}:{image_server_port}")
            print("   Images will not be captured. Check if image_server is running.")

        # Setup image client
        self.image_handler.setup()

        print("Integrated Action-Image Capture System initialized")
        print("Resolution cropped: ", self.config.resolution_cropped)
        print("Shared memory name: ", self.shm.name)
        print("Data save path: ", self.config.task_path)
        print(f"Image server: {image_server_address}:{image_server_port}")

    # Expose hand tracker data for GUI (delegation properties)
    @property
    def hand_data_lock(self):
        """Delegate to hand_tracker.hand_data_lock"""
        return self.hand_tracker.hand_data_lock

    @property
    def current_head_mat(self):
        """Delegate to hand_tracker.current_head_mat"""
        return self.hand_tracker.current_head_mat

    @property
    def current_left_wrist_mat(self):
        """Delegate to hand_tracker.current_left_wrist_mat"""
        return self.hand_tracker.current_left_wrist_mat

    @property
    def current_right_wrist_mat(self):
        """Delegate to hand_tracker.current_right_wrist_mat"""
        return self.hand_tracker.current_right_wrist_mat

    # Delegate methods to components

    def set_gui_log_callback(self, callback):
        """Set GUI log callback function"""
        self.recording_controller.set_gui_log_callback(callback)

    def is_recording(self):
        """Check if currently recording"""
        return self.recording_controller.is_recording()

    def get_status(self):
        """Get current recording status"""
        return self.recording_controller.get_status()

    def manual_start_recording(self):
        """Manually start recording"""
        return self.recording_controller.manual_start_recording()

    def manual_stop_recording(self):
        """Manually stop recording"""
        return self.recording_controller.manual_stop_recording()

    def drop_current_episode(self):
        """Discard current episode"""
        return self.recording_controller.drop_current_episode()

    def get_hand_actions(self):
        """Get hand action data"""
        return self.hand_tracker.get_hand_actions()

    def update_real_time_hand_data(self, processed_mat):
        """Update real-time hand data"""
        self.hand_tracker.update_real_time_data(processed_mat)

    def get_real_time_hand_data(self):
        """Get current real-time hand data"""
        return self.hand_tracker.get_real_time_data()

    def step(self):
        """
        Main step function - process one frame

        Returns:
            bool: True if step successful
        """
        try:
            # Get hand actions
            processed_mat = self.get_hand_actions()
            if processed_mat is None:
                return False

            # Update real-time hand data (for GUI)
            self.update_real_time_hand_data(processed_mat)

            # Post-step callback (recording logic)
            self.post_step_callback(processed_mat)

            return True
        except Exception as e:
            print(f"Error in step: {e}")
            return False

    def post_step_callback(self, processed_mat):
        """
        Post-step callback - handle recording logic

        Args:
            processed_mat: Processed hand data
        """
        if not self.config.if_record:
            return

        # If recording, save data
        if self.is_recording():
            timestamp = time.time()
            head_mat = processed_mat[0]
            left_wrist_mat = processed_mat[1]
            right_wrist_mat = processed_mat[2]
            left_keypoints = processed_mat[3]
            right_keypoints = processed_mat[4]

            # Get current head image
            head_image = self.image_handler.get_current_image()

            # Insert data
            self.recording_controller.insert_data(
                timestamp,
                head_mat,
                left_wrist_mat,
                right_wrist_mat,
                left_keypoints,
                right_keypoints,
                head_image
            )

    def cleanup(self):
        """Clean up resources"""
        try:
            # 1. Stop image handler first, stop writing to shared memory
            self.image_handler.cleanup()

            # 2. Terminate OpenTeleVision subprocess; must wait for full exit before releasing shared memory
            if hasattr(self, 'tv') and hasattr(self.tv, 'process'):
                proc = self.tv.process
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=3.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2.0)  # Brief wait after SIGKILL

            # 3. Subprocess is dead, safe to release shared memory
            if hasattr(self, 'shm'):
                self.shm.close()
                self.shm.unlink()

            # 4. Shut down RecordingController's Manager subprocess
            if hasattr(self, 'recording_controller'):
                try:
                    self.recording_controller.manager.shutdown()
                except Exception:
                    pass

            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")
