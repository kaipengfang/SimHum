"""
Recording Controller Module
Recording control logic
"""
import os
import datetime
from multiprocessing import Manager, Event
from ..dataset import IntegratedDataset


class RecordingController:
    """Control recording start/stop and episode management"""

    def __init__(self, config):
        """
        Initialize recording controller

        Args:
            config: CaptureConfig instance
        """
        self.config = config
        self.freq = config.freq
        self.path = config.task_path
        self.description = config.description
        self.if_record = config.if_record

        # Episode management
        self.episode = 0
        self.dataset = None

        # Recording state (shared across processes)
        self.manager = Manager()
        self.control_dict = self.manager.dict()
        self.control_dict['is_recording'] = False
        self.control_dict['path'] = ""
        self.control_dict['episode'] = 0
        self.toggle_recording = Event()

        # Callbacks
        self.gui_log_callback = None

    def set_gui_log_callback(self, callback):
        """Set GUI log callback function"""
        self.gui_log_callback = callback

    def is_recording(self):
        """Check if currently recording"""
        return self.control_dict["is_recording"]

    def get_status(self):
        """
        Get current recording status

        Returns:
            dict: Status information
        """
        return {
            'is_recording': self.control_dict["is_recording"],
            'episode': self.episode,
            'path': self.control_dict.get("path", ""),
            'data_count': self.dataset.get_data_count() if self.dataset else 0
        }

    def manual_start_recording(self):
        """
        Manually start recording

        Returns:
            bool: True if recording started successfully
        """
        if not self.control_dict["is_recording"]:
            # Create episode path
            episode_path = os.path.join(self.path, f"episode_{self.episode}")
            os.makedirs(self.path, exist_ok=True)

            self.control_dict["path"] = episode_path
            self.control_dict["episode"] = self.episode
            self.control_dict["is_recording"] = True

            # Cleanup any previously existing dataset
            if hasattr(self, 'dataset') and self.dataset:
                del self.dataset

            # Create new dataset object
            self.dataset = IntegratedDataset(self.control_dict["path"], self.freq)

            print(f"Manual recording started. Data will be saved to: {episode_path}")
            print(f"Starting episode {self.episode}")
            return True
        return False

    def manual_stop_recording(self):
        """
        Manually stop recording

        Returns:
            tuple: (success, quality_result)
        """
        if self.control_dict["is_recording"]:
            # Create log callback function
            def log_to_gui(msg):
                if self.gui_log_callback:
                    self.gui_log_callback(msg)
                else:
                    print(msg)

            # Save data to HDF5
            success, quality_result = self.dataset.save_to_hdf5(
                self.description,
                "integrated_action_image_capture",
                log_to_gui
            )

            self.control_dict["is_recording"] = False

            if success:
                print(f"✅ Manual recording stopped. Data saved to: {self.control_dict['path']}")
                # Increment episode number for next recording
                self.episode += 1
                return True, quality_result
            else:
                print(f"❌ Episode {self.episode} discarded due to data quality issues")
                print("💡 Please check VR tracking status, ensure hands are moving normally")
                return False, quality_result
        return False, None

    def drop_current_episode(self):
        """
        Discard current episode (manual call)

        Returns:
            bool: True if episode was dropped
        """
        if self.control_dict["is_recording"]:
            print("DROP manually triggered! Discarding current recording...")
            self.control_dict["is_recording"] = False

            # Reset dataset, clear current data
            if hasattr(self, 'dataset') and self.dataset:
                del self.dataset

            # Don't save any data, don't increment episode number
            print(f"Episode {self.episode} discarded. Next recording will use the same episode number.")
            return True
        else:
            print("No active recording to drop.")
            return False

    def insert_data(self, timestamp, head_mat, left_wrist_mat, right_wrist_mat,
                   left_keypoints, right_keypoints, head_image):
        """
        Insert data into current dataset

        Args:
            timestamp: Timestamp
            head_mat: Head matrix
            left_wrist_mat: Left wrist matrix
            right_wrist_mat: Right wrist matrix
            left_keypoints: Left hand keypoints
            right_keypoints: Right hand keypoints
            head_image: Head camera image
        """
        if self.is_recording() and self.dataset:
            self.dataset.insert(
                timestamp,
                head_mat,
                left_wrist_mat,
                right_wrist_mat,
                left_keypoints,
                right_keypoints,
                head_image
            )
