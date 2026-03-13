"""
Dataset Module
Dataset management and storage
"""
import cv2
import numpy as np
from ..quality import EpisodeDataQualityChecker
from .hdf5_writer import HDF5Writer


class IntegratedDataset:
    """Integrated dataset class, saves both hand actions and image data (head camera only)"""
    
    def __init__(self, path, freq=30):
        self.path = path
        self.freq = freq  # Capture frequency, used to calculate frames to discard
        self.data_dict = {
            '/obs/timestamp': [],
            '/action/cmd/head_mat': [],
            '/action/cmd/rel_left_wrist_mat': [],
            '/action/cmd/rel_right_wrist_mat': [],
            '/action/cmd/rel_left_hand_keypoints': [],
            '/action/cmd/rel_right_hand_keypoints': [],
            '/observation/image/head': []       # Head camera image
        }
        self.hdf5_writer = HDF5Writer(freq=freq)

    def insert(self,
               timestamp,
               head_mat,
               rel_left_wrist_mat,
               rel_right_wrist_mat,
               rel_left_hand_keypoints,
               rel_right_hand_keypoints,
               head_image=None):
        """Insert a complete data entry (action + head image)"""
        self.data_dict['/obs/timestamp'].append(timestamp)
        self.data_dict['/action/cmd/head_mat'].append(head_mat)
        self.data_dict['/action/cmd/rel_left_wrist_mat'].append(rel_left_wrist_mat)
        self.data_dict['/action/cmd/rel_right_wrist_mat'].append(rel_right_wrist_mat)
        self.data_dict['/action/cmd/rel_left_hand_keypoints'].append(rel_left_hand_keypoints)
        self.data_dict['/action/cmd/rel_right_hand_keypoints'].append(rel_right_hand_keypoints)

        # Compress and save head image
        if head_image is not None:
            ret, encoded = cv2.imencode('.jpg', head_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret:
                self.data_dict['/observation/image/head'].append(encoded.tobytes())
            else:
                self.data_dict['/observation/image/head'].append(b'')
                print("Warning: Failed to encode head image")
        else:
            self.data_dict['/observation/image/head'].append(b'')
            print("Warning: Head image is None")
    
    def save_to_hdf5(self, description, embodiment, log_callback=None):
        """Save data to HDF5 file, automatically discard last 3 seconds to avoid saving end gesture, and process EEF data"""
        return self.hdf5_writer.save_to_hdf5(
            self.path,
            self.data_dict,
            description,
            embodiment,
            log_callback
        )
    
    def _save_preview_video(self, frames_to_keep):
        """Save preview video (delegated to HDF5Writer)"""
        return self.hdf5_writer._save_preview_video(self.path, self.data_dict, frames_to_keep)
    
    def get_data_count(self):
        """Get current data count"""
        return len(self.data_dict['/obs/timestamp'])
