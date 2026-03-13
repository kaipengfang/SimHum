"""
Hand Tracker Module
Hand data processing for GUI visualization
"""
import numpy as np
import threading


class HandTracker:
    """Track hand data and provide real-time updates"""

    def __init__(self, config, tv):
        """
        Initialize hand tracker

        Args:
            config: CaptureConfig instance
            tv: OpenTeleVision instance
        """
        self.config = config
        self.tv = tv
        self.freq = config.freq

        # Real-time hand data (for GUI visualization)
        self.current_head_mat = None
        self.current_left_wrist_mat = None
        self.current_right_wrist_mat = None
        self.current_left_hand_keypoints = None
        self.current_right_hand_keypoints = None
        self.hand_data_lock = threading.Lock()

    def get_hand_actions(self):
        """
        Get hand action data from TeleVision

        Returns:
            tuple or None: (head_mat, left_wrist_mat, right_wrist_mat, left_keypoints, right_keypoints)
        """
        try:
            processed_mat = self.tv.processor.process(self.tv)
            return processed_mat
        except Exception as e:
            print(f"Error getting hand actions: {e}")
            return None

    def update_real_time_data(self, processed_mat):
        """
        Update real-time hand data (for GUI visualization)

        Args:
            processed_mat: Processed hand data from TeleVision
        """
        try:
            if processed_mat and len(processed_mat) >= 5:
                with self.hand_data_lock:
                    self.current_head_mat = processed_mat[0] if len(processed_mat) > 0 else None
                    self.current_left_wrist_mat = processed_mat[1] if len(processed_mat) > 1 else None
                    self.current_right_wrist_mat = processed_mat[2] if len(processed_mat) > 2 else None
                    self.current_left_hand_keypoints = processed_mat[3] if len(processed_mat) > 3 else None
                    self.current_right_hand_keypoints = processed_mat[4] if len(processed_mat) > 4 else None
        except Exception:
            pass

    def get_real_time_data(self):
        """
        Get current real-time hand data (thread-safe)

        Returns:
            dict: Dictionary containing current hand data
        """
        with self.hand_data_lock:
            return {
                'head_mat': self.current_head_mat,
                'left_wrist_mat': self.current_left_wrist_mat,
                'right_wrist_mat': self.current_right_wrist_mat,
                'left_keypoints': self.current_left_hand_keypoints,
                'right_keypoints': self.current_right_hand_keypoints
            }
