"""
Episode Data Quality Checker Module
Check episode data quality, ensure no prolonged static hand data
"""
import numpy as np
from .gripper_calculator import GripperCalculator


class EpisodeDataQualityChecker:
    """Check episode data quality, ensure no prolonged static hand data, and validate gripper action patterns"""

    def __init__(self, fps=30, threshold=0.0005, consecutive_frames=30):
        """
        Args:
            fps: Video frame rate
            threshold: Coordinate change threshold (meters), values below this are considered static
            consecutive_frames: Consecutive static frame count threshold
        """
        self.fps = fps
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.gripper_calculator = GripperCalculator()

    # Delegate gripper calculation methods to GripperCalculator
    def calculate_single_frame_gripper(self, fingertips: np.ndarray) -> float:
        """Calculate gripper value for single frame"""
        return self.gripper_calculator.calculate_single_frame_gripper(fingertips)

    def calculate_gripper_from_fingertips(self, fingertips: np.ndarray) -> np.ndarray:
        """Calculate gripper values from fingertips data"""
        return self.gripper_calculator.calculate_gripper_from_fingertips(fingertips)

    def calculate_gripper_dual_hands(self, left_fingertips: np.ndarray, right_fingertips: np.ndarray):
        """Calculate gripper values for both hands"""
        return self.gripper_calculator.calculate_gripper_dual_hands(left_fingertips, right_fingertips)

    def validate_gripper_changepoints(self, gripper_values: np.ndarray, pen: float = 1.0) -> dict:
        """Validate gripper changepoints"""
        return self.gripper_calculator.validate_gripper_changepoints(gripper_values, pen)

    def validate_dual_hand_grippers(self, left_gripper: np.ndarray, right_gripper: np.ndarray, pen: float = 1.0) -> dict:
        """Validate both hands' gripper patterns"""
        return self.gripper_calculator.validate_dual_hand_grippers(left_gripper, right_gripper, pen)

    def check_episode_quality(self, data_dict):
        """
        Check episode data quality

        Args:
            data_dict: Dictionary containing hand data

        Returns:
            dict: Contains check results and detailed information
        """
        try:
            # Get left and right wrist position data
            left_wrist_mat = data_dict.get('/action/cmd/rel_left_wrist_mat', [])
            right_wrist_mat = data_dict.get('/action/cmd/rel_right_wrist_mat', [])

            if not left_wrist_mat or not right_wrist_mat:
                return {
                    'is_valid': False,
                    'issues': ['Missing hand data'],
                    'details': {'total_frames': 0}
                }

            # Convert to numpy arrays
            left_wrist_mat = np.array(left_wrist_mat)
            right_wrist_mat = np.array(right_wrist_mat)

            # Get total frame count
            total_frames = min(len(left_wrist_mat), len(right_wrist_mat))

            if total_frames < self.fps * 3:  # Need at least 3 seconds of data
                return {
                    'is_valid': False,
                    'issues': [f'Insufficient data length, only {total_frames} frames ({total_frames/self.fps:.1f}s)'],
                    'details': {'total_frames': total_frames}
                }

            # Ignore first 1 second and last 4 seconds (last 1s gesture + 3s to be discarded)
            start_frame = self.fps  # Skip first 1 second
            # Skip last 4 seconds: 1s gesture + 3s data to be discarded
            end_frame = total_frames - (4 * self.fps)

            if end_frame <= start_frame:
                return {
                    'is_valid': False,
                    'issues': ['Insufficient data length for valid check'],
                    'details': {'total_frames': total_frames}
                }

            # Extract position data (x, y, z) from 4x4 matrices
            left_positions = left_wrist_mat[start_frame:end_frame, :3, 3]
            right_positions = right_wrist_mat[start_frame:end_frame, :3, 3]

            # Check for static data
            issues = []
            details = {
                'total_frames': total_frames,
                'checked_frames': end_frame - start_frame,
                'start_frame': start_frame,
                'end_frame': end_frame
            }

            # Check left hand
            left_static_info = self._check_static_hand(left_positions, 'left')
            if not left_static_info['is_valid']:
                issues.append(left_static_info['issue'])
            details['left_hand'] = left_static_info

            # Check right hand
            right_static_info = self._check_static_hand(right_positions, 'right')
            if not right_static_info['is_valid']:
                issues.append(right_static_info['issue'])
            details['right_hand'] = right_static_info

            return {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'details': details
            }

        except Exception as e:
            return {
                'is_valid': False,
                'issues': [f'Quality check error: {str(e)}'],
                'details': {}
            }

    def _check_static_hand(self, positions, hand_name):
        """
        Check if hand has prolonged static periods

        Args:
            positions: (N, 3) position array
            hand_name: 'left' or 'right'

        Returns:
            dict: Check result
        """
        # Calculate frame-to-frame displacement
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        # Find static frames (displacement below threshold)
        static_frames = displacements < self.threshold

        # Find consecutive static frame sequences
        max_consecutive_static = 0
        current_consecutive = 0
        static_segments = []
        segment_start = None

        for i, is_static in enumerate(static_frames):
            if is_static:
                if current_consecutive == 0:
                    segment_start = i
                current_consecutive += 1
                max_consecutive_static = max(max_consecutive_static, current_consecutive)
            else:
                if current_consecutive >= self.consecutive_frames:
                    static_segments.append({
                        'start': segment_start,
                        'end': i,
                        'duration': current_consecutive
                    })
                current_consecutive = 0

        # Check last segment
        if current_consecutive >= self.consecutive_frames:
            static_segments.append({
                'start': segment_start,
                'end': len(static_frames),
                'duration': current_consecutive
            })

        is_valid = max_consecutive_static < self.consecutive_frames

        result = {
            'is_valid': is_valid,
            'max_consecutive_static': max_consecutive_static,
            'static_segments': static_segments,
            'static_duration_seconds': max_consecutive_static / self.fps
        }

        if not is_valid:
            result['issue'] = (
                f'{hand_name.capitalize()} hand has {max_consecutive_static} consecutive static frames '
                f'({max_consecutive_static/self.fps:.1f}s), exceeds threshold {self.consecutive_frames} frames '
                f'({self.consecutive_frames/self.fps:.1f}s)'
            )

        return result
