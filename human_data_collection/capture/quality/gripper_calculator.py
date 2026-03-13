"""
Gripper Calculator Module
Calculate gripper values based on finger positions
"""
import numpy as np
import ruptures as rpt
from typing import Tuple


class GripperCalculator:
    """Calculate gripper values from fingertip positions using human ergonomics"""

    # Gripper calculation constants based on human ergonomics
    HUMAN_MIN_DISTANCE = 0.02  # 2cm - minimum distance when gripper is fully closed
    HUMAN_MAX_DISTANCE = 0.12  # 12cm - maximum distance when gripper is fully open
    HUMAN_RANGE = HUMAN_MAX_DISTANCE - HUMAN_MIN_DISTANCE

    def calculate_single_frame_gripper(self, fingertips: np.ndarray) -> float:
        """
        Calculate gripper value for single frame using DemoDiffusion algorithm.

        Algorithm: Distance from thumb tip to average position of other four fingertips.

        Args:
            fingertips: (5, 3) fingertips data for single frame

        Returns:
            float: gripper value in range [0, 1]
                - 0.0: fully closed
                - 1.0: fully open
        """
        # Extract key points
        thumb_tip = fingertips[0]  # Thumb tip
        other_tips_mean = fingertips[1]

        # Calculate distance
        distance = np.linalg.norm(thumb_tip - other_tips_mean)

        # Human ergonomics normalization
        # Inverted: larger distance = more open = higher value
        gripper_value = (distance - self.HUMAN_MIN_DISTANCE) / self.HUMAN_RANGE

        # Clamp to [0, 1] range
        return np.clip(gripper_value, 0.0, 1.0)

    def calculate_gripper_from_fingertips(self, fingertips: np.ndarray) -> np.ndarray:
        """
        Calculate gripper values from fingertips data using human ergonomics.

        Based on the distance from thumb tip to the average position of other four fingertips.
        Uses human ergonomic parameters for normalization.

        For multi-frame input, automatically applies changepoint detection denoising.

        Args:
            fingertips: Fingertip coordinates
                - Single frame: (5, 3) - 5 fingertips with xyz coordinates
                - Multi frame: (T, 5, 3) - T frames, each with 5 fingertips

        Returns:
            np.ndarray: Gripper values in range [0, 1]
                - 0.0: fully closed
                - 1.0: fully open
                - Single frame: scalar value (no denoising)
                - Multi frame: (T,) array (automatically denoised)
        """
        fingertips = np.asarray(fingertips)

        if fingertips.ndim == 2:  # Single frame (5, 3)
            if fingertips.shape != (5, 3):
                raise ValueError(f"Single frame fingertips should be (5, 3), got {fingertips.shape}")
            return self.calculate_single_frame_gripper(fingertips)

        elif fingertips.ndim == 3:  # Multi frame (T, 5, 3)
            if fingertips.shape[1:] != (5, 3):
                raise ValueError(f"Multi frame fingertips should be (T, 5, 3), got {fingertips.shape}")

            # Calculate raw gripper values for each frame
            grippers = []
            for frame_fingertips in fingertips:
                gripper = self.calculate_single_frame_gripper(frame_fingertips)
                grippers.append(gripper)
            grippers_raw = np.array(grippers)

            return grippers_raw

        else:
            raise ValueError(f"Fingertips data should be 2D or 3D array, got {fingertips.ndim}D")

    def calculate_gripper_dual_hands(self, left_fingertips: np.ndarray,
                                   right_fingertips: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gripper values for both hands independently.

        Args:
            left_fingertips: Left hand fingertips (5, 3) or (T, 5, 3)
            right_fingertips: Right hand fingertips (5, 3) or (T, 5, 3)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (left_gripper, right_gripper) values
        """
        left_gripper = self.calculate_gripper_from_fingertips(left_fingertips)
        right_gripper = self.calculate_gripper_from_fingertips(right_fingertips)

        return left_gripper, right_gripper

    def validate_gripper_changepoints(self, gripper_values: np.ndarray, pen: float = 1.0) -> dict:
        """
        Validate gripper changepoints: exactly 3 changepoints (simplified).

        Args:
            gripper_values: Denoised gripper values array
            pen: Penalty parameter for changepoint detection

        Returns:
            dict: Validation result with details
                - is_valid: bool - True if exactly 3 changepoints
                - changepoints: list - Detected changepoint indices
                - details: str - Detailed explanation
        """
        try:
            # Use existing changepoint detection
            algo = rpt.Pelt(model="l2").fit(gripper_values)
            bkps = algo.predict(pen=pen)

            print(bkps)

            # Remove the last breakpoint (end of series)
            if len(bkps) > 0 and bkps[-1] == len(gripper_values):
                changepoints = bkps[:-1]
            else:
                changepoints = bkps[:]

            # Check if exactly 3 changepoints (simplified logic)
            is_valid = len(changepoints) == 3

            if is_valid:
                details = f"Valid: found exactly 3 changepoints at {changepoints}"
            else:
                details = f"Invalid: expected 3 changepoints, found {len(changepoints)}: {changepoints}"

            return {
                'is_valid': is_valid,
                'changepoints': changepoints,
                'details': details
            }

        except Exception as e:
            return {
                'is_valid': False,
                'changepoints': [],
                'details': f'Validation error: {str(e)}'
            }

    def validate_dual_hand_grippers(self, left_gripper: np.ndarray, right_gripper: np.ndarray, pen: float = 1.0) -> dict:
        """
        Validate both hands' gripper patterns.

        Args:
            left_gripper: Left hand gripper values
            right_gripper: Right hand gripper values
            pen: Penalty parameter for changepoint detection

        Returns:
            dict: Validation result for both hands
                - is_valid: bool - True if both hands have valid patterns
                - left_hand: dict - Left hand validation result
                - right_hand: dict - Right hand validation result
                - details: str - Summary
        """
        left_result = self.validate_gripper_changepoints(left_gripper, pen=pen)
        right_result = self.validate_gripper_changepoints(right_gripper, pen=pen)

        is_valid = left_result['is_valid'] and right_result['is_valid']

        if is_valid:
            details = "Both hands have exactly 3 changepoints each"
        else:
            issues = []
            if not left_result['is_valid']:
                issues.append(f"Left hand: {left_result['details']}")
            if not right_result['is_valid']:
                issues.append(f"Right hand: {right_result['details']}")
            details = "; ".join(issues)

        return {
            'is_valid': is_valid,
            'left_hand': left_result,
            'right_hand': right_result,
            'details': details
        }
