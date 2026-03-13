"""
Gripper calculation and state vector building utilities

Shared gripper processing functions following DRY principle.
Used by human data processors for fingertip-based gripper calculation.
"""

import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation

from .constants import HUMAN_MIN_DISTANCE, HUMAN_MAX_DISTANCE, HUMAN_RANGE


def denoise_changepoint_detection(grippers, pen=1.0):
    """
    Denoise gripper values using change point detection (ruptures library)

    Args:
        grippers: (T,) array of raw gripper values
        pen: Penalty value for change point detection (default: 1.0)

    Returns:
        Denoised gripper values (T,) array
    """
    import ruptures as rpt

    # Use PELT algorithm with l2 model
    algo = rpt.Pelt(model="l2").fit(grippers)
    bkps = algo.predict(pen=pen)

    # Reconstruct denoised signal
    denoised = np.zeros_like(grippers)
    start_idx = 0
    for end_idx in bkps:
        
        segment_mean = np.mean(grippers[start_idx:end_idx])
        if start_idx != 0:
            changed = denoised[start_idx] - denoised[start_idx - 1]
            denoised[start_idx - 1] = denoised[start_idx] - changed / 3
            denoised[start_idx - 2] = denoised[start_idx - 1] - changed / 3 
        denoised[start_idx:end_idx] = segment_mean
        start_idx = end_idx

    return denoised


def calculate_gripper_from_fingertips(fingertips: np.ndarray) -> np.ndarray:
    """
    Calculate gripper values from fingertips data using human ergonomics.

    Based on the distance from thumb tip to the average position of other four fingertips.
    Uses human ergonomic parameters for normalization.

    For multi-frame input, automatically applies denoising filter (threshold=0.5, lookahead=50)
    to remove measurement noise and preserve true gripper extrema.

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
        return calculate_single_frame_gripper(fingertips)
    
    elif fingertips.ndim == 3:  # Multi frame (T, 5, 3)
        if fingertips.shape[1:] != (5, 3):
            raise ValueError(f"Multi frame fingertips should be (T, 5, 3), got {fingertips.shape}")

        # Calculate raw gripper values for each frame
        grippers = []
        for frame_fingertips in fingertips:
            gripper = calculate_single_frame_gripper(frame_fingertips)
            grippers.append(gripper)
        grippers_raw = np.array(grippers)

        # Apply denoising filter with fixed parameters
        grippers_denoised = denoise_changepoint_detection(
            grippers_raw,
            pen=2.0
        )
        return grippers_denoised
    
    else:
        raise ValueError(f"Fingertips data should be 2D or 3D array, got {fingertips.ndim}D")


def calculate_single_frame_gripper(fingertips: np.ndarray) -> float:
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
    other_tips_mean = fingertips[1:3].mean(axis=0)  # Average of other four fingertips

    # Calculate distance
    distance = np.linalg.norm(thumb_tip - other_tips_mean)

    # Human ergonomics normalization
    # Inverted: larger distance = more open = higher value
    gripper_value = (distance - HUMAN_MIN_DISTANCE) / HUMAN_RANGE

    # Clamp to [0, 1] range
    return np.clip(gripper_value, 0.0, 1.0)


def calculate_gripper_dual_hands(left_fingertips: np.ndarray, 
                               right_fingertips: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate gripper values for both hands independently.
    
    Args:
        left_fingertips: Left hand fingertips (5, 3) or (T, 5, 3)
        right_fingertips: Right hand fingertips (5, 3) or (T, 5, 3)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (left_gripper, right_gripper) values
    """
    left_gripper = calculate_gripper_from_fingertips(left_fingertips)
    right_gripper = calculate_gripper_from_fingertips(right_fingertips)
    
    return left_gripper, right_gripper


def extract_fingertips_from_keypoints(hand_keypoints: np.ndarray) -> np.ndarray:
    """
    Extract five fingertip coordinates from 25-point hand keypoint data.
    
    Uses the MediaPipe hand landmark model indices to extract the tip of each finger.
    The fingertip indices correspond to the distal phalanges of thumb, index, middle,
    ring, and pinky fingers respectively.
    
    Args:
        hand_keypoints: Hand keypoint array of shape (N, 25, 3) where N is the number
                      of timesteps, 25 is the number of hand landmarks, and 3 represents
                      xyz coordinates in 3D space.
                      
    Returns:
        np.ndarray: Array of shape (N, 5, 3) containing the 3D coordinates of the five
                   fingertips in order: [thumb, index, middle, ring, pinky].
                   
    Raises:
        ValueError: If input hand_keypoints doesn't have the expected (*, 25, 3) shape.
    """
    if hand_keypoints.shape[-2:] != (25, 3):
        raise ValueError(f"Hand keypoints should have shape (*, 25, 3), got {hand_keypoints.shape}")
    
    # MediaPipe hand landmark indices for fingertips
    # Based on the 21-point hand model extended to 25 points
    fingertip_indices = [4, 9, 14, 19, 24]  # thumb, index, middle, ring, pinky
    
    # Extract fingertip coordinates using advanced indexing
    fingertips = hand_keypoints[..., fingertip_indices, :]  # shape: (N, 5, 3)
    
    return fingertips


def build_state_vector(left_eef: np.ndarray, right_eef: np.ndarray,
                      left_fingertips: np.ndarray, right_fingertips: np.ndarray,
                      output_format: int = 46) -> np.ndarray:
    """
    Build state vectors from end-effector and fingertip data in multiple formats.
    
    Args:
        left_eef: Left wrist EEF poses of shape (T, 7) as [x,y,z,qw,qx,qy,qz]
        right_eef: Right wrist EEF poses of shape (T, 7) as [x,y,z,qw,qx,qy,qz]
        left_fingertips: Left fingertip positions of shape (T, 5, 3)
        right_fingertips: Right fingertip positions of shape (T, 5, 3)
        output_format: Output dimension format (46, 44, or 16)
        
    Returns:
        np.ndarray: State vectors with specified format:
                   - 46D: [left_eef(7) + left_gripper(1) + right_eef(7) + right_gripper(1) + 
                           left_fingertips(15) + right_fingertips(15)]
                   - 44D: [left_eef(7) + right_eef(7) + left_fingertips(15) + right_fingertips(15)]
                   - 16D: [left_eef(7) + left_gripper(1) + right_eef(7) + right_gripper(1)]
    """
    # Build state vectors based on output format
    if output_format == 46:
        # Full format: [left_eef(7) + left_gripper(1) + right_eef(7) + right_gripper(1) + fingertips(30)]
        left_gripper, right_gripper = calculate_gripper_dual_hands(left_fingertips, right_fingertips)
        left_fingertips_flat = left_fingertips.reshape(left_fingertips.shape[0], -1)
        right_fingertips_flat = right_fingertips.reshape(right_fingertips.shape[0], -1)
        
        state_vectors = np.concatenate([
            left_eef,                           # (T, 7)  - Left wrist EEF
            left_fingertips_flat,               # (T, 15) - Left fingertips flattened
            left_gripper[..., np.newaxis],      # (T, 1)  - Left gripper
            right_eef,                          # (T, 7)  - Right wrist EEF  
            right_fingertips_flat,              # (T, 15) - Right fingertips flattened
            right_gripper[..., np.newaxis],     # (T, 1)  - Right gripper
        ], axis=1)  # Final shape: (T, 46)
        
    elif output_format == 44:
        # Pose format (no gripper): [left_eef(7) + right_eef(7) + left_fingertips(15) + right_fingertips(15)]
        left_fingertips_flat = left_fingertips.reshape(left_fingertips.shape[0], -1)
        right_fingertips_flat = right_fingertips.reshape(right_fingertips.shape[0], -1)
        
        state_vectors = np.concatenate([
            left_eef,                           # (T, 7)  - Left wrist EEF
            left_fingertips_flat,               # (T, 15) - Left fingertips flattened
            right_eef,                          # (T, 7)  - Right wrist EEF
            right_fingertips_flat               # (T, 15) - Right fingertips flattened
        ], axis=1)  # Final shape: (T, 44)
        
    elif output_format == 16:
        # EEF format (no fingertips): [left_eef(7) + left_gripper(1) + right_eef(7) + right_gripper(1)]
        left_gripper, right_gripper = calculate_gripper_dual_hands(left_fingertips, right_fingertips)
        
        state_vectors = np.concatenate([
            left_eef,                           # (T, 7)  - Left wrist EEF
            left_gripper[..., np.newaxis],      # (T, 1)  - Left gripper
            right_eef,                          # (T, 7)  - Right wrist EEF  
            right_gripper[..., np.newaxis],     # (T, 1)  - Right gripper
        ], axis=1)  # Final shape: (T, 16)
        
    else:
        raise ValueError(f"Unsupported output_format: {output_format}. Supported: 46, 44, 16")
    
    return state_vectors