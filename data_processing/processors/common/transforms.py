"""
Coordinate transformation utilities

Handles coordinate system conversions and matrix transformations.
Following DRY principle - shared by multiple processors.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from typing import Optional, Tuple


def robot_to_human_coords(pos: np.ndarray) -> np.ndarray:
    """
    Rotate Robot coordinates to Human coordinates (90° counter-clockwise around Z).
    
    Robot: X=right, Y=forward, Z=up
    Human: X=forward, Y=left, Z=up
    
    Transformation (90° CCW rotation):
    Human_X = Robot_Y  (Robot's Y becomes Human's X)
    Human_Y = -Robot_X (Robot's X becomes Human's -Y)
    Human_Z = Robot_Z  (Z remains unchanged)
    
    Args:
        pos: Position in Robot coordinates [x, y, z]
        
    Returns:
        Position in Human coordinates [x', y', z']
    """
    x_new = pos[1]   # Robot Y -> Human X
    y_new = -pos[0]  # Robot X -> Human -Y
    z_new = pos[2]   # Z unchanged
    return np.array([x_new, y_new, z_new])


def transform_quaternion_robot_to_human(quat: np.ndarray) -> np.ndarray:
    """
    Rotate quaternion from Robot to Human coordinate system.
    
    When the coordinate system rotates 90° CCW around Z, we need to properly
    transform the quaternion to represent the same physical orientation in 
    the new coordinate system.
    
    The correct transformation is: R_human = T @ R_robot @ T^T
    where T is the coordinate transformation matrix and R is rotation matrix.
    
    Args:
        quat: Quaternion in Robot coords [qw, qx, qy, qz]
        
    Returns:
        Quaternion in Human coords [qw', qx', qy', qz']
    """
    # Convert quaternion to rotation matrix
    quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # [qx, qy, qz, qw] for scipy
    r_robot = Rotation.from_quat(quat_scipy)
    R_robot = r_robot.as_matrix()
    
    # Coordinate transformation matrix (90° CCW rotation around Z)
    # This transforms coordinates: [x', y'] = [y, -x]
    T = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 1]])
    
    # Transform the rotation matrix to the new coordinate system
    # R_human = T @ R_robot @ T.T
    R_human = T @ R_robot @ T.T
    
    # Convert back to quaternion
    r_human = Rotation.from_matrix(R_human)
    quat_human_scipy = r_human.as_quat()  # Returns [qx, qy, qz, qw]
    
    # Convert back to [qw, qx, qy, qz] format
    quat_human = np.array([quat_human_scipy[3], quat_human_scipy[0], 
                           quat_human_scipy[1], quat_human_scipy[2]])
    
    return quat_human


def transform_endpose_to_human_coords(endpose: np.ndarray) -> np.ndarray:
    """
    Transform complete endpose (position + quaternion) from Robot to Human coords.
    
    Args:
        endpose: 7D endpose [x, y, z, qw, qx, qy, qz] in Robot coords
        
    Returns:
        7D endpose in Human coords
    """
    # Extract position and quaternion
    pos = endpose[:3]
    quat = endpose[3:7]
    
    # Transform position
    pos_human = robot_to_human_coords(pos)
    
    # Transform quaternion
    quat_human = transform_quaternion_robot_to_human(quat)
    
    # Combine back
    return np.concatenate([pos_human, quat_human])


def unwrap_euler_angles(euler_sequence: np.ndarray) -> np.ndarray:
    """
    Unwrap euler angle sequence to ensure continuity across frames.

    Args:
        euler_sequence: Euler angles of shape (N, 3) in radians

    Returns:
        Unwrapped euler angles with continuous transitions
    """
    if len(euler_sequence) == 0:
        return euler_sequence

    unwrapped = euler_sequence.copy()
    for i in range(1, len(euler_sequence)):
        delta = unwrapped[i] - unwrapped[i-1]
        # Wrap delta to [-π, π]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        unwrapped[i] = unwrapped[i-1] + delta

    return unwrapped


def matrix_to_xyz_quaternion(matrix_4x4: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 homogeneous transformation matrices to xyz+quaternion format.
    Uses euler angle unwrapping to prevent quaternion jumps.

    Shared by human data processors following DRY principle.

    Args:
        matrix_4x4: Transformation matrices of shape (N, 4, 4)

    Returns:
        np.ndarray: Array containing [x, y, z, qw, qx, qy, qz] for each matrix
    """
    if matrix_4x4.shape[-2:] != (4, 4):
        raise ValueError(f"Input matrix should have shape (*, 4, 4), got {matrix_4x4.shape}")
    
    # Extract translation component (x, y, z) from the last column
    translation = matrix_4x4[..., :3, 3]  # shape: (..., 3)
    
    # Extract 3x3 rotation matrix from upper-left block
    rotation_matrices = matrix_4x4[..., :3, :3]  # shape: (..., 3, 3)
    
    # Convert rotation matrices to quaternions using scipy
    # Reshape for scipy compatibility (handles batch processing)
    original_shape = rotation_matrices.shape[:-2]
    rotation_matrices_flat = rotation_matrices.reshape(-1, 3, 3)
    
    # Use scipy Rotation class for robust conversion
    rotations = Rotation.from_matrix(rotation_matrices_flat)
    euler = rotations.as_euler('xyz', degrees=False)  # shape: (N, 3)

    # Apply euler angle unwrapping to ensure continuity
    euler_unwrapped = unwrap_euler_angles(euler)

    # Convert unwrapped euler angles back to quaternions
    rotations_unwrapped = Rotation.from_euler('xyz', euler_unwrapped, degrees=False)
    quaternions = rotations_unwrapped.as_quat()  # scipy format: [qx, qy, qz, qw]

    # Reorder to scalar-first format: [qw, qx, qy, qz] for consistency
    quaternions = np.roll(quaternions, shift=1, axis=-1)
    
    # Reshape back to original batch dimensions
    quaternions = quaternions.reshape(*original_shape, 4)

    # Combine translation and quaternion: [x, y, z, qw, qx, qy, qz]
    xyz_quat = np.concatenate([translation, quaternions], axis=-1)

    return xyz_quat


def adjust_sg_params(T: int, window_length: int, polyorder: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Adjust Savitzky-Golay parameters to ensure they are valid.

    Args:
        T: Sequence length
        window_length: Desired window length
        polyorder: Desired polynomial order

    Returns:
        Tuple of (adjusted_window_length, adjusted_polyorder) or (None, None)
    """
    # Ensure window is odd and not larger than T
    wl = min(window_length, T if T % 2 == 1 else max(T - 1, 1))

    # Ensure minimum window size
    if wl < 3:
        return None, None

    # Ensure polynomial order is valid
    po = min(polyorder, wl - 1)

    return wl, po


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w,x,y,z] to [x,y,z,w] format."""
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x,y,z,w] to [w,x,y,z] format."""
    return q[..., [3, 0, 1, 2]]


def ensure_quaternion_continuity_xyzw(qs_xyzw: np.ndarray) -> np.ndarray:
    """
    Ensure quaternion sequence continuity by flipping quaternions
    when adjacent quaternions have negative dot product.

    Args:
        qs_xyzw: Quaternion sequence in [x,y,z,w] format, shape (T, 4)

    Returns:
        Continuous quaternion sequence
    """
    qs = qs_xyzw.copy()
    for i in range(1, qs.shape[0]):
        if np.dot(qs[i - 1], qs[i]) < 0.0:
            qs[i] = -qs[i]
    return qs


def smooth_quaternion_sequence_wxyz(q_wxyz: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """
    Smooth quaternion sequence using Savitzky-Golay filter in rotation vector space.

    Args:
        q_wxyz: Quaternion sequence in [w,x,y,z] format, shape (T, 4)
        window_length: Window length for SG filter
        polyorder: Polynomial order for SG filter

    Returns:
        Smoothed quaternion sequence in [w,x,y,z] format
    """
    if q_wxyz.ndim != 2 or q_wxyz.shape[1] != 4:
        return q_wxyz

    T = q_wxyz.shape[0]
    wl, po = adjust_sg_params(T, window_length, polyorder)

    if wl is None:
        return q_wxyz

    # Convert to xyzw and ensure continuity
    q_xyzw = wxyz_to_xyzw(q_wxyz)
    q_xyzw = ensure_quaternion_continuity_xyzw(q_xyzw)

    # Convert to rotation vectors
    rotvec = Rotation.from_quat(q_xyzw).as_rotvec()

    # Apply SG smoothing
    rotvec_smooth = savgol_filter(rotvec, window_length=wl, polyorder=po, axis=0, mode="interp")

    # Convert back to quaternions and normalize
    q_smooth_xyzw = Rotation.from_rotvec(rotvec_smooth).as_quat()
    q_smooth_xyzw /= np.linalg.norm(q_smooth_xyzw, axis=1, keepdims=True)

    return xyzw_to_wxyz(q_smooth_xyzw)


def smooth_pose_sequence(states: np.ndarray, window_length: int = 31, polyorder: int = 3) -> np.ndarray:
    """
    Smooth pose sequence using Savitzky-Golay filter.

    For dual-arm robot states with format:
    [Lx, Ly, Lz, Lqw, Lqx, Lqy, Lqz, Lg, Rx, Ry, Rz, Rqw, Rqx, Rqy, Rqz, Rg]

    Or single-arm states with format:
    [x, y, z, qw, qx, qy, qz, gripper] or [x, y, z, qw, qx, qy, qz]

    Args:
        states: Input states with shape (T, N) where N is 16 (dual-arm), 8 (single-arm with gripper), or 7 (single-arm without gripper)
        window_length: SG filter window length (must be odd)
        polyorder: SG filter polynomial order (must be < window_length)

    Returns:
        Smoothed states with same shape
    """
    states_array = np.asarray(states, dtype=np.float64)

    if states_array.ndim != 2:
        return states

    T, N = states_array.shape

    if T < 3:
        return states

    # Adjust parameters for actual sequence length
    wl, po = adjust_sg_params(T, window_length, polyorder)

    if wl is None:
        return states

    smoothed = states_array.copy()

    if N == 16:  # Dual-arm format: [L_xyz, L_quat_wxyz, L_gripper, R_xyz, R_quat_wxyz, R_gripper]
        # Smooth position data (xyz) for both arms
        smoothed[:, 0:3] = savgol_filter(states_array[:, 0:3], window_length=wl, polyorder=po, axis=0, mode="interp")
        smoothed[:, 8:11] = savgol_filter(states_array[:, 8:11], window_length=wl, polyorder=po, axis=0, mode="interp")

        # Smooth quaternion data (wxyz) for both arms
        smoothed[:, 3:7] = smooth_quaternion_sequence_wxyz(states_array[:, 3:7], wl, po)
        smoothed[:, 11:15] = smooth_quaternion_sequence_wxyz(states_array[:, 11:15], wl, po)

        # Gripper values (indices 7 and 15) remain unchanged

    elif N == 8:  # Single-arm with gripper format: [xyz, quat_wxyz, gripper]
        # Smooth position data
        smoothed[:, 0:3] = savgol_filter(states_array[:, 0:3], window_length=wl, polyorder=po, axis=0, mode="interp")

        # Smooth quaternion data
        smoothed[:, 3:7] = smooth_quaternion_sequence_wxyz(states_array[:, 3:7], wl, po)

        # Gripper value (index 7) remains unchanged

    elif N == 7:  # Single-arm without gripper format: [xyz, quat_wxyz]
        # Smooth position data
        smoothed[:, 0:3] = savgol_filter(states_array[:, 0:3], window_length=wl, polyorder=po, axis=0, mode="interp")

        # Smooth quaternion data
        smoothed[:, 3:7] = smooth_quaternion_sequence_wxyz(states_array[:, 3:7], wl, po)

    else:
        # Unsupported format, return original
        return states

    return smoothed