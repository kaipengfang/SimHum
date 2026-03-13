"""
Matrix Conversion Utilities Module
Provides 4x4 transformation matrix conversion functions
"""
import numpy as np
from scipy.spatial.transform import Rotation


def extract_xyz_from_matrix(matrix_4x4):
    """Extract xyz coordinates from a 4x4 transformation matrix"""
    if matrix_4x4.shape != (4, 4):
        return np.array([0.0, 0.0, 0.0])
    return matrix_4x4[:3, 3]


def matrix_to_xyz_quaternion(matrix_4x4):
    """Convert 4x4 transformation matrix to xyz+quaternion format (ported from process_2_eef.py)"""
    if matrix_4x4.shape != (4, 4):
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # Extract translation part
    translation = matrix_4x4[:3, 3]

    # Extract rotation part
    rotation_matrix = matrix_4x4[:3, :3]

    try:
        # Convert to quaternion using scipy
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # [qx, qy, qz, qw]

        # Reorder to [qw, qx, qy, qz]
        quaternion = np.roll(quaternion, shift=1)

        # Combine into [x, y, z, qw, qx, qy, qz]
        return np.concatenate([translation, quaternion])
    except Exception:
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
