# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import numpy as np
import scipy.spatial.transform as st
from torch import nn
from torchvision import transforms

# Import IMAGE_SIZE from data processing constants
try:
    from data_processing.processors.common.constants import IMAGE_SIZE
except ImportError:
    # Fallback if import fails
    IMAGE_SIZE = (240, 320)


def normalize_quaternion(q, eps=1e-8):
    """Normalize quaternion to unit quaternion.

    Args:
        q: Quaternion, shape (..., 4), format (qw, qx, qy, qz)
        eps: Small epsilon value to prevent division by zero

    Returns:
        Normalized quaternion, same shape as input
    """
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    # Prevent division by zero
    norm = np.clip(norm, eps, None)
    q_normalized = q / norm

    # If the original quaternion norm is very small (near zero), replace with identity quaternion [1, 0, 0, 0]
    mask = (norm.squeeze(-1) < eps)[..., np.newaxis]
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    q_normalized = np.where(mask, identity_quat, q_normalized)

    return q_normalized


class RotationTransformer:
    """Transformer for converting between rotation representations.

    Supports conversion between quaternion (scipy format: qx, qy, qz, qw) and rotation matrix.
    Uses scipy.spatial.transform.Rotation internally.
    """

    def __init__(self, from_rep='quat', to_rep='matrix'):
        """
        Args:
            from_rep: Source representation ('quat' or 'matrix')
            to_rep: Target representation ('quat' or 'matrix')
        """
        self.from_rep = from_rep
        self.to_rep = to_rep

    @staticmethod
    def _transform_rotation(x, from_rep, to_rep):
        """Convert rotation representation using scipy.

        Args:
            x: Input rotation in from_rep format
            from_rep: Source representation
            to_rep: Target representation

        Returns:
            Rotation in to_rep format
        """
        rot = getattr(st.Rotation, f'from_{from_rep}')(x)
        out = getattr(rot, f'as_{to_rep}')()
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Convert from source to target representation."""
        return self._transform_rotation(x, from_rep=self.from_rep, to_rep=self.to_rep)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Convert from target to source representation."""
        return self._transform_rotation(x, from_rep=self.to_rep, to_rep=self.from_rep)

class RelativeTransformer:
    """Transformer for computing relative poses between frames for dual-arm robots.

    This class handles conversion between absolute and relative pose representations
    for dual-arm robot data.

    Supported formats:
    - 16D: [left_arm(8), right_arm(8)]
      Each arm: [x, y, z, qw, qx, qy, qz, gripper]

    - 44D: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
      Each eef: [x, y, z, qw, qx, qy, qz]
      fingertips: 30D position data (10 fingertips × 3D xyz)

    - 46D: [left_eef(7), left_fingertips(15), left_gripper(1), right_eef(7), right_fingertips(15), right_gripper(1)]
      Each eef: [x, y, z, qw, qx, qy, qz]
      fingertips: 30D position data (10 fingertips × 3D xyz)

    - Forward pass (backward=False): Converts absolute poses to relative poses
    - Backward pass (backward=True): Converts relative poses to absolute poses
    """

    def __init__(self):
        """Initialize the transformer with quaternion to matrix converter."""
        self.rot_quat2mat = RotationTransformer(from_rep='quat', to_rep='matrix')
    
    def _pose_to_matrix(self, pos: np.ndarray, rot_quat: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 homogeneous transformation matrix.

        Args:
            pos: Position (..., 3) or (3,)
            rot_quat: Quaternion in scipy format (..., 4) or (4,): (qx, qy, qz, qw)

        Returns:
            4x4 homogeneous matrix (..., 4, 4) or (4, 4)
        """
        # Convert quaternion to rotation matrix
        rot_mat = self.rot_quat2mat.forward(rot_quat)  # (..., 3, 3) or (3, 3)

        # Determine if batched or single pose
        if pos.ndim == 1:
            # Single pose: (3,) -> (4, 4)
            T = np.eye(4, dtype=pos.dtype)
            T[:3, :3] = rot_mat
            T[:3, 3] = pos
        else:
            # Batched poses: (..., 3) -> (..., 4, 4)
            batch_shape = pos.shape[:-1]
            T = np.zeros((*batch_shape, 4, 4), dtype=pos.dtype)
            T[..., :3, :3] = rot_mat
            T[..., :3, 3] = pos
            T[..., 3, 3] = 1.0

        return T

    def _matrix_to_pose(self, T: np.ndarray):
        """Convert 4x4 homogeneous matrix to position and quaternion.

        Args:
            T: 4x4 homogeneous matrix (..., 4, 4) or (4, 4)

        Returns:
            pos: Position (..., 3) or (3,)
            rot_quat: Quaternion in scipy format (..., 4) or (4,): (qx, qy, qz, qw)
        """
        pos = T[..., :3, 3]
        rot_mat = T[..., :3, :3]
        rot_quat = self.rot_quat2mat.inverse(rot_mat)  # scipy format: (qx, qy, qz, qw)
        return pos, rot_quat

    def compute_relative_pose(
        self,
        action_pose: np.ndarray,
        base_pose: np.ndarray,
        backward: bool = False
    ):
        """Compute relative or absolute pose transformation using 4x4 homogeneous matrices.

        This method properly handles the coupling between rotation and translation,
        which is important when base_pos or base_rot are non-identity.

        Args:
            action_pose: Action pose [x, y, z, qw, qx, qy, qz], shape (..., 7)
            base_pose: Base pose [x, y, z, qw, qx, qy, qz], shape (..., 7)
            backward: If False, compute relative pose (encode).
                     If True, compute absolute pose (decode).

        Returns:
            np.ndarray: Transformed pose [x, y, z, qw, qx, qy, qz], shape (..., 7)

        Mathematical formulation:
            Forward (absolute -> relative):
                T_relative = T_base^-1 @ T_action
            Backward (relative -> absolute):
                T_absolute = T_base @ T_relative
        """
        # Extract position and quaternion
        action_pos = action_pose[..., :3]
        action_quat_wxyz = action_pose[..., 3:7]
        base_pos = base_pose[..., :3]
        base_quat_wxyz = base_pose[..., 3:7]

        # Convert quaternion from (qw, qx, qy, qz) to scipy format (qx, qy, qz, qw)
        action_rot = np.concatenate([action_quat_wxyz[..., 1:4], action_quat_wxyz[..., 0:1]], axis=-1)
        base_rot = np.concatenate([base_quat_wxyz[..., 1:4], base_quat_wxyz[..., 0:1]], axis=-1)

        # Convert to 4x4 homogeneous matrices
        action_mat = self._pose_to_matrix(action_pos, action_rot)
        base_mat = self._pose_to_matrix(base_pos, base_rot)

        if not backward:
            # Forward pass: absolute -> relative
            # T_relative = T_base^-1 @ T_action
            base_mat_inv = np.linalg.inv(base_mat)
            output_mat = base_mat_inv @ action_mat
        else:
            # Backward pass: relative -> absolute
            # T_absolute = T_base @ T_relative
            output_mat = base_mat @ action_mat

        # Convert back to position + quaternion
        output_pos, output_rot_scipy = self._matrix_to_pose(output_mat)

        # Convert from scipy format (qx, qy, qz, qw) to (qw, qx, qy, qz)
        output_rot = np.concatenate([output_rot_scipy[..., 3:4], output_rot_scipy[..., 0:3]], axis=-1)

        # Concatenate position and rotation
        return np.concatenate([output_pos, output_rot], axis=-1)

    def forward(self, action: np.ndarray, base: np.ndarray, backward: bool = False, data_type: str = "robot"):
        """Transform dual-arm pose from/to relative representation.

        Args:
            action: Dual-arm action data, shape (..., D) where D is 16, 44, or 46
                   - 16D: [left_arm(8), right_arm(8)]
                     Each arm: [x, y, z, qw, qx, qy, qz, gripper]
                   - 44D: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
                     Each eef: [x, y, z, qw, qx, qy, qz]
                     fingertips: 30D position data
                     NOTE: Robot data padded to 44D will have fingertips all zeros
                   - 46D: [left_eef(7), left_fingertips(15), left_gripper(1), right_eef(7), right_fingertips(15), right_gripper(1)]
                     Each eef: [x, y, z, qw, qx, qy, qz]
                     Each fingertip: [x, y, z]
                     NOTE: Robot data padded to 46D will have fingertips all zeros
            base: Base pose, shape (D,) matching action dimension
            backward: If False, convert absolute to relative (encode).
                     If True, convert relative to absolute (decode).

        Returns:
            np.ndarray: Transformed pose, shape (..., D) matching input dimension
                       Quaternion is in (qw, qx, qy, qz) format - same as input

        Example (16D):
            >>> transformer = RelativeTransformer()
            >>> action = np.random.randn(10, 16)  # 10 samples, 16D
            >>> base = np.random.randn(16)
            >>> rel_pose = transformer.forward(action, base, backward=False)
            >>> # rel_pose shape: (10, 16)

        Example (44D - human data):
            >>> action = np.random.randn(10, 44)  # 10 samples, 44D
            >>> base = np.random.randn(44)
            >>> rel_pose = transformer.forward(action, base, backward=False)
            >>> # rel_pose shape: (10, 44)

        Example (44D - padded robot data):
            >>> action = np.zeros((10, 44))  # Padded robot data
            >>> action[:, :7] = np.random.randn(10, 7)  # left eef
            >>> action[:, 22:29] = np.random.randn(10, 7)  # right eef
            >>> base = np.zeros(44)
            >>> base[:7] = np.random.randn(7)
            >>> base[22:29] = np.random.randn(7)
            >>> rel_pose = transformer.forward(action, base, backward=False)
            >>> # Only eef parts are transformed, fingertips remain zeros
        """
        action_dim = action.shape[-1]

        if action_dim == 16 or data_type == "robot":
            # 16D format: [left_arm(8), right_arm(8)]
            # Each arm: [x, y, z, qw, qx, qy, qz, gripper]
            results = []

            for arm_idx in range(2):  # 0: left, 1: right
                arm_start = arm_idx * 8
                arm_end = arm_start + 8

                # Extract arm data: [x, y, z, qw, qx, qy, qz, gripper]
                action_arm = action[..., arm_start:arm_end]
                base_arm = base[..., arm_start:arm_end]

                # Extract pose (first 7D) and gripper (last 1D)
                action_pose = action_arm[..., :7]
                base_pose = base_arm[..., :7]
                action_gripper = action_arm[..., 7:8]

                # Compute relative pose using 4x4 matrices (returns 7D: [x, y, z, qw, qx, qy, qz])
                transformed_pose = self.compute_relative_pose(
                    action_pose, base_pose, backward=backward
                )

                # Concatenate pose + gripper to get 8D
                transformed_arm = np.concatenate([transformed_pose, action_gripper], axis=-1)
                results.append(transformed_arm)

            # Concatenate both arms: [left_arm(8), right_arm(8)] = 16D
            transformed_16d = np.concatenate(results, axis=-1)

            # If this is padded robot data (44D or 46D), append the zero padding back
            if action_dim > 16:
                padded_part = action[..., 16:]  # Get the zero padding
                return np.concatenate([transformed_16d, padded_part], axis=-1)
            else:
                return transformed_16d

        elif action_dim == 44:
            # 44D format: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
            # Each arm: [eef(7), fingertips(15)] = 22D
            results = []

            for arm_idx in range(2):  # 0: left, 1: right
                arm_start = arm_idx * 22
                arm_end = arm_start + 22

                # Extract arm data: [eef(7), fingertips(15)]
                action_arm = action[..., arm_start:arm_end]
                base_arm = base[..., arm_start:arm_end]

                # Extract eef (first 7D) and fingertips (last 15D)
                action_eef = action_arm[..., :7]  # [x, y, z, qw, qx, qy, qz]
                base_eef = base_arm[..., :7]
                action_fingertips = action_arm[..., 7:22]  # Keep unchanged

                # Transform eef using 4x4 matrices (returns 7D)
                transformed_eef = self.compute_relative_pose(
                    action_eef, base_eef, backward=backward
                )

                # Concatenate eef + fingertips to get 22D
                results.append(transformed_eef)
                results.append(action_fingertips)

            # Concatenate both arms: [left_arm(22), right_arm(22)] = 44D
            return np.concatenate(results, axis=-1)

        elif action_dim == 46:
            # 46D format: [left_eef(7), left_fingertips(15), left_gripper(1),
            #              right_eef(7), right_fingertips(15), right_gripper(1)]
            # Each arm: [eef(7), fingertips(15), gripper(1)] = 23D
            results = []

            for arm_idx in range(2):  # 0: left, 1: right
                arm_start = arm_idx * 23
                arm_end = arm_start + 23

                # Extract arm data: [eef(7), fingertips(15), gripper(1)]
                action_arm = action[..., arm_start:arm_end]
                base_arm = base[..., arm_start:arm_end]

                # Extract eef (first 7D), fingertips (next 15D), gripper (last 1D)
                action_eef = action_arm[..., :7]  # [x, y, z, qw, qx, qy, qz]
                base_eef = base_arm[..., :7]
                action_fingertips = action_arm[..., 7:22]  # Keep unchanged
                action_gripper = action_arm[..., 22:23]  # Keep unchanged

                # Transform eef using 4x4 matrices (returns 7D)
                transformed_eef = self.compute_relative_pose(
                    action_eef, base_eef, backward=backward
                )

                # Concatenate eef + fingertips + gripper to get 23D
                results.append(transformed_eef)
                results.append(action_fingertips)
                results.append(action_gripper)

            # Concatenate both arms: [left_arm(23), right_arm(23)] = 46D
            return np.concatenate(results, axis=-1)

        else:
            raise ValueError(f"Unsupported action dimension: {action_dim}. Expected 16, 44, or 46.")


class DataNormalizer:
    """Data normalizer for action and state normalization.

    Implements the normalization used in unified_converter.py:
    - Normalization: normalized = (raw - loc) / scale
    - Unnormalization: raw = normalized * scale + loc

    Supports both NumPy arrays and PyTorch tensors.

    Example:
        >>> # Create from parameters
        >>> normalizer = DataNormalizer(loc=[0.5, 0.3], scale=[0.2, 0.1])
        >>> # Normalize data
        >>> raw_data = np.array([0.7, 0.4])
        >>> normalized = normalizer.normalize(raw_data)
        >>> # Result: [(0.7-0.5)/0.2, (0.4-0.3)/0.1] = [1.0, 1.0]
        >>> # Unnormalize data
        >>> recovered = normalizer.unnormalize(normalized)
        >>> # Result: [0.7, 0.4]
    """

    def __init__(self, loc: np.ndarray, scale: np.ndarray):
        """
        Initialize normalizer with location and scale parameters.

        Args:
            loc: Location parameter (mean or mid-point), shape (D,)
            scale: Scale parameter (std or half-range), shape (D,)
        """
        # Convert to numpy if needed
        if isinstance(loc, (list, tuple)):
            loc = np.array(loc, dtype=np.float32)
        if isinstance(scale, (list, tuple)):
            scale = np.array(scale, dtype=np.float32)

        self.loc = loc.astype(np.float32)
        self.scale = scale.astype(np.float32)

        # Validate dimensions
        assert self.loc.shape == self.scale.shape, \
            f"loc and scale must have same shape, got {self.loc.shape} and {self.scale.shape}"

        self.dim = len(self.loc)

    @classmethod
    def from_json(cls, json_path: str) -> 'DataNormalizer':
        """
        Load normalizer parameters from JSON file.

        Expected JSON format:
        {
            "loc": [v1, v2, ...],
            "scale": [s1, s2, ...]
        }

        Args:
            json_path: Path to JSON file

        Returns:
            DataNormalizer instance
        """
        import json

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Normalization file not found: {json_path}")

        with open(json_path, 'r') as f:
            params = json.load(f)

        if 'loc' not in params or 'scale' not in params:
            raise ValueError(f"JSON must contain 'loc' and 'scale' keys, got {params.keys()}")

        return cls(loc=params['loc'], scale=params['scale'])

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data: normalized = (raw - loc) / scale

        Args:
            data: Raw data, shape (..., D) where D matches normalizer dimension

        Returns:
            Normalized data with same shape as input
        """
        # Handle both numpy and torch
        if torch.is_tensor(data):
            loc = torch.from_numpy(self.loc).to(data.device, dtype=data.dtype)
            scale = torch.from_numpy(self.scale).to(data.device, dtype=data.dtype)
            return (data - loc) / scale
        else:
            return (data - self.loc) / self.scale

    def unnormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Unnormalize data: raw = normalized * scale + loc

        Args:
            data: Normalized data, shape (..., D) where D matches normalizer dimension

        Returns:
            Unnormalized data with same shape as input
        """
        # Handle both numpy and torch
        if torch.is_tensor(data):
            loc = torch.from_numpy(self.loc).to(data.device, dtype=data.dtype)
            scale = torch.from_numpy(self.scale).to(data.device, dtype=data.dtype)
            return data * scale + loc
        else:
            return data * self.scale + self.loc

    def __repr__(self) -> str:
        return f"DataNormalizer(dim={self.dim}, loc_range=[{self.loc.min():.3f}, {self.loc.max():.3f}], scale_range=[{self.scale.min():.3f}, {self.scale.max():.3f}])"


class DualArmNormalizer:
    """Normalizer for dual-arm robot data with selective normalization.

    Supports both 16D and 44D formats:

    16D format:
    - [left_arm(8), right_arm(8)]
    - Each arm: [x, y, z, qw, qx, qy, qz, gripper]

    44D format:
    - [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
    - Each eef: [x, y, z, qw, qx, qy, qz]

    46D format:
    - [left_eef(7), left_fingertips(15), left_gripper(1), right_eef(7), right_fingertips(15), right_gripper(1)]
    - Each eef: [x, y, z, qw, qx, qy, qz]

    Normalization strategy:
    - xyz positions: Normalized
    - quaternions (4D): NOT normalized (keep original values)
    - gripper values: Normalized
    - fingertips positions (30D, only in 44D and 46D): Normalized

    This matches the behavior in unified_converter.py where quaternions
    are kept unchanged to preserve their mathematical properties.

    Example:
        >>> # Load from JSON files
        >>> normalizer = DualArmNormalizer.from_json(
        ...     'agilexrobot_ac_norm.json',
        ...     'agilexrobot_state_norm.json'
        ... )
        >>>
        >>> # Normalize 16D action
        >>> raw_action_16d = np.random.randn(16)
        >>> normalized_16d = normalizer.normalize_action(raw_action_16d)
        >>>
        >>> # Normalize 44D action
        >>> raw_action_44d = np.random.randn(44)
        >>> normalized_44d = normalizer.normalize_action(raw_action_44d)
        >>>
        >>> # Normalize 46D action
        >>> raw_action_46d = np.random.randn(46)
        >>> normalized_46d = normalizer.normalize_action(raw_action_46d)
    """

    def __init__(self, action_normalizer: DataNormalizer, state_normalizer: DataNormalizer):
        """
        Initialize with action and state normalizers.

        Args:
            action_normalizer: Normalizer for action data
            state_normalizer: Normalizer for state data
        """
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer

    @classmethod
    def from_json(cls, action_json_path: str, state_json_path: str) -> 'DualArmNormalizer':
        """
        Load normalizers from JSON files.

        Args:
            action_json_path: Path to action normalization JSON
            state_json_path: Path to state normalization JSON

        Returns:
            DualArmNormalizer instance
        """
        action_norm = DataNormalizer.from_json(action_json_path)
        state_norm = DataNormalizer.from_json(state_json_path)
        return cls(action_norm, state_norm)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action data.

        Note: Quaternions have loc=0.0 and scale=1.0 in the normalization parameters,
        so they remain unchanged through normalization (identity transformation).

        Args:
            action: Raw action data, shape (..., D) where D is 16 or 44 or 46
                   - 16D: [left_arm(8), right_arm(8)]
                     Each arm: [x, y, z, qw, qx, qy, qz, gripper]
                   - 44D: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
                     Each eef: [x, y, z, qw, qx, qy, qz]
                   - 46D: [left_eef(7), left_fingertips(15), left_gripper(1), right_eef(7), right_fingertips(15), right_gripper(1)]
                     Each eef: [x, y, z, qw, qx, qy, qz]

        Returns:
            Normalized action with same shape
        """
        return self.action_normalizer.normalize(action)

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Unnormalize action data.

        Note: Quaternions have loc=0.0 and scale=1.0 in the normalization parameters,
        so they remain unchanged through unnormalization (identity transformation).

        Args:
            action: Normalized action data, shape (..., D) where D is 16 or 44 or 46
                   - 16D: [left_arm(8), right_arm(8)]
                   - 44D: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
                   - 46D: [left_eef(7), left_fingertips(15), left_gripper(1), right_eef(7), right_fingertips(15), right_gripper(1)]

        Returns:
            Unnormalized action with same shape
        """
        return self.action_normalizer.unnormalize(action)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state data.

        Note: Quaternions have loc=0.0 and scale=1.0 in the normalization parameters,
        so they remain unchanged through normalization (identity transformation).

        Args:
            state: Raw state data, shape (..., D) where D is 16 or 44
                   - 16D: [left_arm(8), right_arm(8)]
                     Each arm: [x, y, z, qw, qx, qy, qz, gripper]
                   - 44D: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
                     Each eef: [x, y, z, qw, qx, qy, qz]
                   - 46D: [left_eef(7), left_fingertips(15), left_gripper(1), right_eef(7), right_fingertips(15), right_gripper(1)]
                     Each eef: [x, y, z, qw, qx, qy, qz]

        Returns:
            Normalized state with same shape
        """
        return self.state_normalizer.normalize(state)

    def unnormalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Unnormalize state data.

        Note: Quaternions have loc=0.0 and scale=1.0 in the normalization parameters,
        so they remain unchanged through unnormalization (identity transformation).

        Args:
            state: Normalized state data, shape (..., D) where D is 16 or 44 or 46
                   - 16D: [left_arm(8), right_arm(8)]
                   - 44D: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
                   - 46D: [left_eef(7), left_fingertips(15), left_gripper(1), right_eef(7), right_fingertips(15), right_gripper(1)]

        Returns:
            Unnormalized state with same shape
        """
        return self.state_normalizer.unnormalize(state)

    def __repr__(self) -> str:
        return (f"DualArmNormalizer(\n"
                f"  action: {self.action_normalizer},\n"
                f"  state: {self.state_normalizer}\n"
                f")")


class _MediumAug(nn.Module):
    def __init__(self, pad=0, size=224):
        super().__init__()
        self.pad = pad
        self.size = size
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        extra_dim = len(x.shape) > 4
        if extra_dim:
            assert len(x.shape) == 5
            B, T, C, H, W = x.shape
            x = x.reshape((B * T, C, H, W))

        n, c, h, w = x.size()
        assert h == w
        if self.pad > 0:
            padding = tuple([self.pad] * 4)
            x = torch.nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[: self.size]
        arange = arange.unsqueeze(0).repeat(self.size, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + h - self.size + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = torch.nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )
        x = self.norm(x)

        if extra_dim:
            return x.reshape((B, T, C, self.size, self.size))
        return x


def get_gpu_transform_by_name(name, size=224):
    if name == "gpu_medium":
        return _MediumAug(size=size)
    raise NotImplementedError


def get_transform_by_name(name, size=IMAGE_SIZE):
    if "gpu" in name:
        return None

    # Handle size parameter - support both int and tuple
    if isinstance(size, (tuple, list)):
        target_size = tuple(size)
        effective_size = min(size)  # Use smaller dimension for kernel_size calculation
    else:
        target_size = (size, size)
        effective_size = size

    if name == "preproc":
        return transforms.Compose(
            [
                transforms.Resize(target_size, antialias=False),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if name == "basic":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=target_size, scale=(0.2, 1.0), antialias=False
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if name == "medium":
        kernel_size = int(0.05 * effective_size)
        kernel_size = kernel_size + (1 - kernel_size % 2)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=target_size, scale=(0.9, 1.0), antialias=False
                ),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if name == "hard":
        kernel_size = int(0.05 * effective_size)
        kernel_size = kernel_size + (1 - kernel_size % 2)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=target_size, scale=(0.2, 1.0), antialias=False
                ),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    if name == "advanced":
        kernel_size = int(0.05 * effective_size)
        kernel_size = kernel_size + (1 - kernel_size % 2)
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=target_size, scale=(0.2, 1.0), antialias=False
                ),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    raise NotImplementedError(f"{name} not found!")
