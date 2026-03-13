"""
Global constants for data processing

Centralized constants to avoid circular imports.
"""

# Image processing constants
IMAGE_SIZE = (224, 224)  # Target image size for all cameras

# Camera names
CAM_NAMES = ['head_camera', 'left_camera', 'right_camera']  # Standard camera names
AGILEX_CAM_NAMES = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

# Dimension constants
ROBOT_DIM = 16  # Robot state/action dimension (dual-arm + grippers)
HUMAN_DIM = 46  # Human state/action dimension (dual-arm EEF + grippers + fingertips)

# Global constants for gripper calculation (human ergonomics)
HUMAN_MIN_DISTANCE = 0.04  # 2mm - physical limit when fully closed
HUMAN_MAX_DISTANCE = 0.10  # 4mm - reasonable limit when fully open  
HUMAN_RANGE = HUMAN_MAX_DISTANCE - HUMAN_MIN_DISTANCE