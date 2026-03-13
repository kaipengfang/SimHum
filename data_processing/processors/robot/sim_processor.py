"""
SimRobot data processor

Handles simulated robot data with coordinate transformation support.
"""

import os
import glob
import h5py
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ..base import BaseDataProcessor, DataInfo, EpisodeData, CAM_NAMES, ROBOT_DIM
from ..common.transforms import transform_endpose_to_human_coords


class SimRobotProcessor(BaseDataProcessor):
    """
    Processor for simulated robot data.
    
    Data format:
    - HDF5 files with endpose (dual-arm + gripper positions) and joint_action
    - 16-dimensional action space: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
    - Standard 3-camera setup: head_camera, left_camera, right_camera
    - Actions derived from endpose sequence: action[t] = endpose[t+1]
    """
    
    def __init__(self, config_section: Dict[str, Any], task_id_offset: int = 0):
        """Initialize SimRobotProcessor with optional coordinate alignment."""
        super().__init__(config_section, task_id_offset)
        # Check if we need to align Robot coordinates to Human coordinates
        self.align_to_human_coords = config_section.get('align_to_human_coords', False)
        if self.align_to_human_coords:
            print("\n" + "="*70)
            print("🔄 COORDINATE SYSTEM ALIGNMENT ENABLED")
            print("="*70)
            print("  Source: Robot Coordinate System (X=right, Y=forward, Z=up)")
            print("  Target: Human Coordinate System (X=forward, Y=left, Z=up)")
            print("  Transformation: 90° counter-clockwise rotation around Z-axis")
            print("  - Position: X' = Y, Y' = -X, Z' = Z")
            print("  - Quaternion: Properly transformed via rotation matrix")
            print("="*70 + "\n")
    
    def get_data_info(self) -> DataInfo:
        """Get information about SimRobot data source"""
        episodes = self.load_episodes()
        return DataInfo(
            data_type="SimRobot",
            num_tasks=len(self.tasks),
            task_names=self.tasks,
            total_episodes=len(episodes),
            output_dimension=ROBOT_DIM,
            has_video=False
        )
    
    def load_episodes(self) -> List[Tuple[int, str]]:
        """
        Find all SimRobot HDF5 episode files.
        
        Expected structure:
        base_path/
        ├── task1/
        │   └── **/*.hdf5
        └── task2/
            └── **/*.hdf5
            
        Returns:
            List of (task_id, file_path) tuples
        """
        all_files = []
        for local_id, task_name in enumerate(self.tasks):
            task_id = self.task_id_offset + local_id
            task_path = os.path.join(self.data_path, task_name)
            
            if os.path.exists(task_path):
                # Recursively find all HDF5 files
                files = glob.glob(os.path.join(task_path, '**/*.hdf5'), recursive=True)
                
                # Limit episodes per task if specified
                if self.max_episodes and len(files) > self.max_episodes:
                    files = files[:self.max_episodes]
                
                for file_path in files:
                    all_files.append((task_id, file_path))
                    
                print(f"Found {len(files)} SimRobot episodes in {task_name}")
            else:
                print(f"Warning: SimRobot task directory does not exist: {task_path}")
                
        return all_files
    
    def load_single_episode(self, episode_info: Tuple[int, str]) -> Optional[EpisodeData]:
        """
        Load a single SimRobot episode.
        
        HDF5 structure:
        /endpose/
        ├── left_gripper: (T,) - left gripper openness
        ├── left_endpose: (T, 7) - left arm pose (xyz + quaternion) 
        ├── right_gripper: (T,) - right gripper openness
        └── right_endpose: (T, 7) - right arm pose
        /joint_action/vector: (T, joint_dim) - joint positions (unused)
        /observation/
        ├── head_camera/rgb: (T, encoded_images)
        ├── left_camera/rgb: (T, encoded_images)
        └── right_camera/rgb: (T, encoded_images)
        
        Args:
            episode_info: (task_id, file_path) tuple
            
        Returns:
            EpisodeData with 16-dim states/actions
        """
        task_id, file_path = episode_info
        
        if not os.path.isfile(file_path):
            print(f"SimRobot file does not exist: {file_path}")
            return None
            
        try:
            with h5py.File(file_path, "r") as root:
                # Load endpose data (main state representation)
                left_gripper = root["/endpose/left_gripper"][()]
                left_arm = root["/endpose/left_endpose"][()]
                right_gripper = root["/endpose/right_gripper"][()]
                right_arm = root["/endpose/right_endpose"][()]
                
                # Apply coordinate transformation if enabled
                if self.align_to_human_coords:
                    # Transform each timestep's endpose from Robot to Human coordinates
                    n_frames = len(left_arm)
                    left_arm_transformed = np.zeros_like(left_arm)
                    right_arm_transformed = np.zeros_like(right_arm)
                    
                    for t in range(n_frames):
                        left_arm_transformed[t] = transform_endpose_to_human_coords(left_arm[t])
                        right_arm_transformed[t] = transform_endpose_to_human_coords(right_arm[t])
                    
                    left_arm = left_arm_transformed
                    right_arm = right_arm_transformed
                    
                # OPTIMIZED: Concatenate to 16-dim state vector in one operation
                # [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
                states_raw = np.concatenate([
                    left_arm, 
                    left_gripper[..., np.newaxis],  # (T,) -> (T, 1)
                    right_arm, 
                    right_gripper[..., np.newaxis]
                ], axis=1)
                
                # OPTIMIZED: Use temporal shifting without normalization (maintain original interface)
                # Original: actions_array = states_raw[1:].copy(), states_array = states_raw[:-1]
                # Optimized: avoid .copy() by using views when safe
                actions_array = states_raw[1:] 
                states_array = states_raw[:-1].copy()  
                
                # Load image data from all cameras
                image_dict = {}
                for cam_name in CAM_NAMES:
                    if cam_name in root["/observation/"]:
                        image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
                    else:
                        print(f"Warning: Camera {cam_name} not found in {file_path}")
                
                return EpisodeData(
                    states=states_array,
                    actions=actions_array,
                    images=image_dict,
                    episode_length=len(states_array)
                )
                
        except Exception as e:
            print(f"Error loading SimRobot episode {file_path}: {e}")
            return None
    
    def get_instruction(self, task_name: str) -> str:
        """
        Get instruction text for SimRobot task.
        
        Priority order:
        1. From yaml config 'instruction' field
        2. From instruction.txt in task directory structure
        3. Fallback to task name
        
        Args:
            task_name: Name of the task
            
        Returns:
            Instruction text string
        """
        # Priority 1: Check yaml config for instruction
        if 'instruction' in self.config:
            instruction = self.config['instruction'].strip()
            if not instruction.endswith('.'):
                instruction += '.'
            return f"An Agilex Robot in the Simulated World: {instruction}"
        
        # Priority 2: Look for instruction.txt in task directory
        instruction_path = os.path.join(self.data_path, task_name, 'aloha-agilex_clean_50', 'instruction.txt')
        
        if os.path.exists(instruction_path):
            with open(instruction_path, 'r') as f:
                content = f.read().strip()
                if not content.endswith('.'):
                    content += '.'
                return f"An Agilex Robot in the Simulated World: {content}"
        
        # Priority 3: Fallback to task name
        print(f"Warning: No instruction found in config or file for {task_name}, using task name")
        fallback = task_name.replace('_', ' ')
        if not fallback.endswith('.'):
            fallback += '.'
        return f"An Agilex Robot in the Simulated World: {fallback}"
    
    def get_output_dimension(self) -> int:
        """SimRobot outputs 16-dimensional data"""
        return ROBOT_DIM