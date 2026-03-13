"""
HumanCollect data processor

Handles human demonstration data with dual-arm end-effector and fingertip tracking.
"""

import os
import glob
import h5py
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ..base import BaseDataProcessor, DataInfo, EpisodeData, HUMAN_DIM
from ..common.transforms import matrix_to_xyz_quaternion
from ..common.image_utils import process_head_images
from ..common.gripper_utils import (
    build_state_vector, 
    calculate_gripper_dual_hands, 
    extract_fingertips_from_keypoints
)


class HumanCollectProcessor(BaseDataProcessor):
    """
    Processor for Human Collect data with dual-arm end-effector and fingertip tracking.
    
    This processor handles real human demonstration data collected with hand tracking,
    converting wrist transformation matrices and hand keypoints into a unified 46-dimensional
    state representation suitable for robotic learning.
    
    Data format:
    - HDF5 files with raw hand tracking data or converted eef_state data
    - 46-dimensional state space: dual-arm EEF poses (7+7) + grippers (1+1) + fingertip positions (15+15)
    - Head camera images for visual context
    - Actions derived from temporal progression: action[t] = state[t+1]
    
    State vector composition (46-dim total):
    - Left wrist EEF: [x, y, z, qw, qx, qy, qz] (7-dim)
    - Left gripper: calculated from fingertips distance (1-dim)
    - Right wrist EEF: [x, y, z, qw, qx, qy, qz] (7-dim)
    - Right gripper: calculated from fingertips distance (1-dim)
    - Left fingertips: [thumb_xyz, index_xyz, middle_xyz, ring_xyz, pinky_xyz] (15-dim)
    - Right fingertips: [thumb_xyz, index_xyz, middle_xyz, ring_xyz, pinky_xyz] (15-dim)
    """
    
    def _convert_raw_to_eef_state(self, left_wrist_mat: np.ndarray, right_wrist_mat: np.ndarray,
                                 left_hand_keypoints: np.ndarray, right_hand_keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert raw hand tracking data to structured end-effector state representation.
        
        This method processes raw 4x4 transformation matrices and hand keypoints into a
        standardized format suitable for robot learning. It handles both wrist poses and
        fingertip positions for both hands.
        
        Args:
            left_wrist_mat: Left wrist transformation matrices of shape (T, 4, 4)
            right_wrist_mat: Right wrist transformation matrices of shape (T, 4, 4)
            left_hand_keypoints: Left hand keypoints of shape (T, 25, 3)
            right_hand_keypoints: Right hand keypoints of shape (T, 25, 3)
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing converted data with keys:
                - 'left_eef': Left wrist poses as (T, 7) xyz+quaternion
                - 'right_eef': Right wrist poses as (T, 7) xyz+quaternion
                - 'left_fingertips': Left fingertip positions as (T, 5, 3)
                - 'right_fingertips': Right fingertip positions as (T, 5, 3)
        """

        # Convert wrist transformation matrices to xyz+quaternion format
        left_eef = matrix_to_xyz_quaternion(left_wrist_mat)
        right_eef = matrix_to_xyz_quaternion(right_wrist_mat)

        # Extract fingertip coordinates from hand keypoints
        left_fingertips = extract_fingertips_from_keypoints(left_hand_keypoints)
        right_fingertips = extract_fingertips_from_keypoints(right_hand_keypoints)
        
        return {
            'left_eef': left_eef,
            'right_eef': right_eef,
            'left_fingertips': left_fingertips,
            'right_fingertips': right_fingertips
        }
    
    def get_data_info(self) -> DataInfo:
        """Get information about HumanCollect data source"""
        episodes = self.load_episodes()
        return DataInfo(
            data_type="HumanCollect",
            num_tasks=len(self.tasks),
            task_names=self.tasks,
            total_episodes=len(episodes),
            output_dimension=HUMAN_DIM,
            has_video=True  # Head camera images available
        )
    
    def load_episodes(self) -> List[Tuple[int, str]]:
        """
        Find all HumanCollect HDF5 episode files.
        
        Searches for HDF5 files containing human demonstration data with hand tracking.
        Supports both flat directory structure and task-based subdirectories.
        
        Expected directory structures:
        Option 1 - Task-based:
        base_path/
        ├── task1/
        │   └── **/*.hdf5
        └── task2/
            └── **/*.hdf5
            
        Option 2 - Flat:
        base_path/
        ├── episode_0.hdf5
        ├── episode_1.hdf5
        └── ...
            
        Returns:
            List[Tuple[int, str]]: List of (task_id, file_path) tuples for all found episodes
        """
        all_files = []
        
        # If tasks are specified, search in task subdirectories
        if self.tasks:
            for local_id, task_name in enumerate(self.tasks):
                task_id = self.task_id_offset + local_id
                task_path = os.path.join(self.data_path, task_name)
                
                if os.path.exists(task_path):
                    files = glob.glob(os.path.join(task_path, '**/*.hdf5'), recursive=True)
                    
                    # Limit episodes per task if specified
                    if self.max_episodes and len(files) > self.max_episodes:
                        files = files[:self.max_episodes]
                    
                    for file_path in files:
                        all_files.append((task_id, file_path))
                        
                    print(f"Found {len(files)} HumanCollect episodes in {task_name}")
                else:
                    print(f"Warning: HumanCollect task directory does not exist: {task_path}")
        else:
            # Search directly in base path for flat structure
            task_id = self.task_id_offset
            files = glob.glob(os.path.join(self.data_path, '**/*.hdf5'), recursive=True)
            
            # Limit total episodes if specified
            if self.max_episodes and len(files) > self.max_episodes:
                files = files[:self.max_episodes]
            
            for file_path in files:
                all_files.append((task_id, file_path))
                
            print(f"Found {len(files)} HumanCollect episodes in base directory")
                
        return all_files
    
    def load_single_episode(self, episode_info: Tuple[int, str]) -> Optional[EpisodeData]:
        """
        Load a single HumanCollect episode with automatic data conversion.
        
        This method handles both raw hand tracking data and pre-converted eef_state data.
        If raw data is found, it automatically converts it to the standardized format.
        The resulting 46-dimensional state vectors are ready for robot learning.
        
        Expected HDF5 structures:
        
        Raw data format:
        /action/cmd/
        ├── rel_left_wrist_mat: (T, 4, 4) - left wrist transformation matrices
        ├── rel_right_wrist_mat: (T, 4, 4) - right wrist transformation matrices
        ├── rel_left_hand_keypoints: (T, 25, 3) - left hand keypoint coordinates
        └── rel_right_hand_keypoints: (T, 25, 3) - right hand keypoint coordinates
        
        Converted data format:
        /eef_state/
        ├── left_eef: (T, 7) - left wrist xyz+quaternion
        ├── right_eef: (T, 7) - right wrist xyz+quaternion
        ├── left_fingertips: (T, 5, 3) - left fingertip coordinates
        └── right_fingertips: (T, 5, 3) - right fingertip coordinates
        
        Image data:
        /observation/image/
        └── head: (T,) - encoded head camera images
        
        Args:
            episode_info: Tuple of (task_id, file_path) identifying the episode to load
            
        Returns:
            Optional[EpisodeData]: Episode data with 46-dim states/actions and head images,
                                  or None if loading fails
        """
        task_id, file_path = episode_info
        
        if not os.path.isfile(file_path):
            print(f"HumanCollect file does not exist: {file_path}")
            return None
            
        try:
            with h5py.File(file_path, "r") as root:
                # Convert from raw data
                raw_data_keys = [
                    'action/cmd/rel_left_wrist_mat',
                    'action/cmd/rel_right_wrist_mat',
                    'action/cmd/rel_left_hand_keypoints',
                    'action/cmd/rel_right_hand_keypoints'
                ]

                # Check if all raw data keys exist
                for key in raw_data_keys:
                    if key not in root:
                        print(f"Warning: Missing raw data key {key} in {file_path}")
                        return None

                # Load raw data
                left_wrist_mat = root['action/cmd/rel_left_wrist_mat'][...]
                right_wrist_mat = root['action/cmd/rel_right_wrist_mat'][...]
                left_hand_keypoints = root['action/cmd/rel_left_hand_keypoints'][...]
                right_hand_keypoints = root['action/cmd/rel_right_hand_keypoints'][...]

                # Convert raw data to eef_state format
                converted_data = self._convert_raw_to_eef_state(
                    left_wrist_mat, right_wrist_mat,
                    left_hand_keypoints, right_hand_keypoints
                )

                left_eef = converted_data['left_eef']
                right_eef = converted_data['right_eef']
                left_fingertips = converted_data['left_fingertips']
                right_fingertips = converted_data['right_fingertips']
                
                # Build state vectors with configured format
                state_vectors = build_state_vector(
                    left_eef, right_eef, left_fingertips, right_fingertips, self.human_output_format
                )
                
                # Generate action sequence using temporal progression
                # action[t] = state[t+1], so we have T-1 actions for T states
                states_array = state_vectors[:-1]  # (T-1, 46)
                actions_array = state_vectors[1:].copy()   # (T-1, 46)
                
                # Process head camera images
                image_dict = {}
                if 'observation/image/head' in root:
                    head_images_raw = root['observation/image/head'][...]
                    
                    # Ensure image sequence length matches state sequence
                    min_length = min(len(head_images_raw), len(states_array))
                    head_images_processed = process_head_images(head_images_raw[:min_length])
                    
                    # Store as head_camera for consistency with base class expectations
                    image_dict['head_camera'] = head_images_processed
                else:
                    print(f"Warning: No head camera images found in {file_path}")
                    image_dict['head_camera'] = None
                
                # Create dummy images for left and right cameras to maintain compatibility
                image_dict['left_camera'] = None
                image_dict['right_camera'] = None
                
                return EpisodeData(
                    states=states_array,
                    actions=actions_array,
                    images=image_dict,
                    episode_length=len(states_array)
                )
                
        except Exception as e:
            print(f"Error loading HumanCollect episode {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_instruction(self, task_name: str) -> str:
        """
        Get instruction text for HumanCollect task.
        
        Generates human-readable instruction text for the demonstration task.
        Supports both explicit instruction specification and automatic generation
        from task names.
        
        Priority order:
        1. From yaml config 'instruction' field (explicit specification)
        2. Fallback to task name conversion (automatic generation)
        
        Args:
            task_name: Name of the task for instruction generation
            
        Returns:
            str: Human-readable instruction text with consistent formatting
        """
        # Priority 1: Check yaml config for explicit instruction
        if 'instruction' in self.config:
            instruction = self.config['instruction'].strip()
            if not instruction.endswith('.'):
                instruction += '.'
            return f"A Human Being in the Real World: {instruction}"
        
        # Priority 2: Convert task name to readable instruction
        print(f"Warning: No instruction found in config for {task_name}, using task name")
        instruction = task_name.replace('_', ' ').capitalize() 
        if not instruction.endswith('.'):
            instruction += '.'
        return f"A Human Being in the Real World: {instruction}"
    
    def get_output_dimension(self) -> int:
        """HumanCollect outputs configurable dimensional data"""
        return self.human_output_format