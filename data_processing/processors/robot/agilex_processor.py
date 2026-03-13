"""
AgilexRobot data processor

Handles AgileX real robot data with support for both single-task and multi-task structures.
"""

import os
import glob
import h5py
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ..base import BaseDataProcessor, DataInfo, EpisodeData, AGILEX_CAM_NAMES, ROBOT_DIM
from ..common.transforms import smooth_pose_sequence

class AgilexRobotProcessor(BaseDataProcessor):
    """
    Processor for AgileX real robot data with full compatibility with data_convert_agilex.py.
    
    Supports both single-task and multi-task data structures:
    - Single task: base_path/data/*.hdf5 and base_path/instruction.txt
    - Multi task: base_path/task_name/data/*.hdf5 and base_path/task_name/instruction.txt
    
    Data format:
    - HDF5 files with eef_pose data for dual-arm system
    - 16-dimensional action space: [left_arm(7+1), right_arm(7+1)]
    - AgileX cameras: cam_high, cam_left_wrist, cam_right_wrist
    - Gripper values processed from action array with special scaling
    """
    
    def __init__(self, config_section: Dict[str, Any], task_id_offset: int = 0):
        """Initialize AgilexRobotProcessor with structure detection."""
        super().__init__(config_section, task_id_offset)
        
        # Detect data structure on initialization
        self.structure_type, self.task_info = self._detect_data_structure()
        self.use_qpos = config_section.get('use_qpos', False)
        
        # Build task list based on detected structure
        if self.structure_type == 'single':
            task_name, _ = self.task_info
            self.detected_tasks = [task_name]
        else:  # multi
            self.detected_tasks = [name for name, _ in self.task_info]
        
        # Override tasks if not specified in config
        if not self.tasks:
            self.tasks = self.detected_tasks
            
        print(f"AgilexRobotProcessor initialized with {self.structure_type} structure")
        print(f"Tasks: {self.tasks}")
    
    def _detect_data_structure(self) -> Tuple[str, Any]:
        """
        Auto-detect single-task or multi-task structure.

        Returns:
            (structure_type, task_info) where:
            - structure_type: 'single' or 'multi'
            - task_info: (task_name, task_path) for single or [(task_name, task_path), ...] for multi
        """
        # Check for single-task structure
        data_dir = os.path.join(self.data_path, 'data')
        instruction_file = os.path.join(self.data_path, 'instruction.txt')

        # Check if data subdirectory exists or if there are hdf5 files directly in the path
        has_data_subdir = os.path.exists(data_dir)
        has_direct_hdf5 = len(glob.glob(os.path.join(self.data_path, '*.hdf5'))) > 0 if not has_data_subdir else False

        if (has_data_subdir or has_direct_hdf5) and os.path.exists(instruction_file):
            task_name = os.path.basename(self.data_path)
            print(f"Detected SINGLE task structure: {task_name}")
            return 'single', (task_name, self.data_path)

        # Check for multi-task structure
        task_dirs = []
        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            if os.path.isdir(item_path):
                sub_data_dir = os.path.join(item_path, 'data')
                sub_instruction_file = os.path.join(item_path, 'instruction.txt')
                # Check if subdirectory has data/ folder or direct hdf5 files
                has_sub_data_dir = os.path.exists(sub_data_dir)
                has_sub_direct_hdf5 = len(glob.glob(os.path.join(item_path, '*.hdf5'))) > 0 if not has_sub_data_dir else False

                if (has_sub_data_dir or has_sub_direct_hdf5) and os.path.exists(sub_instruction_file):
                    task_dirs.append((item, item_path))

        if task_dirs:
            task_dirs.sort()  # Ensure consistent ordering
            task_names = [name for name, _ in task_dirs]
            print(f"Detected MULTI task structure: {len(task_dirs)} tasks: {task_names}")
            return 'multi', task_dirs

        # Default to single task structure without instruction file
        print(f"Warning: No clear structure detected, assuming single task without instruction")
        return 'single', (os.path.basename(self.data_path), self.data_path)
    
    def get_data_info(self) -> DataInfo:
        """Get information about AgileX data source"""
        episodes = self.load_episodes()
        return DataInfo(
            data_type="AgilexRobot",
            num_tasks=len(self.tasks),
            task_names=self.tasks,
            total_episodes=len(episodes),
            output_dimension=ROBOT_DIM,
            has_video=False
        )
    
    def load_episodes(self) -> List[Tuple[int, str]]:
        """
        Find all AgileX HDF5 episode files based on detected structure.

        Returns:
            List of (task_id, file_path) tuples
        """
        all_files = []

        if self.structure_type == 'single':
            # Single task structure
            task_id = self.task_id_offset
            data_dir = os.path.join(self.data_path, 'data')

            # Check if data subdirectory exists, otherwise use base path
            if os.path.exists(data_dir):
                search_dir = data_dir
            else:
                search_dir = self.data_path

            files = glob.glob(os.path.join(search_dir, '**/*.hdf5'), recursive=True)

            # Limit episodes if specified
            if self.max_episodes and len(files) > self.max_episodes:
                files = files[:self.max_episodes]

            for file_path in files:
                all_files.append((task_id, file_path))

            print(f"Found {len(files)} AgileX episodes in single task (from {search_dir})")

        else:  # multi task structure
            for local_id, (task_name, task_path) in enumerate(self.task_info):
                # Only process tasks that are in the configured task list
                if task_name not in self.tasks:
                    continue

                task_id = self.task_id_offset + local_id
                data_dir = os.path.join(task_path, 'data')

                # Check if data subdirectory exists, otherwise use task path
                if os.path.exists(data_dir):
                    search_dir = data_dir
                else:
                    search_dir = task_path

                files = glob.glob(os.path.join(search_dir, '**/*.hdf5'), recursive=True)

                # Limit episodes per task if specified
                if self.max_episodes and len(files) > self.max_episodes:
                    files = files[:self.max_episodes]

                for file_path in files:
                    all_files.append((task_id, file_path))

                print(f"Found {len(files)} AgileX episodes in task {task_name} (from {search_dir})")

        return all_files
    
    def load_single_episode(self, episode_info: Tuple[int, str]) -> Optional[EpisodeData]:
        """
        Load a single AgileX episode with data_convert_agilex.py compatible processing.
        
        HDF5 structure:
        /eef_pose/puppet_eef_pose/
        ├── left_eef_4D: (T, 8) - left arm pose + gripper 
        └── right_eef_4D: (T, 8) - right arm pose + gripper
        /action: (T, 14) - joint actions with gripper values at indices 6 and 13
        /observations/images/
        ├── cam_high: (T, encoded)
        ├── cam_left_wrist: (T, encoded)
        └── cam_right_wrist: (T, encoded)
        
        Returns:
            EpisodeData with 16-dim states/actions
        """
        task_id, file_path = episode_info
        
        if not os.path.isfile(file_path):
            print(f"AgileX file does not exist: {file_path}")
            return None
            
        try:
            with h5py.File(file_path, 'r') as f:
                if self.use_qpos:
                    # qpos mode - use action data directly
                    if 'action' not in f:
                        print(f"Warning: qpos mode but /action not found in {file_path}")
                        return None
                    
                    action_data = f['action'][:]
                    left_state = action_data[:, :7]
                    right_state = action_data[:, 7:14]

                    # Create action arrays (just copy state for qpos mode)
                    left_action = left_state.copy()
                    right_action = right_state.copy()
                    
                else:
                    # eef_pose mode (default) - matches data_convert_agilex.py logic
                    if '/eef_pose/puppet_eef_pose/left_eef_4D' not in f:
                        print(f"Warning: eef_pose data not found in {file_path}")
                        return None
                    
                    left_state = f['eef_pose/puppet_eef_pose/left_eef_4D'][:]
                    right_state = f['eef_pose/puppet_eef_pose/right_eef_4D'][:]
                    qpos_action = f['action'][:]
                    
                    # smooth pose sequences to reduce noise
                    left_state = smooth_pose_sequence(left_state)
                    right_state = smooth_pose_sequence(right_state)

                    # Create action arrays from state data
                    left_action = left_state.copy()
                    right_action = right_state.copy()
                    
                    # Replace gripper values with processed qpos_action gripper values
                    # This matches data_convert_agilex.py lines 366-372
                    left_action[:, 7] = qpos_action[:, 6]
                    right_action[:, 7] = qpos_action[:, 13]
                    
                    # Process gripper dimensions (scale and normalize)
                    left_action[:, 7] = np.clip(left_action[:, 7] * 1e6, 0, 70000)
                    right_action[:, 7] = np.clip(right_action[:, 7] * 1e6, 0, 70000)
                    left_action[:, 7] = left_action[:, 7] / 70000
                    right_action[:, 7] = right_action[:, 7] / 70000
                
                # Load image data from AgileX cameras
                image_dict = {}
                for agilex_cam in AGILEX_CAM_NAMES:
                    cam_path = f'/observations/images/{agilex_cam}'
                    if cam_path in f:
                        image_dict[agilex_cam] = f[cam_path][:]
                
                # Determine minimum length for alignment
                min_len = min(left_state.shape[0], right_state.shape[0])
                for cam in image_dict:
                    min_len = min(min_len, image_dict[cam].shape[0])
                
                # Align all sequences to minimum length
                left_state = left_state[:min_len]
                right_state = right_state[:min_len]
                left_action = left_action[:min_len]
                right_action = right_action[:min_len]
                for cam in image_dict:
                    image_dict[cam] = image_dict[cam][:min_len]
                
                # Process each timestep (avoiding all-zero gripper values)
                # This matches data_convert_agilex.py lines 397-407
                for j in range(min_len - 1):
                    if self.use_qpos:
                        # qpos mode gripper index is 6
                        if left_state[j, 6] == 0: left_state[j, 6] = 0.001
                        if right_state[j, 6] == 0: right_state[j, 6] = 0.001
                        if left_action[j+1, 6] == 0: left_action[j+1, 6] = 0.001
                        if right_action[j+1, 6] == 0: right_action[j+1, 6] = 0.001
                    else:
                        # eef_pose mode gripper index is 7
                        if left_state[j, 7] == 0: left_state[j, 7] = 0.001
                        if right_state[j, 7] == 0: right_state[j, 7] = 0.001
                        if left_action[j+1, 7] == 0: left_action[j+1, 7] = 0.001
                        if right_action[j+1, 7] == 0: right_action[j+1, 7] = 0.001
                
                # Create state and action sequences
                # State: current timestep, Action: next timestep (like data_convert_agilex.py)
                states_list = []
                actions_list = []
                
                for j in range(min_len - 1):
                    # Concatenate left and right for current state
                    if self.use_qpos:
                        state = np.concatenate([left_state[j], right_state[j]])
                    else:
                        # For eef_pose mode, take first 8 dims (7 pose + 1 gripper)
                        state = np.concatenate([left_state[j, :8], right_state[j, :8]])
                    
                    # Concatenate left and right for next timestep as action
                    if self.use_qpos:
                        action = np.concatenate([left_action[j+1], right_action[j+1]])
                    else:
                        action = np.concatenate([left_action[j+1, :8], right_action[j+1, :8]])
                    
                    states_list.append(state)
                    actions_list.append(action)
                
                states_array = np.array(states_list)
                actions_array = np.array(actions_list)
                
                # Remap AgileX camera names to standard names
                standard_image_dict = {}
                cam_mapping = {
                    'cam_high': 'head_camera',
                    'cam_left_wrist': 'left_camera', 
                    'cam_right_wrist': 'right_camera'
                }
                
                for agilex_name, standard_name in cam_mapping.items():
                    if agilex_name in image_dict:
                        # Trim to match state/action length
                        standard_image_dict[standard_name] = image_dict[agilex_name][:len(states_array)]
                
                return EpisodeData(
                    states=states_array,
                    actions=actions_array,
                    images=standard_image_dict,
                    episode_length=len(states_array)
                )
                
        except Exception as e:
            print(f"Error loading AgileX episode {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_instruction(self, task_name: str) -> str:
        """
        Get instruction text for AgileX task.
        
        Handles both single-task and multi-task structures:
        - Single: reads from base_path/instruction.txt
        - Multi: reads from base_path/task_name/instruction.txt
        
        Returns:
            Instruction text with AgileX prefix
        """
        instruction_text = None
        
        # First check config for explicit instruction
        if 'instruction' in self.config:
            instruction_text = self.config['instruction'].strip()
        else:
            # Load from file based on structure
            if self.structure_type == 'single':
                instruction_file = os.path.join(self.data_path, 'instruction.txt')
                if os.path.exists(instruction_file):
                    with open(instruction_file, 'r') as f:
                        instruction_text = f.read().strip()
            else:  # multi
                # Find the task path
                for t_name, t_path in self.task_info:
                    if t_name == task_name:
                        instruction_file = os.path.join(t_path, 'instruction.txt')
                        if os.path.exists(instruction_file):
                            with open(instruction_file, 'r') as f:
                                instruction_text = f.read().strip()
                        break
        
        # Fallback to task name if no instruction found
        if not instruction_text:
            print(f"Warning: No instruction found for {task_name}, using task name")
            instruction_text = task_name.replace('_', ' ')
        
        # Ensure ends with period
        if not instruction_text.endswith('.'):
            instruction_text += '.'
        
        return f"An Agilex Robot in the Real World: {instruction_text}"
    
    def get_output_dimension(self) -> int:
        """AgilexRobot outputs 16-dimensional data"""
        return ROBOT_DIM