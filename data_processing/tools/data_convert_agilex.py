"""
Unified AgileX dataset conversion tool - supports single-task and multi-task

Supported data structures:
1. Single-task structure:
   task_path/
     data/
       episode_0.hdf5
       episode_1.hdf5
       ...
     instruction.txt

2. Multi-task structure:
   dataset_path/
     task1/
       data/
         episode_0.hdf5
         ...
       instruction.txt
     task2/
       data/
         episode_0.hdf5
         ...
       instruction.txt
     ...

Features:
- Automatically detects single-task or multi-task data structure
- Reads datasets (hdf5 format) and processes into unified trajectory data
- Supports state normalization (Gaussian / min-max)
- Supports image resize and encoding
- Automatically handles task instruction embedding
- Optional coordinate transformation

Output files:
- buf.pkl: contains all trajectory data
- ac_norm.json: action normalization parameters
- state_norm.json: state normalization parameters
- task_embeddings.pkl: instruction embeddings list
- task_instruction.txt: instruction text
- task_mapping.json: task name to ID mapping (multi-task only)

Usage:
# Single task
python data_convert_agilex.py -p /path/to/single_task --target /path/to/output

# Multi task
python data_convert_agilex.py -p /path/to/multi_tasks --target /path/to/output
"""

import argparse
import pickle as pkl
from tqdm import tqdm
import glob
import numpy as np
import os
import cv2
import h5py
import json
import torch
from simhum.models.qwen import QwenEmbedder
from scipy.spatial.transform import Rotation

IMAGE_SIZE = (256, 256)
CAM_NAMES = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

def detect_data_structure(base_path):
    """
    Automatically detect data structure type

    Args:
        base_path: Dataset path

    Returns:
        tuple: (structure_type, task_info)
        structure_type: 'single' or 'multi'
        task_info:
            - single task: (task_name, task_path)
            - multi task: [(task_name, task_path), ...]
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Path {base_path} does not exist.")
    
    # Check if it directly contains data/ dir and instruction.txt (single-task structure)
    data_dir = os.path.join(base_path, 'data')
    instruction_file = os.path.join(base_path, 'instruction.txt')
    
    if os.path.exists(data_dir) and os.path.exists(instruction_file):
        task_name = os.path.basename(base_path)
        print(f"Detected SINGLE task structure: {task_name}")
        return 'single', (task_name, base_path)
    
    # Check if it is a multi-task structure
    task_dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            sub_data_dir = os.path.join(item_path, 'data')
            sub_instruction_file = os.path.join(item_path, 'instruction.txt')
            if os.path.exists(sub_data_dir) and os.path.exists(sub_instruction_file):
                task_dirs.append((item, item_path))
    
    if task_dirs:
        task_dirs.sort()  # Ensure consistent task ordering
        task_names = [name for name, _ in task_dirs]
        print(f"Detected MULTI task structure: {len(task_dirs)} tasks: {task_names}")
        return 'multi', task_dirs
    
    # If neither matches, raise an error
    raise ValueError(f"Invalid data structure in {base_path}. Expected either:\n"
                     f"1. Single task: {base_path}/data/ and {base_path}/instruction.txt\n"
                     f"2. Multi task: {base_path}/task_name/data/ and {base_path}/task_name/instruction.txt")

def get_episodes_from_task(task_path):
    """
    Get all episode hdf5 file paths for a single task

    Args:
        task_path: Task directory path

    Returns:
        episode_paths: List of hdf5 file paths
    """
    data_dir = os.path.join(task_path, 'data')
    if not os.path.exists(data_dir):
        return []
    return glob.glob(os.path.join(data_dir, '**/*.hdf5'), recursive=True)

def _resize_and_encode(bgr_image, size=IMAGE_SIZE):
    """Resize image and encode to jpg format"""
    resize_height, resize_width = size
    frame = cv2.imdecode(bgr_image, cv2.IMREAD_COLOR)
    original_height, original_width = frame.shape[:2]
    scale = resize_height / original_height
    new_width = int(original_width * scale)
    resized = cv2.resize(frame, (new_width, resize_height), interpolation=cv2.INTER_AREA)
    
    # Center crop or pad width
    if new_width > resize_width:
        start_x = (new_width - resize_width) // 2
        resized = resized[:, start_x:start_x + resize_width]
    elif new_width < resize_width:
        pad_left = (resize_width - new_width) // 2
        pad_right = resize_width - new_width - pad_left
        resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    _, encoded = cv2.imencode(".jpg", resized)
    return encoded

def _gaussian_norm(all_acs):
    """Gaussian normalization"""
    print('Using gaussian norm')
    all_acs_arr = np.array(all_acs)
    mean = np.mean(all_acs_arr, axis=0)
    std = np.std(all_acs_arr, axis=0)
    std[std == 0] = 1e-17
    for a in all_acs:
        a -= mean
        a /= std
    return dict(loc=mean.tolist(), scale=std.tolist())

def _max_min_norm(all_acs):
    """Min-max normalization"""
    print('Using max min norm')
    all_acs_arr = np.array(all_acs)
    max_ac = np.max(all_acs_arr, axis=0)
    min_ac = np.min(all_acs_arr, axis=0)
    mid = (max_ac + min_ac) / 2
    delta = (max_ac - min_ac) / 2
    delta[delta == 0] = 1e-16  # Prevent division by zero
    for a in all_acs:
        a -= mid
        a /= delta
    return dict(loc=mid.tolist(), scale=delta.tolist())

def process_instructions(structure_type, task_info, target_path):
    """
    Unified instruction processing for single-task and multi-task

    Args:
        structure_type: 'single' or 'multi'
        task_info: Task information
        target_path: Output directory
        
    Returns:
        task_mapping: {task_name: task_id}
    """
    embedder = QwenEmbedder(device='cuda:6' if torch.cuda.is_available() else 'cpu')
    
    all_embeddings = []
    all_instructions = []
    task_mapping = {}
    
    if structure_type == 'single':
        task_name, task_path = task_info
        instruction_file = os.path.join(task_path, 'instruction.txt')
        
        with open(instruction_file, 'r') as f:
            instruction = f.read().strip()
        
        embedding = embedder.get_pooled_embeddings(instruction).cpu().numpy()
        all_embeddings.append(embedding)
        all_instructions.append(instruction)
        task_mapping[task_name] = 0
        
        print(f"Single task: {task_name} -> {instruction[:50]}...")
        
    elif structure_type == 'multi':
        for task_id, (task_name, task_path) in enumerate(task_info):
            instruction_file = os.path.join(task_path, 'instruction.txt')
            
            with open(instruction_file, 'r') as f:
                instruction = f.read().strip()
            
            embedding = embedder.get_pooled_embeddings(instruction).cpu().numpy()
            all_embeddings.append(embedding)
            all_instructions.append(instruction)
            task_mapping[task_name] = task_id
            
            print(f"Task {task_id}: {task_name} -> {instruction[:50]}...")
    
    # Save all embeddings and instructions
    embeddings_file = os.path.join(target_path, "task_embeddings.pkl")
    with open(embeddings_file, 'wb') as f:
        pkl.dump(all_embeddings, f)
    
    instruction_file = os.path.join(target_path, 'task_instruction.txt')
    with open(instruction_file, 'w') as f:
        for instruction in all_instructions:
            f.write(instruction + '\n')
    
    # Only save task mapping for multi-task
    if structure_type == 'multi':
        mapping_file = os.path.join(target_path, 'task_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(task_mapping, f, indent=2)
        print(f"Saved task mapping to {mapping_file}")
    
    print(f"Saved {len(all_embeddings)} task embedding(s) to {embeddings_file}")
    return task_mapping

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion to rotation matrix"""
    qw, qx, qy, qz = quat
    quat_scipy_format = [qx, qy, qz, qw]
    rot_obj = Rotation.from_quat(quat_scipy_format)
    return rot_obj.as_matrix()

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion using scipy"""
    rot_obj = Rotation.from_matrix(R)
    quat_scipy = rot_obj.as_quat()  # Returns [qx, qy, qz, qw]
    # Convert to [qw, qx, qy, qz] format
    return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]], dtype=np.float64)

def pose_to_homogeneous_matrix(pos, quat):
    """Convert position and quaternion to 4x4 homogeneous matrix"""
    R = quaternion_to_rotation_matrix(quat)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def homogeneous_matrix_to_pose(T):
    """Extract position and quaternion from 4x4 homogeneous matrix"""
    pos = T[:3, 3]
    R = T[:3, :3]
    quat = rotation_matrix_to_quaternion(R)
    return pos, quat

def transform_real_to_sim_world(real_eef_pose, arm_tag="left"):
    """Transform real robot EEF data to simulation world coordinate system"""
    real_pos = np.array(real_eef_pose[:3], dtype=np.float64)
    real_quat = np.array(real_eef_pose[3:7], dtype=np.float64)
    gripper_state = float(real_eef_pose[7])
    
    real_pos_with_offset = real_pos
    T_real = pose_to_homogeneous_matrix(real_pos_with_offset, real_quat)
    
    # Coordinate transform matrix: Real(X-forward Y-left Z-up) -> Sim(X-right Y-forward Z-up)
    T_coord_transform = np.array([
        [0, -1, 0, 0],  # Sim X = -Real Y
        [1,  0, 0, 0],  # Sim Y = Real X
        [0,  0, 1, 0],  # Sim Z = Real Z
        [0,  0, 0, 1]   # Homogeneous coordinate
    ], dtype=np.float64)
    
    T_sim = T_coord_transform @ T_real
    world_pos, sim_quat = homogeneous_matrix_to_pose(T_sim)
    
    world_eef_pose = np.array([
        world_pos[0], world_pos[1], world_pos[2],
        sim_quat[0], sim_quat[1], sim_quat[2], sim_quat[3],
        gripper_state
    ], dtype=np.float64)
    
    return world_eef_pose

def convert_unified_dataset(
    base_path, 
    target_path, 
    gaussian_norm=False, 
    use_qpos=False, 
    use_coordinate_transform=True
):
    """
    Unified conversion pipeline - supports single-task and multi-task

    Args:
        base_path: Dataset path
        target_path: Output directory
        gaussian_norm: Whether to use Gaussian normalization
        use_qpos: Whether to use qpos format
        use_coordinate_transform: Whether to apply coordinate transformation
    """
    print(f'gaussian_norm={gaussian_norm}')
    print(f'use_coordinate_transform={use_coordinate_transform}')
    print(f'Processing dataset: {base_path}')
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    # 1. Automatically detect data structure
    structure_type, task_info = detect_data_structure(base_path)
    
    # 2. Process instructions
    task_mapping = process_instructions(structure_type, task_info, target_path)
    
    # 3. Prepare task list for unified processing
    if structure_type == 'single':
        task_name, task_path = task_info
        tasks_to_process = [(task_name, task_path)]
    else:
        tasks_to_process = task_info
    
    # 4. Process data for all tasks
    all_trajs = []
    all_acs = []
    all_states = []
    
    for task_name, task_path in tasks_to_process:
        task_id = task_mapping[task_name]
        print(f"\nProcessing task {task_id}: {task_name}")
        
        episode_paths = get_episodes_from_task(task_path)
        print(f"Found {len(episode_paths)} episodes for task {task_name}")
        
        if not episode_paths:
            print(f"Warning: No episodes found for task {task_name}")
            continue
        
        task_trajs = []
        for episode_path in tqdm(episode_paths, desc=f"Processing {task_name}"):
            proc_traj = []
            
            with h5py.File(episode_path, 'r') as f:
                if use_qpos:
                    left_state = f['action'][:, :7]
                    right_state = f['action'][:, 7:14]
                else:
                    left_state = f['eef_pose/puppet_eef_pose/left_eef_4D'][:]
                    right_state = f['eef_pose/puppet_eef_pose/right_eef_4D'][:]
                    qpos_action = f['action'][:]
                    left_action = left_state.copy()
                    right_action = right_state.copy()
                    left_action[:, 7] = qpos_action[:, 6]
                    right_action[:, 7] = qpos_action[:, 13]
                    # Process gripper dimension
                    left_action[:, 7] = np.clip(left_action[:, 7] * 1e6, 0, 70000)
                    right_action[:, 7] = np.clip(right_action[:, 7] * 1e6, 0, 70000)
                    left_action[:, 7] = left_action[:, 7] / 70000
                    right_action[:, 7] = right_action[:, 7] / 70000
                
                if left_state is None or right_state is None:
                    continue
                
                # Read image sequences
                image_dict = {cam: f[f'/observations/images/{cam}'][:] 
                             for cam in CAM_NAMES if f'/observations/images/{cam}' in f}
                
                # Align step counts
                lens = [left_state.shape[0], right_state.shape[0]]
                for cam in image_dict:
                    lens.append(image_dict[cam].shape[0])
                num_steps = min(lens)
                
                # Optional coordinate transformation
                if use_coordinate_transform:
                    for i in range(num_steps - 1):
                        left_action[i] = transform_real_to_sim_world(left_action[i])
                        left_state[i] = transform_real_to_sim_world(left_state[i])
                        right_action[i] = transform_real_to_sim_world(right_action[i])
                        right_state[i] = transform_real_to_sim_world(right_state[i])
                
                # Process each timestep
                for j in range(num_steps - 1):
                    # Avoid all-zero gripper dimension
                    if use_qpos:
                        if left_state[j, 6] == 0: left_state[j, 6] = 0.001
                        if right_state[j, 6] == 0: right_state[j, 6] = 0.001
                        if left_action[j+1, 6] == 0: left_action[j+1, 6] = 0.001
                        if right_action[j+1, 6] == 0: right_action[j+1, 6] = 0.001
                    else:
                        if left_state[j, 7] == 0: left_state[j, 7] = 0.001
                        if right_state[j, 7] == 0: right_state[j, 7] = 0.001
                        if left_action[j+1, 7] == 0: left_action[j+1, 7] = 0.001
                        if right_action[j+1, 7] == 0: right_action[j+1, 7] = 0.001
                    
                    state = np.concatenate([left_state[j], right_state[j]])
                    action = np.concatenate([left_action[j+1], right_action[j+1]])
                    
                    obs = dict(state=state)
                    for idx, key in enumerate(CAM_NAMES):
                        if key in image_dict:
                            obs[f'enc_cam_{idx}'] = _resize_and_encode(image_dict[key][j])
                    
                    obs['instruction_id'] = task_id
                    reward = 0  # dummy reward
                    proc_traj.append((obs, action, reward))
                    all_states.append(state)
                    all_acs.append(action)
            
            if proc_traj:
                task_trajs.append(proc_traj)
        
        all_trajs.extend(task_trajs)
        print(f"Task {task_name}: processed {len(task_trajs)} episodes")
    
    # 5. Unified normalization
    total_tasks = len(tasks_to_process)
    print(f"\nNormalizing based on {len(all_acs)} total samples from {total_tasks} task(s)")
    ac_dict = _max_min_norm(all_acs) if not gaussian_norm else _gaussian_norm(all_acs)
    state_dict = _max_min_norm(all_states) if not gaussian_norm else _gaussian_norm(all_states)
    
    # 6. Save all data
    with open(os.path.join(target_path, 'ac_norm.json'), 'w') as f:
        json.dump(ac_dict, f)
    with open(os.path.join(target_path, 'state_norm.json'), 'w') as f:
        json.dump(state_dict, f)
    with open(os.path.join(target_path, 'buf.pkl'), 'wb') as f:
        pkl.dump(all_trajs, f)
    
    # 7. Output summary
    print(f"\n=== Conversion Summary ===")
    print(f"Data structure: {structure_type.upper()}")
    print(f"Total tasks: {total_tasks}")
    print(f"Total episodes: {len(all_trajs)}")
    print(f"Total samples: {len(all_acs)}")
    print(f"Output saved to: {target_path}")
    print(f"Files created:")
    print(f"  - buf.pkl: {len(all_trajs)} episodes")
    print(f"  - task_embeddings.pkl: {total_tasks} task embedding(s)")
    print(f"  - task_instruction.txt: {total_tasks} instruction(s)")
    if structure_type == 'multi':
        print(f"  - task_mapping.json: task name to ID mapping")
    print(f"  - ac_norm.json, state_norm.json: normalization parameters")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified AgileX dataset conversion tool - supports single-task and multi-task")
    parser.add_argument('-p', '--path', required=True, help='Dataset path (single-task directory or multi-task root directory)')
    parser.add_argument('-target', '--target', required=True, help='Output directory')
    parser.add_argument('--gaussian_norm', action='store_true', help='Use Gaussian normalization')
    parser.add_argument('--use_qpos', action='store_true', default=False, help='Use qpos format')
    parser.add_argument('--no_coordinate_transform', action='store_true', default=True, help='Disable coordinate transformation')
    
    args = parser.parse_args()
    convert_unified_dataset(
        os.path.expanduser(args.path),
        os.path.expanduser(args.target),
        args.gaussian_norm,
        args.use_qpos,
        use_coordinate_transform=not args.no_coordinate_transform
    )