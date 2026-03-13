
import os
import pickle as pkl
import numpy as np
import cv2
from tqdm import tqdm
import glob
import h5py
import json
import argparse
import torch
from simhum.models.qwen import QwenEmbedder
"""
Data Converter for Robotics Tasks

Usage Examples:
    # Example with specific path
    python data_converter.py --path /path/to/datasets/robotwin/ --target /path/to/datasets/robotwin/processed/all_tasks

    # Process all tasks (multi-task)
    python data_converter.py --path /path/to/data --target /path/to/output
    # Output to: /path/to/output/multi_task_5

    # Process single task
    python data_converter.py --path /path/to/data --target /path/to/output --tasks stack_bowls_two
    # Output to: /path/to/output/stack_bowls_two

    # Process specified multiple tasks
    python data_converter.py --path /path/to/data --target /path/to/output --tasks stack_bowls_two click_bell
    # Output to: /path/to/output/multi_task_2

Data Structure:
    /path/to/data/
     ├── task1/
     │   ├── data/
     │   │   ├── episode0.hdf5
     │   │   │   ├── endpose
     │   │   │   │   |──left_gripper
     │   │   │   │   |──left_endpose
     │   │   │   │   |──right_gripper
     │   │   │   │   |──right_endpose
     │   │   │   ├── joint_action/
     │   │   │   │   ├── vector
     │   │   │   │   └── ...
     │   │   │   └── observation/
     │   │   │       ├── head_camera
     │   │   │       ├── left_camera
     │   │   │       ├── right_camera
     │   │   ├── episode1.hdf5
     │   │   └── ...
     │   └── instruction.txt
     ├── task2/
     │   ├── data/
     │   │   ├── episode0.hdf5
     │   │   ├── episode1.hdf5
     │   │   └── ...
     │   └── instruction.txt
     └── ...  
"""

IMAGE_SIZE = (256, 256)
CAM_NAMES = ['head_camera', 'left_camera', 'right_camera']
TASK_NAMES = ['stack_bowls_two', 'place_bread_basket', 'click_bell', 'grab_roller', 'put_object_cabinet']

def crawler(base_path, task_names):
    """
    Get hdf5 files from specified task name directories under base_path
    """
    all_files = []
    for id, task_name in enumerate(task_names):
        task_path = os.path.join(base_path, task_name)
        if os.path.exists(task_path):
            files = glob.glob(os.path.join(task_path, '**/*.hdf5'), recursive=True)

            for i in files:
                all_files.append((id, i))
          
            print(f"Found {len(files)} files in {task_name}")
        else:
            print(f"Warning: Task directory {task_path} does not exist")
    return all_files



def _resize_and_encode(bgr_image, size=IMAGE_SIZE):
   
    resize_height, resize_width = size
    frame = cv2.imdecode(np.frombuffer(bgr_image, np.uint8), cv2.IMREAD_COLOR)
    # First resize by height ratio
    original_height, original_width = frame.shape[:2]
    scale = resize_height / original_height
    new_width = int(original_width * scale)
    resized = cv2.resize(frame, (new_width, resize_height), interpolation=cv2.INTER_AREA)
    
    # Then center crop width to target size
    if new_width > resize_width:
        # Need to crop width
        start_x = (new_width - resize_width) // 2
        resized = resized[:, start_x:start_x + resize_width]
    elif new_width < resize_width:
        # Need to pad width
        pad_left = (resize_width - new_width) // 2
        pad_right = resize_width - new_width - pad_left
        resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    _, encoded = cv2.imencode(".jpg", resized)
    return encoded

def _decode_bgr_image(img_str):
    return cv2.imdecode(img_str, 1)


def _to_np(grip_value):
    return np.array([grip_value])


def _gaussian_norm(all_acs):
    print('Using gaussian norm')
    all_acs_arr = np.array(all_acs)
    mean = np.mean(all_acs_arr, axis=0)
    std =  np.std(all_acs_arr, axis=0)
    if not std.all(): # handle situation w/ all 0 actions
        std[std == 0] = 1e-17

    for a in all_acs:
        a -= mean
        a /= std

    return dict(loc=mean.tolist(), scale=std.tolist())


def _max_min_norm(all_acs):
    print('Using max min norm')
    all_acs_arr = np.array(all_acs)
    max_ac = np.max(all_acs_arr, axis=0)
    min_ac = np.min(all_acs_arr, axis=0)

    mid = (max_ac + min_ac) / 2
    delta = (max_ac - min_ac) / 2

    for a in all_acs:
        a -= mid
        a /= delta
    return dict(loc=mid.tolist(), scale=delta.tolist())

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        
      left_gripper, left_arm = (
          root["/endpose/left_gripper"][()],
          root["/endpose/left_endpose"][()],
      )
      right_gripper, right_arm = (
          root["/endpose/right_gripper"][()],
          root["/endpose/right_endpose"][()],
      )
      left_gripper = left_gripper[..., np.newaxis]
      right_gripper = right_gripper[..., np.newaxis]
      vector_ee = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
        
      vector_qpos = root["/joint_action/vector"][()]
      image_dict = dict()
      for cam_name in root[f"/observation/"].keys():
          image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return vector_ee, vector_qpos, image_dict


def cal_instruction_embedding(base_path, target_path, task_names):
    # Initialize DistilBERT embedder for instruction processing
    embedder = QwenEmbedder(device='cuda' if torch.cuda.is_available() else 'cpu')
    task_instruction = []
    for task in task_names:
        path = os.path.join(base_path, task, 'aloha-agilex_clean_50', 'instruction.txt')
        assert os.path.exists(path), f"Instruction file not found in {path}"
        with open(path, 'r') as f:
            task_instruction.append(f.read().strip())
    task_embeddings = []
    for instruction in task_instruction:
        task_embeddings.append(embedder.get_pooled_embeddings(instruction).cpu().numpy())
    embeddings_file = os.path.join(target_path, "task_embeddings.pkl")

    with open(embeddings_file, 'wb') as f:
        pkl.dump(task_embeddings, f)    
    with open(os.path.join(target_path, 'task_instruction.txt'), 'w') as f:
        for instruction in task_instruction:    
            f.write(instruction + '\n')
    print(f"Process {len(task_names)} task instructions.")
    print(f"Task embeddings saved to {embeddings_file}")
    print(f"Task instruction saved to {os.path.join(target_path, 'task_instruction.txt')}")

def convert_dataset(base_path, target_path, gaussian_norm, task_names):
    print(f'gaussian_norm={gaussian_norm}')
    print(f'Processing tasks: {task_names}')
    print()
    
    # Determine the final target path
    if len(task_names) > 1:
        final_target_path = os.path.join(target_path, f'multi_task_{len(task_names)}')
    else:
        final_target_path = os.path.join(target_path, task_names[0])
    
    print(f'Target path: {final_target_path}')
    if not os.path.exists(final_target_path):
        os.makedirs(final_target_path)
    cal_instruction_embedding(base_path, final_target_path, task_names )
   
    episode_paths = crawler(base_path, task_names)
    print(f'Total episodes found: {len(episode_paths)}')
    # Randomly shuffle episode_paths
    import random
    random.shuffle(episode_paths)

    out_trajs, all_acs, all_states = [], [], []
    for id, episode_path in tqdm(episode_paths):
        proc_traj = []
        vector_ee, _, image_dict = load_hdf5(episode_path)
       
        action_arrays, state_arrays, img_arrays = (
        [],
        [],
        [],
      )
        for j in range(0, vector_ee.shape[0]):
            obs = {}
            for idx, key in enumerate(CAM_NAMES):
                obs[f'enc_cam_{idx}'] = _resize_and_encode(image_dict[key][j])
            img_arrays.append(obs)
            if j != vector_ee.shape[0] - 1:
                state_arrays.append(vector_ee[j])
                all_states.append(vector_ee[j])
            if j != 0:
                action_arrays.append(vector_ee[j])
                all_acs.append(vector_ee[j])
        for j in range(0, len(img_arrays)-1):
            obs = img_arrays[j]
            obs['state'] = state_arrays[j]
            obs['instruction_id'] = id
            proc_traj.append((obs, action_arrays[j], 0))
            # for t, a in enumerate(actions):
            #     all_acs.append(a) # for normalization later

            #     reward = 0 # dummy reward
            #     obs = dict(state=f['observations']['qpos'][t])
                
            #     for idx, cam_name in enumerate(CAM_NAMES):
            #         bgr_img = _decode_bgr_image(f['observations'][cam_name]['rgb'][t])
            #         obs[f'enc_cam_{idx}'] = _resize_and_encode(bgr_img)
            #     proc_traj.append((obs, a, reward))
        out_trajs.append(proc_traj)
    ac_dict = _max_min_norm(all_acs) if not gaussian_norm \
              else _gaussian_norm(all_acs)
    state_dict = _max_min_norm(all_states) if not gaussian_norm \
              else _gaussian_norm(all_states)
    
    with open(os.path.join(final_target_path, 'ac_norm.json'), 'w') as f:
        json.dump(ac_dict, f)
    with open(os.path.join(final_target_path, 'state_norm.json'), 'w') as f:
        json.dump(state_dict, f)
    with open(os.path.join(final_target_path, 'buf.pkl'), 'wb') as f:
        pkl.dump(out_trajs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='/path/to/datasets/robotwin/clean/origin_data')
    parser.add_argument('-target', '--target', default='/path/to/datasets/robotwin/clean/processed')
    parser.add_argument('--gaussian_norm', action='store_true')
    parser.add_argument('--tasks', nargs='+', default=TASK_NAMES, help='List of task names to process')
    args = parser.parse_args()
    convert_dataset(os.path.expanduser(args.path), os.path.expanduser(args.target), args.gaussian_norm, args.tasks)