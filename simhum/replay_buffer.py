# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle as pkl
import random
import shutil
import json
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch
import tqdm
from robobuf import ReplayBuffer as RB, Transition, ObsWrapper
from torch.utils.data import Dataset, IterableDataset
from simhum.transforms import RelativeTransformer, DualArmNormalizer
        
# cache loading from the buffer list to half memory overhead
buf_cache = dict()
BUF_SHUFFLE_RNG = 3904767649
# Import IMAGE_SIZE from data processing constants

from data_processing.processors.common.constants import IMAGE_SIZE


# helper functions
_img_to_tensor = (
    lambda x: torch.from_numpy(x.copy()).permute((0, 3, 1, 2)).float() / 255
)
_to_tensor = lambda x: torch.from_numpy(x).float()


def _cached_load(path):
    global buf_cache

    if path in buf_cache:
        return buf_cache[path]
    with open(path, "rb") as f:
        buf = pkl.load(f)
        if len(buf[0]) == 3:
            buf = [buf]
        # Custom loading with (240, 320) image size
        buf = _load_traj_list_with_custom_size(buf, H=IMAGE_SIZE[0], W=IMAGE_SIZE[1])
    buf_cache[path] = buf
    return buf


def _load_traj_list_with_custom_size(traj_list, H=240, W=320):
    """Custom version of RB.load_traj_list that allows specifying image dimensions"""
    buffer = RB()
    
    for traj in traj_list:
        trans_count = 0
        for i, trans in enumerate(traj):
            obs, action, reward = trans
            if "actor" in obs.keys() and obs["actor"] in ["ai_agent"]:
                continue
            # Use custom ObsWrapper with specified H, W
            transition = Transition(ObsWrapper(obs, H=H, W=W), action, reward)
            buffer.add(transition, trans_count == 0)
            trans_count += 1
    
    return buffer



def _get_imgs(t, cam_idx, past_frames):
    imgs = []
    while len(imgs) < past_frames + 1:
        imgs.append(t.obs.image(cam_idx)[None])

        if t.prev is not None:
            t = t.prev
    return np.concatenate(imgs, axis=0)


def _parallel_relative_transform(args):
    """
    Multiprocessing worker function: performs relative position transformation on extracted actions and states.

    Args:
        args: tuple of (batch_actions, batch_states, data_type)
            batch_actions: np.array of shape (batch_size, ac_chunk, action_dim)
            batch_states: np.array of shape (batch_size, state_dim)
            data_type: 'robot' or 'human'

    Returns:
        np.array: processed_actions of shape (batch_size*ac_chunk, action_dim)
    """
    batch_actions, batch_states, data_type = args

    # Each worker creates its own RelativeTransformer instance
    relative_transformer = RelativeTransformer()

    batch_relative_actions = []
    for i in range(len(batch_actions)):
        # Perform relative position transformation for all timesteps of each sample
        action_for_norm = relative_transformer.forward(
            batch_actions[i], batch_states[i], backward=False, data_type=data_type
        )
        # action_for_norm: (ac_chunk, action_dim)
        batch_relative_actions.append(action_for_norm)

    # Merge all actions in the current batch: (batch_size*ac_chunk, action_dim)
    processed_actions = np.concatenate(batch_relative_actions, axis=0)
    return processed_actions



class IterableWrapper(IterableDataset):
    def __init__(self, wrapped_dataset, human_ratio=None, max_count=float("inf")):
        self.wrapped = wrapped_dataset
        self.ctr, self.max_count = 0, max_count
        self.human_ratio = human_ratio
        self.robot_human_bound = wrapped_dataset.robot_human_bound

    def __iter__(self):
        self.ctr = 0
        return self

    def __next__(self):
        if self.ctr > self.max_count:
            raise StopIteration
        
        self.ctr += 1

        if self.human_ratio is not None and self.robot_human_bound != len(self.wrapped):
            if np.random.random() < self.human_ratio:
                # Human data: [robot_human_bound, len(wrapped))
                idx = int(np.random.randint(self.robot_human_bound, len(self.wrapped)))
            else:
                # Robot data: [0, robot_human_bound)
                idx = int(np.random.randint(0, self.robot_human_bound))
        else:
            idx = int(np.random.choice(len(self.wrapped)))

        return self.wrapped[idx]

    def __len__(self):
        return len(self.wrapped)


   
import pickle

class RobobufReplayBuffer(Dataset):
    def __init__(
        self,
        buffer_path,
        transform=None,
        n_test_trans=500,
        mode="train",
        ac_chunk=1,
        cam_indexes=[0],
        goal_indexes=[],
        goal_geom_prob=0.01,
        past_frames=0,
        ac_dim=7,
        use_relative_action=False,
    ):
        self.buffer_path = buffer_path  # Save buffer_path for get_normalizer()
        self.use_relative_action = use_relative_action
        assert mode in ("train", "test"), "Mode must be train/test"
        buf = _cached_load(buffer_path)
        assert len(buf) > n_test_trans, "Not enough transitions!"

        norm_file = os.path.join(os.path.dirname(buffer_path), "ac_norm.json")
        
        if os.path.exists(norm_file):
            shutil.copyfile(norm_file, "./ac_norm.json")

        self.human_indices, self.robot_indices = self._find_data_boundaries_binary(buf)

        print(f"Analysis complete: {len(self.human_indices)} human samples, {len(self.robot_indices)} robot samples")

        # Compute normalization parameters early (before building s_a_mask)
        # Separately compute for human and robot data
        if self.use_relative_action:
            self.relative_transformer = RelativeTransformer()

        self.human_normalizer = None
        self.robot_normalizer = None

        if len(self.human_indices) > 0:
            self.human_normalizer = self.get_normalizer(buf, self.human_indices, ac_chunk, data_type='human')
        if len(self.robot_indices) > 0:
            self.robot_normalizer = self.get_normalizer(buf, self.robot_indices, ac_chunk, data_type='robot')

         # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)

        # get and shuffle list of buf indices
        rng.shuffle(self.human_indices)
        rng.shuffle(self.robot_indices)

        # split data according to mode with balanced sampling for test set
        half_test = n_test_trans // 2
        if mode == "test":
            # Take first half and last half to ensure human/robot balance
            if len(self.human_indices) == 0:
                index_list = self.robot_indices[:n_test_trans]
                self.robot_human_bound = len(index_list)
            elif len(self.robot_indices) == 0:
                index_list = self.human_indices[:n_test_trans]
                self.robot_human_bound = len(index_list)
            else :
                index_list = self.robot_indices[:half_test] + self.human_indices[:half_test] 
                self.robot_human_bound = len(self.robot_indices[:half_test])
        else:
            # Training mode: exclude first and last half_test samples
            if len(self.human_indices) == 0:
                index_list = self.robot_indices[n_test_trans:]
                self.robot_human_bound = len(index_list)
            elif len(self.robot_indices) == 0:
                index_list = self.human_indices[n_test_trans:]
                self.robot_human_bound = len(index_list)
            else :
                index_list = self.robot_indices[half_test:] + self.human_indices[half_test:]
                self.robot_human_bound = len(self.robot_indices[half_test:])
            
        self.transform = transform
        self.s_a_mask = []

        self.cam_indexes = cam_indexes = list(cam_indexes)
        self.past_frames = past_frames
        print(f"Building {mode} buffer with cam_indexes={cam_indexes}")

        self.goal_geom_prob = goal_geom_prob
        self.goal_indexes = set(goal_indexes)
        assert all([g in self.cam_indexes for g in self.goal_indexes])

        # First, build s_a_mask (needed for computing normalization)
        for idx in tqdm.tqdm(index_list):
            t = buf[idx]
            loop_t, chunked_actions, loss_mask = t, [], []
            for _ in range(ac_chunk):
                chunked_actions.append(loop_t.action[None])
                loss_mask.append(1.0)

                if loop_t.next is None:
                    break
                loop_t = loop_t.next

            if len(chunked_actions) < ac_chunk:
                for _ in range(ac_chunk - len(chunked_actions)):
                    chunked_actions.append(chunked_actions[-1])
                    loss_mask.append(0.0)

            a_t = np.concatenate(chunked_actions, 0).astype(np.float32)
            assert ac_dim == a_t.shape[-1]

            loss_mask = np.array(loss_mask, dtype=np.float32)
            self.s_a_mask.append((t, a_t, loss_mask, loop_t))

    def _find_data_boundaries_binary(self, buf):
        """Use binary search to quickly find human/robot data boundaries"""
        n = len(buf)
        if n == 0:
            return [], []

        human_indices = []
        robot_indices = []
        boundaries = []
        current_pos = 0

        print(f"Binary search analysis: {n} samples")

        while current_pos < n:
            # current_dim = self._get_action_dim(wrapped_dataset[current_pos])
            
            current_dim = self._get_action_dim(buf, current_pos)
            # Binary search to find the next position with a different dimension
            next_boundary = self._binary_search_boundary(buf, current_pos, n - 1, current_dim)

            boundaries.append({
                'start': current_pos,
                'end': next_boundary,
                'action_dim': current_dim,
                'count': next_boundary - current_pos + 1
            })

            current_pos = next_boundary + 1

        # Print discovered boundary information
        for i, boundary in enumerate(boundaries):
            data_type = "human" if boundary['action_dim'] != 16 else "robot"
            print(f"Boundary {i+1}: {data_type} data [{boundary['start']}:{boundary['end']}] "
                  f"({boundary['count']} samples, action_dim={boundary['action_dim']})")

        # Build human_indices and robot_indices based on boundaries
        for boundary in boundaries:
            indices = list(range(boundary['start'], boundary['end'] + 1))
            if boundary['action_dim'] != 16:  # human
                human_indices.extend(indices)
            else:  # robot (typically 16D)
                robot_indices.extend(indices)

        return human_indices, robot_indices
    
    def _binary_search_boundary(self, buf, start, end, target_dim):
        """Binary search to find the boundary where dimension changes"""
        while start < end:
            mid = (start + end + 1) // 2
            mid_dim = self._get_action_dim(buf, mid)

            if mid_dim == target_dim:
                start = mid
            else:
                end = mid - 1

        return start

    def _get_action_dim(self, buf, current_pos):
        """Quickly get the action_dim of a sample"""
        step = buf[current_pos]
        current_dim = step.obs.obs['action_dim']
        return current_dim

    def get_normalizer(self, buf, indices, ac_chunk, data_type='human'):
        """
        Compute and return a normalizer (similar to UMI's approach) for a specific data type (human or robot).

        Key strategy:
        - State: uses normalization parameters from **absolute positions**
        - Action: determined by self.use_relative_action:
          - True: uses normalization parameters from relative positions (relative to current state)
          - False: uses normalization parameters from absolute positions

        Workflow:
        1. Check if normalization parameter files already exist; if so, load them directly
        2. If not, iterate over specified indices to compute normalization parameters:
           - State: collect all absolute position states, compute statistics
           - Action: convert to relative/absolute positions based on use_relative_action, compute statistics
           - Save normalization parameters to JSON files
        3. Return a DualArmNormalizer object

        Normalization strategy (similar to UMI):
        - xyz position: range normalization (min-max)
        - quaternion: identity normalization (not normalized)
        - gripper: range normalization (min-max)
        - fingertips: range normalization (min-max)

        Args:
            buf: ReplayBuffer object
            indices: List of data indices for computing normalization parameters
            ac_chunk: Action chunk size
            data_type: Data type ('human' or 'robot')

        Returns:
            DualArmNormalizer: Normalizer instance
        """
        buffer_dir = os.path.dirname(self.buffer_path)

        # Normalization file paths - named separately by data type
        # action: uses relative or absolute positions based on use_relative_action
        # state: based on absolute positions
        if self.use_relative_action:
            ac_norm_file = os.path.join(buffer_dir, f"{data_type}_relative_ac_norm.json")
        else:
            ac_norm_file = os.path.join(buffer_dir, f"{data_type}_absolute_ac_norm.json")
        state_norm_file = os.path.join(buffer_dir, f"{data_type}_absolute_state_norm.json")

        # Check if normalization parameter files already exist
        if os.path.exists(ac_norm_file) and os.path.exists(state_norm_file):
            action_type = "relative" if self.use_relative_action else "absolute"
            print(f"✓ Loading existing {data_type} normalization parameters:")
            print(f"  - Action ({action_type}): {ac_norm_file}")
            print(f"  - State (absolute): {state_norm_file}")
            normalizer = DualArmNormalizer.from_json(ac_norm_file, state_norm_file)
            print(f"  - {normalizer}")
            return normalizer

        # If files do not exist, iterate over the dataset to compute normalization parameters
        action_type = "relative" if self.use_relative_action else "absolute"
        print("="*70)
        print(f"Computing {data_type} normalization parameters (similar to UMI)...")
        print("  - State: from absolute poses")
        print(f"  - Action: from {action_type} poses")
        print("="*70)

        # Stage 1: Collect data (single process, avoid serialization issues)
        print(f"Stage 1: Collecting {len(indices)} {data_type} samples from buffer...")

        # Pre-fetch data dimensions for memory pre-allocation
        sample_t = buf[indices[0]]
        action_dim = sample_t.action.shape[-1]
        state_dim = sample_t.obs.state.shape[-1]

        # Pre-allocate full arrays to avoid large-scale merging later
        total_samples = len(indices)
        all_actions = np.empty((total_samples, ac_chunk, action_dim), dtype=np.float32)
        all_states = np.empty((total_samples, state_dim), dtype=np.float32)

        batch_size = 1000  # Process 1000 samples per batch
        num_batches = (len(indices) + batch_size - 1) // batch_size

        for batch_idx in tqdm.tqdm(range(num_batches), desc=f"Collecting {data_type} data"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]

            for i, idx in enumerate(batch_indices):
                t = buf[idx]
                global_idx = start_idx + i

                # Write directly into pre-allocated array
                all_states[global_idx] = t.obs.state

                # Collect action chunk
                loop_t, chunked_actions = t, []
                for _ in range(ac_chunk):
                    chunked_actions.append(loop_t.action[None])
                    if loop_t.next is None:
                        break
                    loop_t = loop_t.next

                # If fewer than ac_chunk, pad with the last action
                if len(chunked_actions) < ac_chunk:
                    for _ in range(ac_chunk - len(chunked_actions)):
                        chunked_actions.append(chunked_actions[-1])

                a_t = np.concatenate(chunked_actions, 0).astype(np.float32)
                all_actions[global_idx] = a_t

        # Stage 2: Parallel relative position transformation (if needed)
        if self.use_relative_action:
            print(f"Stage 2: Parallel relative transformation...")
            num_workers = max(1, min(cpu_count() // 4, num_batches))  # Use at most 1/4 CPU cores, not exceeding batch count
            print(f"Using {num_workers} worker processes...")

            # Prepare tasks: split data into batches for parallel processing
            tasks = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_samples)
                batch_actions = all_actions[start_idx:end_idx]
                batch_states = all_states[start_idx:end_idx]
                tasks.append((batch_actions, batch_states, data_type))

            with Pool(processes=num_workers) as pool:
                # Execute relative position transformation in parallel
                processed_actions_list = list(tqdm.tqdm(
                    pool.imap(_parallel_relative_transform, tasks),
                    total=len(tasks),
                    desc=f"Transforming {data_type} actions (parallel)"
                ))

            # Merge results - use pre-allocation to avoid repeated memory allocation
            total_action_samples = total_samples * ac_chunk
            actions_for_norm = np.empty((total_action_samples, action_dim), dtype=np.float32)

            offset = 0
            for batch_result in processed_actions_list:
                batch_len = len(batch_result)
                actions_for_norm[offset:offset + batch_len] = batch_result
                offset += batch_len
        else:
            # Directly use the first timestep's action - avoid list comprehension
            actions_for_norm = all_actions[:, 0, :]  # (N, action_dim)

        # States are already in the pre-allocated array, use directly
        absolute_states = all_states

        print(f"\nCollected {data_type} data:")
        print(f"  - {len(actions_for_norm)} {action_type} action samples")
        print(f"  - {len(absolute_states)} absolute state samples")

        # Compute normalization parameters (unified interface)
        action_norm_params = self._compute_norm_params(actions_for_norm, data_type)
        state_norm_params = self._compute_norm_params(absolute_states, data_type)

        # Save normalization parameters
        with open(ac_norm_file, 'w') as f:
            json.dump(action_norm_params, f, indent=2)
        with open(state_norm_file, 'w') as f:
            json.dump(state_norm_params, f, indent=2)

        print(f"\n✓ Saved {data_type} normalization parameters:")
        print(f"  - Action ({action_type}): {ac_norm_file}")
        print(f"  - State (absolute): {state_norm_file}")

        # Load and return the normalizer
        normalizer = DualArmNormalizer.from_json(ac_norm_file, state_norm_file)
        print(f"  - {normalizer}")

        return normalizer

    def _compute_norm_params(self, data, data_type):
        """
        Unified function for computing normalization parameters.

        Automatically identifies data dimensions and applies the corresponding normalization
        strategy based on data type (when mixing human and robot, robot data is zero-padded):

        Supported dimensions:
        - 16D: [left_arm(8), right_arm(8)]
               Each arm: [xyz(3), quat(4), gripper(1)]
        - 44D: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
        - 46D: [left_eef(7), left_fingertips(15), left_gripper(1),
                right_eef(7), right_fingertips(15), right_gripper(1)]

        Normalization strategy:
        - xyz position: range normalization (min-max)
        - quaternion: identity (not normalized)
        - gripper: range normalization (min-max)
        - fingertips: range normalization (min-max)

        Args:
            data: numpy array of shape (N, D) where D is 16, 44, or 46

        Returns:
            dict: {'loc': list, 'scale': list} normalization parameters
        """
        dim = data.shape[-1]
        loc = np.zeros(dim)
        scale = np.ones(dim)

        def apply_range_norm(loc, scale, data, indices):
            """Helper function: apply range normalization to specified indices"""
            loc[indices] = (data[:, indices].max(axis=0) + data[:, indices].min(axis=0)) / 2
            scale[indices] = (data[:, indices].max(axis=0) - data[:, indices].min(axis=0)) / 2
            scale[indices] = np.where(scale[indices] == 0, 1e-16, scale[indices])

        if dim == 16 or data_type == "robot":
            # 16D structure: [left_arm(8), right_arm(8)]
            # Or zero-padded robot data in 44D/46D: [left_arm(8), right_arm(8), zeros(28/30)]
            # Only compute normalization parameters for the first 16 valid dimensions
            # Left arm: xyz(0:3) + quat(3:7) + gripper(7:8)
            apply_range_norm(loc, scale, data, slice(0, 3))    # left xyz
            apply_range_norm(loc, scale, data, slice(7, 8))    # left gripper
            # Right arm: xyz(8:11) + quat(11:15) + gripper(15:16)
            apply_range_norm(loc, scale, data, slice(8, 11))   # right xyz
            apply_range_norm(loc, scale, data, slice(15, 16))  # right gripper
            # Note: for 44D/46D, the zero-padded portion keeps loc=0, scale=1 (identity transform)

        elif dim == 44:
            # 44D structure: [left_eef(7), left_fingertips(15), right_eef(7), right_fingertips(15)]
            # Left: eef_xyz(0:3) + eef_quat(3:7) + fingertips(7:22)
            apply_range_norm(loc, scale, data, slice(0, 3))    # left eef xyz
            apply_range_norm(loc, scale, data, slice(7, 22))   # left fingertips
            # Right: eef_xyz(22:25) + eef_quat(25:29) + fingertips(29:44)
            apply_range_norm(loc, scale, data, slice(22, 25))  # right eef xyz
            apply_range_norm(loc, scale, data, slice(29, 44))  # right fingertips

        elif dim == 46:
            # 46D structure: [left_eef(7), left_fingertips(15), left_gripper(1),
            #          right_eef(7), right_fingertips(15), right_gripper(1)]
            # Left: eef_xyz(0:3) + eef_quat(3:7) + fingertips(7:22) + gripper(22:23)
            apply_range_norm(loc, scale, data, slice(0, 3))    # left eef xyz
            apply_range_norm(loc, scale, data, slice(7, 22))   # left fingertips
            apply_range_norm(loc, scale, data, slice(22, 23))  # left gripper
            # Right: eef_xyz(23:26) + eef_quat(26:30) + fingertips(30:45) + gripper(45:46)
            apply_range_norm(loc, scale, data, slice(23, 26))  # right eef xyz
            apply_range_norm(loc, scale, data, slice(30, 45))  # right fingertips
            apply_range_norm(loc, scale, data, slice(45, 46))  # right gripper

        else:
            raise ValueError(f"Unsupported data dimension: {dim}. Expected 16, 44, or 46.")

        return {'loc': loc.tolist(), 'scale': scale.tolist()}

    def __len__(self):
        return len(self.s_a_mask)

    def __getitem__(self, idx):
        """
        Load and preprocess a single training sample.

        Pipeline:
        1. Load raw data (step, action, mask, goal)
        2. Sample goal frame (for goal-conditioned policy)
        3. Transform action to relative pose (if enabled)
        4. Build observation dict (state + metadata + language)
        5. Build image dict (multi-camera frames)
        6. Normalize and tensorize

        Returns:
            tuple: ((i_t, o_t), a_t, loss_mask)
                i_t: dict of camera images
                o_t: dict of observations (state, language, metadata)
                a_t: normalized actions
                loss_mask: action loss mask
        """
        # Step 1: Load raw data from buffer
        step, a_t, loss_mask, goal = self.s_a_mask[idx]

        # Step 2: Sample goal frame (for goal conditioning)
        if self.goal_indexes:
            while np.random.uniform() > self.goal_geom_prob and goal.next is not None:
                goal = goal.next

        # Step 3: Select appropriate normalizer based on data type
        action_dim = step.obs.obs.get('action_dim', step.obs.state.shape[-1])
        if action_dim == 44 or action_dim == 46:  # Human data
            normalizer = self.human_normalizer
            data_type = "human"
        else:  # Robot data (16D)
            normalizer = self.robot_normalizer
            data_type = "robot"

        # Step 4: Transform action (absolute -> relative if enabled)
        if self.use_relative_action:
            actions = self.relative_transformer.forward(a_t, step.obs.state, backward=False, data_type=data_type)
        else:
            actions = a_t

        a_t = _to_tensor(normalizer.normalize_action(actions))

        # Step 5: Build observation dict
        o_t = {}

        # 5.1 State (normalized absolute pose)
        o_t['state'] = _to_tensor(normalizer.normalize_state(step.obs.state))

        if 'action_dim' in step.obs.obs.keys():
            o_t['action_dim'] = step.obs.obs['action_dim']
        if 'human_gripper' in step.obs.obs:
            o_t['human_gripper'] = step.obs.obs['human_gripper']

        # Step 5: Build image dict (multi-camera)
        i_t = {}
        for cam_id, cam_idx in enumerate(self.cam_indexes):
            # 5.1 Get observation frames (current + history)
            i_c = _get_imgs(step, cam_idx, self.past_frames)

            # 5.2 Prepend goal frame (if goal conditioning enabled)
            if self.goal_indexes:
                if cam_idx in self.goal_indexes:
                    g_c = _get_imgs(goal, cam_idx, 0)
                else:
                    g_c = np.zeros_like(i_c[:1])
                i_c = np.concatenate((g_c, i_c), axis=0)

            # 5.3 Convert to tensor and apply augmentation
            i_c = _img_to_tensor(i_c)
            if self.transform is not None:
                i_c = self.transform(i_c)

            i_t[f"cam{cam_id}"] = i_c
        
        loss_mask = _to_tensor(loss_mask)[:, None].repeat((1, a_t.shape[-1]))

        # Sanity check
        assert loss_mask.shape[0] == a_t.shape[0], \
            f"Action and mask temporal dimension mismatch: {a_t.shape[0]} vs {loss_mask.shape[0]}"

        return (i_t, o_t), a_t, loss_mask
