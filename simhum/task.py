# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, IterableDataset

from simhum.replay_buffer import IterableWrapper

_TEST_WORKERS = 4

def seed_worker(worker_id):
    initial_seed = torch.initial_seed()
    np.random.seed(initial_seed % (2**32) + worker_id)

def _build_data_loader(buffer, batch_size, num_workers, is_train=False, human_ratio=None):
    # Key insight: For Accelerate, we should NOT convert regular datasets to IterableDataset
    # Let Accelerate handle distributed sampling automatically for regular datasets
    if is_train and not isinstance(buffer, IterableDataset):
        # Check if we're in distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # For distributed training, keep as regular Dataset
            # Accelerate's prepare() will automatically add DistributedSampler
            # DO NOT wrap with IterableWrapper - this causes performance issues!
            pass
        else:
            # Single GPU training - convert to IterableWrapper as before
            buffer = IterableWrapper(buffer, human_ratio=human_ratio)
    return DataLoader(
        buffer,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not isinstance(buffer, IterableDataset),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        worker_init_fn=seed_worker,
        prefetch_factor=2 if num_workers > 0 else None,  # Increased from 2 to 4
        timeout=300 if num_workers > 0 else 0,  # Add timeout to prevent deadlocks
    )


class DefaultTask:
    def __init__(
        self,
        train_buffer,
        test_buffer,
        n_cams,
        obs_dim,
        ac_dim,
        batch_size,
        num_workers,
        human_ratio=None,
    ):
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.train_loader = _build_data_loader(
            train_buffer, batch_size, num_workers, is_train=True, human_ratio=human_ratio
        )
        if test_buffer is not None:
            test_workers = min(num_workers, _TEST_WORKERS)
            self.test_loader = _build_data_loader(test_buffer, batch_size, test_workers)

    def eval(self, trainer, global_step):
        losses = []
        for batch in self.test_loader:
            with torch.no_grad():
                loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())

        mean_val_loss = np.mean(losses)
        print(f"Step: {global_step}\tVal Loss: {mean_val_loss:.4f}")
        if wandb.run is not None:
            wandb.log({"eval/task_loss": mean_val_loss}, step=global_step)


class BCTask(DefaultTask):
    def _extract_ground_truth_components(self, actions, mask, action_dims):
        """
        Extract ground truth components for hybrid evaluation.
        
        Args:
            actions: [B, T, max_dim] - Ground truth actions
            mask: [B, T, max_dim] - Action mask
            action_dims: [B] - Action dimensions
        
        Returns:
            gt_dict: Dictionary with ground truth components
        """
        B, T = actions.shape[:2]
        device = actions.device
       
        # Initialize outputs
        robot_mask = (action_dims == 16)
        human_mask = ~robot_mask
        gt_dict = {}
        
        if robot_mask.any():
            # Robot samples: extract first 16 dims
            gt_dict['robot'] = {
                'actions': actions[robot_mask, :, :16],
                'mask': mask[robot_mask, :, :16]
            }
        
        if human_mask.any():
            # Human samples: extract robot and hands components using index mapping
            # Use the same indices as in the model
            HUMAN_ACTION_INDICES = {
                'left_eef': slice(0, 7),
                'left_finger': slice(7, 22),
                'left_gripper': 22,
                'right_eef': slice(23, 30),
                'right_finger': slice(30, 45),
                'right_gripper': 45
            }
            
            human_actions = actions[human_mask]  # [human_count, T, 46]
            human_masks = mask[human_mask]       # [human_count, T, 46]
            human_count = human_mask.sum()
            if human_actions.shape[-1] == 44:
                
                gt_dict['human_hands'] = {
                    'actions': human_actions,
                    'mask': human_masks
                }
                return gt_dict
            # Extract human_robot (16-dim: EEF + grippers)
            human_robot_actions = torch.zeros(human_count, T, 16, device=device)
            human_robot_masks = torch.zeros(human_count, T, 16, device=device)
            
            # left_eef, left_gripper, right_eef, right_gripper
            human_robot_actions[:, :, 0:7] = human_actions[:, :, HUMAN_ACTION_INDICES['left_eef']]
            human_robot_actions[:, :, 7:8] = human_actions[:, :, HUMAN_ACTION_INDICES['left_gripper']:HUMAN_ACTION_INDICES['left_gripper']+1]
            human_robot_actions[:, :, 8:15] = human_actions[:, :, HUMAN_ACTION_INDICES['right_eef']]
            human_robot_actions[:, :, 15:16] = human_actions[:, :, HUMAN_ACTION_INDICES['right_gripper']:HUMAN_ACTION_INDICES['right_gripper']+1]
            
            human_robot_masks[:, :, 0:7] = human_masks[:, :, HUMAN_ACTION_INDICES['left_eef']]
            human_robot_masks[:, :, 7:8] = human_masks[:, :, HUMAN_ACTION_INDICES['left_gripper']:HUMAN_ACTION_INDICES['left_gripper']+1]
            human_robot_masks[:, :, 8:15] = human_masks[:, :, HUMAN_ACTION_INDICES['right_eef']]
            human_robot_masks[:, :, 15:16] = human_masks[:, :, HUMAN_ACTION_INDICES['right_gripper']:HUMAN_ACTION_INDICES['right_gripper']+1]
            
            # Extract human_hands (44-dim: EEF + fingers)
            human_hands_actions = torch.zeros(human_count, T, 44, device=device)
            human_hands_masks = torch.zeros(human_count, T, 44, device=device)
            
            human_hands_actions[:, :, 0:7] = human_actions[:, :, HUMAN_ACTION_INDICES['left_eef']]
            human_hands_actions[:, :, 7:22] = human_actions[:, :, HUMAN_ACTION_INDICES['left_finger']]
            human_hands_actions[:, :, 22:29] = human_actions[:, :, HUMAN_ACTION_INDICES['right_eef']]
            human_hands_actions[:, :, 29:44] = human_actions[:, :, HUMAN_ACTION_INDICES['right_finger']]
            
            human_hands_masks[:, :, 0:7] = human_masks[:, :, HUMAN_ACTION_INDICES['left_eef']]
            human_hands_masks[:, :, 7:22] = human_masks[:, :, HUMAN_ACTION_INDICES['left_finger']]
            human_hands_masks[:, :, 22:29] = human_masks[:, :, HUMAN_ACTION_INDICES['right_eef']]
            human_hands_masks[:, :, 29:44] = human_masks[:, :, HUMAN_ACTION_INDICES['right_finger']]
            
            gt_dict['human_pseudo_gripper'] = {
                'actions': human_robot_actions,
                'mask': human_robot_masks
            }
            gt_dict['human_hands'] = {
                'actions': human_hands_actions,
                'mask': human_hands_masks
            }
        
        return gt_dict
    
    def _calculate_component_metrics(self, pred_actions, gt_actions, gt_mask):
        """Calculate L2 and LSig metrics for a component."""
        # Calculate L2 loss
        l2_delta = torch.square(gt_mask * (pred_actions - gt_actions))
        l2_delta = l2_delta.sum((1, 2)) / gt_mask.sum((1, 2))
        
        # Calculate sign agreement loss
        lsig = torch.logical_or(
            torch.logical_and(gt_actions > 0, pred_actions <= 0),
            torch.logical_and(gt_actions <= 0, pred_actions > 0),
        )
        lsig = (lsig.float() * gt_mask).sum((1, 2)) / gt_mask.sum((1, 2))
        
        return l2_delta.mean().item(), lsig.mean().item()

    def eval(self, trainer, global_step):
        losses = []
        
        # Initialize metric collectors
        action_l2, action_lsig = [], []
        robot_l2, robot_lsig = [], []
        human_robot_l2, human_robot_lsig = [], []
        human_hands_l2, human_hands_lsig = [], []
        
        for batch in self.test_loader:
            (imgs, obs), actions, mask = batch
            imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
            obs = {k: v.to(trainer.device_id) for k, v in obs.items()}
            actions, mask = [
                ar.to(trainer.device_id) for ar in (actions, mask)
            ]

            with torch.no_grad():
                loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())

                # Get predictions
                pred_actions = trainer.model.get_actions(imgs, obs)
                # Check if predictions are dictionary (hybrid mode) or tensor (robot_only mode)
                if isinstance(pred_actions, dict):
                    
                    # Hybrid mode: calculate component-wise metrics
                    action_dims = obs.get('action_dim', None)
                    if action_dims is not None:
                        gt_dict = self._extract_ground_truth_components(actions, mask, action_dims)
                        
                        # Calculate metrics for each component
                        robot_mask = (action_dims == 16)
                        human_mask = ~robot_mask
                        # import ipdb; ipdb.set_trace()
                        if robot_mask.any() and 'robot' in gt_dict:
                            l2, lsig = self._calculate_component_metrics(
                                pred_actions['robot'],
                                gt_dict['robot']['actions'],
                                gt_dict['robot']['mask']
                            )
                            robot_l2.append(l2)
                            robot_lsig.append(lsig)
                        # import ipdb; ipdb.set_trace()
                        if human_mask.any() and 'human_pseudo_gripper' in gt_dict:
                            l2, lsig = self._calculate_component_metrics(
                                pred_actions['human_pseudo_gripper'],
                                gt_dict['human_pseudo_gripper']['actions'],
                                gt_dict['human_pseudo_gripper']['mask']
                            )
                            human_robot_l2.append(l2)
                            human_robot_lsig.append(lsig)
                        if human_mask.any() and 'human_hands' in gt_dict:
                            l2, lsig = self._calculate_component_metrics(
                                pred_actions['human_hands'],
                                gt_dict['human_hands']['actions'],
                                gt_dict['human_hands']['mask']
                            )
                            human_hands_l2.append(l2)
                            human_hands_lsig.append(lsig)
                else:
                    # Robot-only mode: use original single-action evaluation
                    l2_delta = torch.square(mask * (pred_actions - actions))
                    l2_delta = l2_delta.sum((1, 2)) / mask.sum((1, 2))

                    lsig = torch.logical_or(
                        torch.logical_and(actions > 0, pred_actions <= 0),
                        torch.logical_and(actions <= 0, pred_actions > 0),
                    )
                    lsig = (lsig.float() * mask).sum((1, 2)) / mask.sum((1, 2))

                    action_l2.append(l2_delta.mean().item())
                    action_lsig.append(lsig.mean().item())

        # Calculate and log results
        mean_val_loss = np.mean(losses)
        
        if action_l2:  # Robot-only mode
            ac_l2, ac_lsig = np.mean(action_l2), np.mean(action_lsig)
            print(f"Step: {global_step}\tVal Loss: {mean_val_loss:.4f}")
            print(f"Step: {global_step}\tAC L2={ac_l2:.2f}\tAC LSig={ac_lsig:.2f}")

            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/task_loss": mean_val_loss,
                        "eval/action_l2": ac_l2,
                        "eval/action_lsig": ac_lsig,
                    },
                    step=global_step,
                )
        else:  # Hybrid mode
            print(f"Step: {global_step}\tVal Loss: {mean_val_loss:.4f}")
            
            log_dict = {"eval/task_loss": mean_val_loss}
            
            if robot_l2:
                robot_l2_mean, robot_lsig_mean = np.mean(robot_l2), np.mean(robot_lsig)
                print(f"Step: {global_step}\tRobot L2={robot_l2_mean:.2f}\tRobot LSig={robot_lsig_mean:.2f}")
                log_dict.update({
                    "eval/robot_l2": robot_l2_mean,
                    "eval/robot_lsig": robot_lsig_mean,
                })
            
            if human_robot_l2:
                hr_l2_mean, hr_lsig_mean = np.mean(human_robot_l2), np.mean(human_robot_lsig)
                print(f"Step: {global_step}\tHuman-Robot L2={hr_l2_mean:.2f}\tHuman-Robot LSig={hr_lsig_mean:.2f}")
                log_dict.update({
                    "eval/human_pseudo_gripper_l2": hr_l2_mean,
                    "eval/human_pseudo_gripper_lsig": hr_lsig_mean,
                })
            
            if human_hands_l2:
                hh_l2_mean, hh_lsig_mean = np.mean(human_hands_l2), np.mean(human_hands_lsig)
                print(f"Step: {global_step}\tHuman-Hands L2={hh_l2_mean:.2f}\tHuman-Hands LSig={hh_lsig_mean:.2f}")
                log_dict.update({
                    "eval/human_hands_l2": hh_l2_mean,
                    "eval/human_hands_lsig": hh_lsig_mean,
                })

            if wandb.run is not None:
                wandb.log(log_dict, step=global_step)
