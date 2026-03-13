# Copyright (c) Sudeep Dasari, 2023
# Modernized Accelerate Task Manager
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb
from simhum.task import BCTask


class AccelerateBCTask:
    """
    Modern Accelerate task manager
    - Automatically obtains accelerator instance from trainer
    - Intelligently handles distributed evaluation
    - Fully compatible with BCTask interface
    - Transparent data loading and metric aggregation
    """

    def __init__(self, base_task: BCTask, trainer=None):
        """
        Initialize Accelerate task manager

        Args:
            base_task: Base BCTask instance
            trainer: AccelerateBehaviorCloning trainer (optional, used to automatically obtain accelerator)
        """
        self.base_task = base_task
        self._trainer = trainer

        # Copy key attributes from the base task
        for attr in ['obs_dim', 'ac_dim', 'n_cams']:
            setattr(self, attr, getattr(base_task, attr))

        # Initialize state
        self._prepared = False
        self.train_loader = None
        self.test_loader = None

        # If trainer is available, prepare dataloaders immediately
        if trainer is not None:
            self._ensure_prepared()

    @property
    def accelerator(self):
        """Get the accelerator instance"""
        if self._trainer and hasattr(self._trainer, 'accelerator'):
            return self._trainer.accelerator
        else:
            raise RuntimeError("AccelerateBCTask requires a trainer with accelerator")

    def _ensure_prepared(self):
        """Lazily prepare dataloaders - only executed when needed"""
        if self._prepared or not self._trainer:
            return

        # Use trainer's accelerator to prepare dataloaders
        self.train_loader = self.accelerator.prepare(self.base_task.train_loader)

        if hasattr(self.base_task, 'test_loader'):
            self.test_loader = self.accelerator.prepare(self.base_task.test_loader)

        self._prepared = True

    def set_trainer(self, trainer):
        """Set trainer and prepare dataloaders"""
        self._trainer = trainer
        self._ensure_prepared()

    def eval(self, trainer, global_step):
        """
        Distributed evaluation - automatically aggregates results from all processes.
        Supports hybrid and robot-only modes.
        """
        # Ensure dataloaders are prepared
        if not self._prepared:
            self.set_trainer(trainer)

        if not hasattr(self, 'test_loader') or self.test_loader is None:
            return

        # Collect evaluation metrics
        losses = []
        action_l2, action_lsig = [], []
        robot_l2, robot_lsig = [], []
        human_robot_l2, human_robot_lsig = [], []
        human_hands_l2, human_hands_lsig = [], []

        # Run evaluation on all processes
        for batch in self.test_loader:
            (imgs, obs), actions, mask = batch

            with torch.no_grad():
                # Use trainer's autocast for evaluation
                with self.accelerator.autocast():
                    loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())

                # Get model predicted actions
                model = self.accelerator.unwrap_model(trainer.model)
                pred_actions = model.get_actions(imgs, obs)

                # Check prediction result type
                if isinstance(pred_actions, dict):
                    # Hybrid mode: handle dict-type predictions
                    action_dims = obs.get('action_dim', None)
                    if action_dims is not None:
                        gt_dict = self.base_task._extract_ground_truth_components(actions, mask, action_dims)

                        # Compute metrics for each component
                        robot_mask = (action_dims == 16)
                        human_mask = ~robot_mask

                        if robot_mask.any() and 'robot' in gt_dict:
                            l2, lsig = self.base_task._calculate_component_metrics(
                                pred_actions['robot'][robot_mask],
                                gt_dict['robot']['actions'],
                                gt_dict['robot']['mask']
                            )
                            robot_l2.append(l2)
                            robot_lsig.append(lsig)

                        if human_mask.any() and 'human_robot' in gt_dict:
                            l2, lsig = self.base_task._calculate_component_metrics(
                                pred_actions['human_robot'][human_mask],
                                gt_dict['human_robot']['actions'],
                                gt_dict['human_robot']['mask']
                            )
                            human_robot_l2.append(l2)
                            human_robot_lsig.append(lsig)

                        if human_mask.any() and 'human_hands' in gt_dict:
                            l2, lsig = self.base_task._calculate_component_metrics(
                                pred_actions['human_hands'][human_mask],
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

        # Aggregate results from all processes - all processes must participate
        device = self.accelerator.device
        all_losses = self.accelerator.gather_for_metrics(torch.tensor(losses, device=device))

        # Pre-prepare tensors for other metrics (all processes)
        if action_l2:  # Robot-only mode
            all_action_l2 = self.accelerator.gather_for_metrics(torch.tensor(action_l2, device=device))
            all_action_lsig = self.accelerator.gather_for_metrics(torch.tensor(action_lsig, device=device))
        else:  # Hybrid mode - all processes must participate in gather
            if robot_l2:
                all_robot_l2 = self.accelerator.gather_for_metrics(torch.tensor(robot_l2, device=device))
                all_robot_lsig = self.accelerator.gather_for_metrics(torch.tensor(robot_lsig, device=device))
            if human_robot_l2:
                all_hr_l2 = self.accelerator.gather_for_metrics(torch.tensor(human_robot_l2, device=device))
                all_hr_lsig = self.accelerator.gather_for_metrics(torch.tensor(human_robot_lsig, device=device))
            if human_hands_l2:
                all_hh_l2 = self.accelerator.gather_for_metrics(torch.tensor(human_hands_l2, device=device))
                all_hh_lsig = self.accelerator.gather_for_metrics(torch.tensor(human_hands_lsig, device=device))

        # Only log and print results on the main process
        if self.accelerator.is_main_process:
            mean_val_loss = all_losses.mean().item()

            if action_l2:  # Robot-only mode

                ac_l2 = all_action_l2.mean().item()
                ac_lsig = all_action_lsig.mean().item()

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
                    robot_l2_mean = all_robot_l2.mean().item()
                    robot_lsig_mean = all_robot_lsig.mean().item()
                    print(f"Step: {global_step}\tRobot L2={robot_l2_mean:.2f}\tRobot LSig={robot_lsig_mean:.2f}")
                    log_dict.update({
                        "eval/robot_l2": robot_l2_mean,
                        "eval/robot_lsig": robot_lsig_mean,
                    })

                if human_robot_l2:
                    hr_l2_mean = all_hr_l2.mean().item()
                    hr_lsig_mean = all_hr_lsig.mean().item()
                    print(f"Step: {global_step}\tHuman-Robot L2={hr_l2_mean:.2f}\tHuman-Robot LSig={hr_lsig_mean:.2f}")
                    log_dict.update({
                        "eval/human_robot_l2": hr_l2_mean,
                        "eval/human_robot_lsig": hr_lsig_mean,
                    })

                if human_hands_l2:
                    hh_l2_mean = all_hh_l2.mean().item()
                    hh_lsig_mean = all_hh_lsig.mean().item()
                    print(f"Step: {global_step}\tHuman-Hands L2={hh_l2_mean:.2f}\tHuman-Hands LSig={hh_lsig_mean:.2f}")
                    log_dict.update({
                        "eval/human_hands_l2": hh_l2_mean,
                        "eval/human_hands_lsig": hh_lsig_mean,
                    })

                if wandb.run is not None:
                    wandb.log(log_dict, step=global_step)

    def __getattr__(self, name):
        """Proxy other attributes to base_task"""
        return getattr(self.base_task, name)