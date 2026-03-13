# Copyright (c) Sudeep Dasari, 2023
# Modernized Accelerate Behavior Cloning Trainer
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from simhum.trainers.base import BaseTrainer


class AccelerateBehaviorCloning(BaseTrainer):
    """
    Modern Accelerate behavior cloning trainer
    - Automatic AMP mixed precision support
    - Transparent DDP distributed training
    - Fully compatible with BaseTrainer interface
    - Integrated robot_only_mode and other modern features
    """

    def __init__(self, model, device_id, optim_builder, schedule_builder=None,
                 enable_amp=True, mixed_precision="fp16"):
        """
        Initialize Accelerate trainer

        Args:
            model: Model instance
            device_id: GPU device ID (compatible with existing interface)
            optim_builder: Optimizer builder
            schedule_builder: Scheduler builder
            enable_amp: Whether to enable AMP
            mixed_precision: Mixed precision type ("fp16", "bf16", "no")
        """
        # Automatically create Accelerator - need to handle DDP unused parameters
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision if enable_amp else "no",
            log_with=None,  # WandB is managed externally
            kwargs_handlers=[ddp_kwargs],
        )

        # Store original device_id for compatibility
        self.device_id = self.accelerator.device
        self._original_device_id = device_id

        # Initialize model and optimizer
        self.model = model
        self.optim = optim_builder(self.model.parameters())
        self.schedule = (
            None if schedule_builder is None else schedule_builder(self.optim)
        )

        # Use Accelerate to prepare model and optimizer
        # Note: do not prepare scheduler, it causes a periodic bug at 13600 steps
        self.model, self.optim = self.accelerator.prepare(self.model, self.optim)
        # if self.schedule is not None:
        #     self.schedule = self.accelerator.prepare(self.schedule)  # Commented out: fixes LR periodic jump bug

        # Initialize other BaseTrainer attributes (skip device setup)
        self._trackers = dict()
        self._is_train = True
        self.set_train()

    def training_step(self, batch, global_step):
        """
        Training step - uses Accelerate's autocast and automatic data transfer
        """
        (imgs, obs), actions, mask = batch

        # Accelerate handles device transfer automatically, no manual to(device) needed
        ac_flat = actions.reshape((actions.shape[0], -1))
        mask_flat = mask.reshape((mask.shape[0], -1))

        # Use Accelerate's autocast for mixed precision
        with self.accelerator.autocast():
            loss = self.model(imgs, obs, ac_flat, mask_flat)

        # Handle composite loss (supports diffusion_dual, etc.)
        if isinstance(loss, dict):
            if self.accelerator.is_main_process:
                self.log("robot_loss", global_step, loss['robot_loss'].item())
                self.log("human_robot_loss", global_step, loss['human_robot_loss'].item())
                self.log("human_hands_loss", global_step, loss['human_hands_loss'].item())
                self.log("bc_loss", global_step, loss['loss'].item())
            loss = loss['loss']
        else:
            if self.accelerator.is_main_process:
                self.log("bc_loss", global_step, loss.item())

        if self.is_train and self.accelerator.is_main_process:
            self.log("lr", global_step, self.lr)

        return loss

    def save_checkpoint(self, save_path, global_step):
        """
        Save checkpoint - compatible with original format, following Accelerate best practices
        """
        # Fix: sync all processes before saving (per documentation)
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            return

        # Get unwrapped model to maintain format compatibility
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Prepare save dict (compatible with BaseTrainer format)
        schedule_state = dict() if self.schedule is None else self.schedule.state_dict()
        save_dict = dict(
            model=unwrapped_model.state_dict(),
            optim=self.optim.state_dict(),
            schedule=schedule_state,
            global_step=global_step,
        )

        # Use Accelerate to save (handles distributed scenarios)
        self.accelerator.save(save_dict, save_path)

    def load_checkpoint(self, load_path, strict=True, load_optimizer=True, robot_config=None):
        """
        Load checkpoint - all processes load to ensure DDP consistency
        """
        # Get unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Fix: all processes load the checkpoint
        # Temporarily replace model with unwrapped version to reuse BaseTrainer logic
        original_wrapped_model = self.model
        self.model = unwrapped_model

        try:
            # All processes call BaseTrainer's load_checkpoint
            global_step = super().load_checkpoint(load_path, strict, load_optimizer, robot_config)
        finally:
            # Restore wrapped model
            self.model = original_wrapped_model

        # Sync all processes to ensure loading is complete
        self.accelerator.wait_for_everyone()

        return global_step

    def log(self, key, global_step, value):
        """
        Logging - only log on the main process
        """
        if self.accelerator.is_main_process:
            super().log(key, global_step, value)

    def set_device(self, device_id):
        """
        Device setup - Accelerate handles this automatically, kept for interface compatibility
        """
        # Accelerate handles devices automatically; empty implementation for BaseTrainer compatibility
        pass