# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import traceback
import shutil

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

import hydra
from simhum import misc, transforms
from simhum.trainers.bc import BehaviorCloning
base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path=os.path.join(base_path, "experiments"), config_name="finetune.yaml")
def bc_finetune(cfg: DictConfig):
    try:
        resume_model = misc.init_job(cfg)

        # set random seeds for reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed + 1)

        # build agent from hydra configs
        with open("agent_config.yaml", "w") as f:
            agent_yaml = OmegaConf.to_yaml(cfg.agent, resolve=True)
            f.write(agent_yaml)

        agent = hydra.utils.instantiate(cfg.agent)
        
        # Handle robot-only mode configuration and backward compatibility
        def _handle_robot_only_config(cfg):
            """Handle robot-only mode configuration with backward compatibility"""
            robot_only_mode = cfg.get('robot_only_mode', False)
            robot_only_finetune = cfg.get('robot_only_finetune', None)

            # Backward compatibility check
            if robot_only_finetune is not None:
                import warnings
                warnings.warn("robot_only_finetune is deprecated, use robot_only_mode instead",
                            DeprecationWarning, stacklevel=2)
                if not robot_only_mode:
                    robot_only_mode = robot_only_finetune
                    print("Auto-migrating: robot_only_finetune -> robot_only_mode")

            return robot_only_mode

        # Set robot-only mode for supported models (only for finetune mode)
        if cfg.mode == 'finetune':
            robot_only_mode = _handle_robot_only_config(cfg)

            # Robot-only mode configuration will be handled during checkpoint loading
        
        trainer: BehaviorCloning = hydra.utils.instantiate(cfg.trainer, model=agent, device_id=0)

        # build task, replay buffer, and dataloader
        task = hydra.utils.instantiate(
            cfg.task, batch_size=cfg.batch_size, num_workers=cfg.num_workers, human_ratio=cfg.human_ratio
        )

        # Copy normalization parameters to model output directory
        # This ensures inference can find the normalization files with the model checkpoint
        buffer_dir = os.path.dirname(cfg.buffer_path)

        # Check which action normalization file exists based on use_relative_action setting
        use_relative = cfg.task.train_buffer.get('use_relative_action', False)

        # Human normalization files
        if use_relative:
            human_ac_norm_src = os.path.join(buffer_dir, "human_relative_ac_norm.json")
        else:
            human_ac_norm_src = os.path.join(buffer_dir, "human_absolute_ac_norm.json")
        human_state_norm_src = os.path.join(buffer_dir, "human_absolute_state_norm.json")

        # Copy human normalization files to unified names
        if os.path.exists(human_ac_norm_src):
            shutil.copyfile(human_ac_norm_src, "human_action_norm.json")
            print(f"✓ Copied human action normalization: {human_ac_norm_src} -> human_action_norm.json")
        else:
            print(f"⚠ Warning: Human action normalization file not found: {human_ac_norm_src}")

        if os.path.exists(human_state_norm_src):
            shutil.copyfile(human_state_norm_src, "human_state_norm.json")
            print(f"✓ Copied human state normalization: {human_state_norm_src} -> human_state_norm.json")
        else:
            print(f"⚠ Warning: Human state normalization file not found: {human_state_norm_src}")

        # Robot normalization files
        if use_relative:
            robot_ac_norm_src = os.path.join(buffer_dir, "robot_relative_ac_norm.json")
        else:
            robot_ac_norm_src = os.path.join(buffer_dir, "robot_absolute_ac_norm.json")
        robot_state_norm_src = os.path.join(buffer_dir, "robot_absolute_state_norm.json")

        # Copy robot normalization files to unified names
        if os.path.exists(robot_ac_norm_src):
            shutil.copyfile(robot_ac_norm_src, "robot_action_norm.json")
            print(f"✓ Copied robot action normalization: {robot_ac_norm_src} -> robot_action_norm.json")
        else:
            print(f"⚠ Warning: Robot action normalization file not found: {robot_ac_norm_src}")

        if os.path.exists(robot_state_norm_src):
            shutil.copyfile(robot_state_norm_src, "robot_state_norm.json")
            print(f"✓ Copied robot state normalization: {robot_state_norm_src} -> robot_state_norm.json")
        else:
            print(f"⚠ Warning: Robot state normalization file not found: {robot_state_norm_src}")

        # create a gpu train transform (if used)
        gpu_transform = (
            transforms.get_gpu_transform_by_name(cfg.train_transform)
            if "gpu" in cfg.train_transform
            else None
        )

        # Handle different training modes
        if resume_model is not None:
            assert os.path.exists(resume_model), f"Checkpoint {resume_model} does not exist!"
            
            if cfg.mode == 'resume':
                print(f"Resume mode: Loading full checkpoint from {resume_model}")
                misc.GLOBAL_STEP = trainer.load_checkpoint(resume_model, load_optimizer=True)
                print(f"Resuming training from step {misc.GLOBAL_STEP}")
                
            elif cfg.mode == 'finetune':
                # Prepare robot-only mode configuration for checkpoint loading
                robot_config = None
                if cfg.get('robot_only_mode', False):
                    robot_config = {
                        'enabled': True,
                        'use_human_adaptor': cfg['use_human_adaptor'],
                        'frozen_encoder': cfg['robot_only_frozen_encoder'],
                        'frozen_sim_adaptor': cfg['robot_only_frozen_sim_adaptor'],
                        'frozen_human_adaptor': cfg['robot_only_frozen_human_adaptor'],
                        'frozen_diffusion': cfg['robot_only_frozen_dit'],
                    }

                trainer.load_checkpoint(
                    resume_model,
                    strict=False,
                    load_optimizer=False,
                    robot_config=robot_config
                )
                # Always start from step 0 in finetune mode
                misc.GLOBAL_STEP = 0
                print("Starting fine-tuning from step 0")
                
            else:
                raise ValueError(f"Unknown mode: {cfg.mode}. Expected 'resume' or 'finetune'")
        else:
            # No checkpoint specified, start from scratch
            misc.GLOBAL_STEP = 0
            print("No checkpoint specified, starting training from scratch")
        
        # Save initial checkpoint if starting from step 0
        if misc.GLOBAL_STEP == 0:
            initial_checkpoint = f"{cfg.exp_name}_step_{misc.GLOBAL_STEP}.ckpt"
            trainer.save_checkpoint(initial_checkpoint, misc.GLOBAL_STEP)
        
        assert misc.GLOBAL_STEP >= 0, "GLOBAL_STEP not loaded correctly!"

        # register checkpoint handler and enter train loop
        checkpoint_base = f"{cfg.exp_name}_step_{misc.GLOBAL_STEP}.ckpt"
        misc.set_checkpoint_handler(trainer, checkpoint_base)
        print(f"Starting at Global Step {misc.GLOBAL_STEP}")

        trainer.set_train()
        train_iterator = iter(task.train_loader)
        for itr in (
            pbar := tqdm.tqdm(range(cfg.max_iterations), postfix=dict(Loss=None))
        ):
            if itr < misc.GLOBAL_STEP:
                continue

            # infinitely sample batches until the train loop is finished
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(task.train_loader)
                batch = next(train_iterator)

            # handle the image transform on GPU if specified
            if gpu_transform is not None:
                (imgs, obs), actions, mask = batch
                imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                imgs = {k: gpu_transform(v) for k, v in imgs.items()}
                batch = ((imgs, obs), actions, mask)

            trainer.optim.zero_grad()
            loss = trainer.training_step(batch, misc.GLOBAL_STEP)
            loss.backward()
            trainer.optim.step()

            pbar.set_postfix(dict(Loss=loss.item()))
            misc.GLOBAL_STEP += 1

            if misc.GLOBAL_STEP % cfg.schedule_freq == 0:
                trainer.step_schedule()

            if misc.GLOBAL_STEP % cfg.eval_freq == 0:
                trainer.set_eval()
                task.eval(trainer, misc.GLOBAL_STEP)
                trainer.set_train()

            if misc.GLOBAL_STEP >= cfg.max_iterations:
                trainer.save_checkpoint(f"{cfg.exp_name}_final.ckpt", misc.GLOBAL_STEP)
                return
            elif misc.GLOBAL_STEP % cfg.save_freq == 0:
                trainer.save_checkpoint(f"{cfg.exp_name}_step_{misc.GLOBAL_STEP}.ckpt", misc.GLOBAL_STEP)

    # gracefully handle and log errors
    except Exception:
        traceback.print_exc(file=open("exception.log", "w"))
        with open("exception.log", "r") as f:
            print(f.read())


if __name__ == "__main__":
    bc_finetune()
