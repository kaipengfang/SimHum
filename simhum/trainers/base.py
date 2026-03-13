# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from pathlib import Path
import os

import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

TRAIN_LOG_FREQ, EVAL_LOG_FREQ = 100, 1


class RunningMean:
    def __init__(self, max_len=TRAIN_LOG_FREQ):
        self._values = []
        self._ctr, self._max_len = 0, max_len

    def append(self, item):
        self._ctr = (self._ctr + 1) % self._max_len
        if len(self._values) < self._max_len:
            self._values.append(item)
        else:
            self._values[self._ctr] = item

    @property
    def mean(self):
        if len(self._values) == 0:
            raise ValueError
        return np.mean(self._values)


class BaseTrainer(ABC):
    def __init__(self, model, device_id, optim_builder, schedule_builder=None):
        self.model, self.device_id = model, device_id
        self.set_device(device_id)
        self.optim = optim_builder(self.model.parameters())
        self.schedule = (
            None if schedule_builder is None else schedule_builder(self.optim)
        )
        self._trackers = dict()
        self._is_train = True
        self.set_train()

    @abstractmethod
    def training_step(self, batch_input, global_step):
        pass

    @property
    def lr(self):
        if self.schedule is None:
            return self.optim.param_groups[0]["lr"]
        return self.schedule.get_last_lr()[0]

    def step_schedule(self):
        if self.schedule is None:
            return
        self.schedule.step()

    def save_checkpoint(self, save_path, global_step):
        model = self.model
        model_weights = (
            model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        )
        schedule_state = dict() if self.schedule is None else self.schedule.state_dict()
        save_dict = dict(
            model=model_weights,
            optim=self.optim.state_dict(),
            schedule=schedule_state,
            global_step=global_step,
        )
        torch.save(save_dict, save_path)

    def _get_agent_display_name(self, agent):
        """Get a user-friendly display name for the agent"""
        name_map = {
            'DiffusionTransformerAgent_Hybrid': 'Hybrid Diffusion Transformer',
            'DiffusionTransformerAgent_Dual': 'Dual Diffusion Transformer',
            'DiffusionTransformerAgent': 'Diffusion Transformer'
        }
        return name_map.get(agent.__class__.__name__, agent.__class__.__name__)

    def _truncate_path(self, path, levels=2):
        """Truncate path to show only the last N levels"""
        parts = Path(path).parts
        if len(parts) > levels + 1:
            return f".../{'/'.join(parts[-(levels+1):])}"
        return str(path)

    def _get_model_stats(self, model):
        """Get model parameter statistics in millions"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params / 1e6,
            'trainable': trainable_params / 1e6,
            'frozen': frozen_params / 1e6,
            'trainable_pct': trainable_params / total_params * 100 if total_params > 0 else 0
        }

    def _get_file_size_mb(self, path):
        """Get file size in MB"""
        try:
            size_bytes = os.path.getsize(path)
            return size_bytes / (1024 * 1024)
        except:
            return 0

    def _print_loading_summary(self, agent, load_path, robot_config, loading_results, model_stats_before, model_stats_after):
        """Print a beautiful, structured loading summary"""
        agent_name = self._get_agent_display_name(agent)
        truncated_path = self._truncate_path(load_path)
        file_size = self._get_file_size_mb(load_path)

        # Determine mode
        mode = "Robot-Only Fine-tuning" if (robot_config and robot_config.get('enabled')) else "Standard Fine-tuning"

        print("\n" + "═" * 80)
        print("                       CHECKPOINT LOADING SUMMARY")
        print("═" * 80)
        print(f"║ Agent Type    │ {agent_name}")
        print(f"║ Mode          │ {mode}")
        print(f"║ Checkpoint    │ {truncated_path}")
        if 'dimension_info' in loading_results:
            dim_info = loading_results['dimension_info']
            print(f"║ Dimensions    │ Checkpoint: {dim_info['checkpoint']}D → Current: {dim_info['current']}D {dim_info['status']}")

        print("═" * 80)
        print("MODEL STATISTICS")
        print(f"• Total Parameters        │ {model_stats_after['total']:.1f}M")
        print(f"• Trainable Parameters    │ {model_stats_after['trainable']:.1f}M ({model_stats_after['trainable_pct']:.1f}%)")
        if model_stats_after['frozen'] > 0:
            print(f"• Frozen Parameters       │ {model_stats_after['frozen']:.1f}M ({100 - model_stats_after['trainable_pct']:.1f}%)")

        # Show skipped components
        if loading_results['skipped_keys']:
            print("═" * 80)
            print(f"SKIPPED COMPONENTS ({len(loading_results['skipped_keys'])})")
            for key_info in loading_results['skipped_keys']:
                print(f"• {key_info}")

        # Show frozen components
        if loading_results['frozen_components']:
            print("═" * 80)
            print(f"FROZEN COMPONENTS ({len(loading_results['frozen_components'])})")
            for component_info in loading_results['frozen_components']:
                print(f"• {component_info}")

        print("═" * 80)
        print("LOADING RESULTS")
        total_loaded = loading_results['loaded_count']
        total_available = loading_results['total_count']
        skipped_count = len(loading_results['skipped_keys'])
        missing_count = loading_results.get('missing_count', 0)

        print(f"• Successfully loaded: {total_loaded}/{total_available} parameters ({total_loaded/total_available*100:.1f}%)")
        if skipped_count > 0:
            print(f"• Skipped (robot-only): {skipped_count} parameters ({skipped_count/total_available*100:.1f}%)")
        if missing_count > 0:
            missing_keys = loading_results.get('missing_keys')
            print(f"• Missing (random init): {missing_count} parameters ({missing_count/total_available*100:.1f}%)")
            for i in missing_keys:
                print(i)
        print(f"• Checkpoint file size: {file_size:.1f}MB")
        print("═" * 80 + "\n")

    def load_checkpoint(self, load_path, strict=True, load_optimizer=True, robot_config=None):
        """Enhanced checkpoint loading with robot-only mode support and beautiful reporting"""

        # Load checkpoint
        load_dict = torch.load(load_path, map_location="cpu")
        model = self.model
        model = model.module if isinstance(model, DDP) else model

        # Get model stats before loading
        model_stats_before = self._get_model_stats(model)

        # Initialize loading results
        loading_results = {
            'skipped_keys': [],
            'frozen_components': [],
            'loaded_count': 0,
            'total_count': 0,
            'missing_count': 0,
        }
        
        if not strict:
            # Flexible loading for fine-tuning with different architectures
            model_state = model.state_dict()
            pretrained_state = load_dict["model"]
            loading_results['total_count'] = len(pretrained_state)

            # Handle robot-only mode
            skip_keys = []
            if robot_config and robot_config.get('enabled'):
                skip_keys = self._handle_robot_only_mode(model, robot_config, load_path, loading_results)

            # Filter out incompatible keys and skip_keys
            compatible_state = {}
            incompatible_keys = []

            for k, v in pretrained_state.items():
                if k in skip_keys:
                    # Skip key info already added in _handle_robot_only_mode
                    continue

                if k in model_state:
                    if model_state[k].shape == v.shape:
                        compatible_state[k] = v
                    else:
                        incompatible_keys.append(f"{k}: shape mismatch {v.shape} vs {model_state[k].shape}")
                else:
                    incompatible_keys.append(f"{k}: not in current model")

            # Count missing keys in current model
            missing_keys = []
            for k in model_state.keys():
                if k not in compatible_state and k not in skip_keys:
                    missing_keys.append(k)

            loading_results['loaded_count'] = len(compatible_state)
            loading_results['missing_count'] = len(missing_keys)
            loading_results['missing_keys'] = missing_keys
            # Load the compatible state
            model_state.update(compatible_state)
            model.load_state_dict(model_state)

        else:
            # Strict loading
            model.load_state_dict(load_dict["model"])
            loading_results['total_count'] = len(load_dict["model"])
            loading_results['loaded_count'] = len(load_dict["model"])

        # Load optimizer and scheduler if requested
        if load_optimizer:
            try:
                if "optim" in load_dict:
                    self.optim.load_state_dict(load_dict["optim"])
                if self.schedule is not None and "schedule" in load_dict:
                    self.schedule.load_state_dict(load_dict["schedule"])
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")

        # Get model stats after loading and potential freezing
        model_stats_after = self._get_model_stats(model)

        # Print beautiful summary
        self._print_loading_summary(model, load_path, robot_config, loading_results,
                                   model_stats_before, model_stats_after)

        return load_dict.get("global_step", 0)

    def _handle_robot_only_mode(self, model, robot_config, load_path, loading_results):
        """Handle robot-only mode: detect dimensions, skip keys, and freeze components"""
        skip_keys = []

        # Detect dimension mismatch and prepare skip keys
        if hasattr(model, 'should_skip_keys_for_robot_only'):
            need_skip, skip_keys = model.should_skip_keys_for_robot_only(load_path)

            # Add dimension info to loading results
            if hasattr(model, 'detect_checkpoint_action_dim'):
                detected_dim = model.detect_checkpoint_action_dim(load_path)
                current_dim = getattr(model, '_ac_dim', 'unknown')

                if detected_dim is not None:
                    status = "(mismatch detected)" if detected_dim != current_dim else "(match)"
                    loading_results['dimension_info'] = {
                        'checkpoint': detected_dim,
                        'current': current_dim,
                        'status': status
                    }

            # Add skip key details
            if need_skip:
                for key in skip_keys:
                    loading_results['skipped_keys'].append(f"{key} (dimension mismatch)")

        # Handle freezing operations
        agent_class = model.__class__.__name__
        frozen_encoder = robot_config.get('frozen_encoder', True)
        frozen_sim_adaptor = robot_config.get('frozen_sim_adaptor', True)
        frozen_human_adaptor = robot_config.get('frozen_human_adaptor', True)
        frozen_diffusion = robot_config.get('frozen_diffusion', True)
        if frozen_encoder and hasattr(model, 'visual_features'):
            # Freeze encoder
            frozen_params = 0
            for param in model.visual_features.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            loading_results['frozen_components'].append(f"Visual Encoder: {frozen_params/1e6:.1f}M parameters")

        # Agent-specific freezing
        if agent_class == 'DiffusionTransformerAgent_Dual':
            # Freeze unused adaptor
            if frozen_sim_adaptor and hasattr(model, 'sim_vision_adaptor'):
                frozen_params = 0
                for param in model.sim_vision_adaptor.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                loading_results['frozen_components'].append(f"Sim Vision Adaptor: {frozen_params/1e6:.1f}M parameters")
            if frozen_human_adaptor and hasattr(model, 'human_vision_adaptor'):
                frozen_params = 0
                for param in model.human_vision_adaptor.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                loading_results['frozen_components'].append(f"Human Vision Adaptor: {frozen_params/1e6:.1f}M parameters")
            if frozen_diffusion:
                frozen_params = 0
                for param in model.noise_net.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                loading_results['frozen_components'].append(f"Diffusion Backbone: {frozen_params/1e6:.1f}M parameters")

        elif agent_class == 'DiffusionTransformerAgent_Hybrid':
            # Set obs_processor to robot_only mode
            if hasattr(model, 'obs_processor') and hasattr(model.obs_processor, 'set_robot_only_mode'):
                model.obs_processor.set_robot_only_mode(True)

        return skip_keys

    def _load_callback(self, load_path, load_dict):
        pass

    @property
    def is_train(self):
        return self._is_train

    def set_train(self):
        self._is_train = True
        self.model = self.model.train()

    def set_eval(self):
        self._is_train = False
        self.model = self.model.eval()

        # reset running mean for eval trackers
        for k in self._trackers:
            if "eval/" in k:
                self._trackers[k] = RunningMean()

    def log(self, key, global_step, value):
        log_freq = TRAIN_LOG_FREQ if self._is_train else EVAL_LOG_FREQ
        key_prepend = "train/" if self._is_train else "eval/"
        key = key_prepend + key

        if key not in self._trackers:
            self._trackers[key] = RunningMean()

        tracker = self._trackers[key]
        tracker.append(value)

        if global_step % log_freq == 0 and wandb.run is not None:
            wandb.log({key: tracker.mean}, step=global_step)

    def set_device(self, device_id):
        self.model = self.model.to(device_id)
