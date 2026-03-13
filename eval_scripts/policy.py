"""
Policy wrapper for real-time robot inference.

Loads a trained diffusion model checkpoint, handles observation preprocessing
(image resize + ImageNet normalization, state normalization), runs chunked
inference with temporal ensemble smoothing, and optionally converts relative
actions to absolute actions. Supports both single-step and batch output modes.
"""
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
import hydra

from simhum.transforms import DualArmNormalizer, RelativeTransformer
from utils import LogTool, Tools, IMGSIZE


class Policy:
    def __init__(self, agent_path, model_name, args):
        LogTool.section("POLICY INITIALIZATION")

        self.args = args
        self.device = torch.device(args.device)
        LogTool.info("INIT", f"Device: {self.device}")
        LogTool.info("INIT", f"Checkpoint: {agent_path}/{model_name}")

        self._load_configs(agent_path)
        self._load_model(agent_path, model_name)
        self.reset()

    def _load_configs(self, agent_path):
        """Load config files, normalizer, and observation settings."""
        LogTool.info("CONFIG", "Loading config files...", 1)

        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            self._agent_config = yaml.safe_load(f)
        with open(Path(agent_path, "exp_config.yaml"), "r") as f:
            exp_config = yaml.safe_load(f)
        with open(Path("../experiments/obs_config.yaml"), "r") as f:
            obs_config = yaml.safe_load(f)

        # Action/state normalizer: maps raw values to [-1, 1] range used by the model
        self.normalizer = DualArmNormalizer.from_json(
            str(Path(agent_path, "robot_action_norm.json")),
            str(Path(agent_path, "robot_state_norm.json")),
        )
        LogTool.info("CONFIG", f"Normalizer: {self.normalizer}", 1)

        # Relative action mode: if enabled, model outputs are relative to current state
        # and must be converted back to absolute coordinates before execution
        self.use_relative_action = exp_config["params"]["task"]["train_buffer"]["use_relative_action"]
        if self.use_relative_action:
            self.relative_transformer = RelativeTransformer()
        LogTool.info("CONFIG", f"Relative action mode: {'enabled' if self.use_relative_action else 'disabled'}", 1)

        # Observation settings
        self.img_keys = obs_config["imgs"]
        self.pred_horizon = self.args.pred_horizon
        # ImageNet normalization, consistent with training-time preprocessing
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        LogTool.info("CONFIG", f"Camera count: {len(self.img_keys)} ({', '.join(self.img_keys)})", 1)
        LogTool.info("CONFIG", f"Prediction horizon: {self.pred_horizon}", 1)
        LogTool.success("CONFIG", "Config files loaded successfully", 1)

    def _load_model(self, agent_path, model_name):
        """Instantiate model from config, load weights, and compile for inference."""
        LogTool.info("MODEL", "Initializing model...", 1)
        agent = hydra.utils.instantiate(self._agent_config)
        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu")
        agent.load_state_dict(save_dict["model"])
        agent.eval().to(self.device)

        # Note: torch.compile is not used here because the DDIM scheduler
        # contains dynamic control flow that causes repeated graph breaks,
        # making compilation slower than eager mode for diffusion models.
        self.agent = agent.get_actions
        LogTool.success("MODEL", f"Model loaded (step: {save_dict.get('global_step', 'unknown')})", 1)

        del self._agent_config

    def reset(self):
        """Reset all inference state (action history, smoothing, timing)."""
        old_len = len(self.act_history) if hasattr(self, 'act_history') else 0
        self.act_history = deque(maxlen=self.pred_horizon)
        self.last_ac = None
        self._last_time = None
        LogTool.success("RESET", f"Policy state reset (history: {old_len} -> 0)", 1)

    # ---- Preprocessing ----

    def _proc_images(self, img_dict, size=IMGSIZE):
        """Resize, convert BGR->RGB, normalize, and batch images for the model."""
        torch_imgs = {}
        for i, k in enumerate(self.img_keys):
            if k not in img_dict:
                raise KeyError(f"missing image key: {k}")
            bgr_img = Tools.resize_like_dataset(img_dict[k][:, :, :3], size=size)
            rgb_img = bgr_img[:, :, ::-1].copy()
            rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
            rgb_img = self.transform(rgb_img)[None].to(self.device)
            torch_imgs[f"cam{i}"] = rgb_img[None]
        return torch_imgs

    def _proc_state(self, eef_state):
        """Normalize 16D EEF state using the training-time normalizer."""
        normalized_state = self.normalizer.normalize_state(eef_state)
        return torch.from_numpy(normalized_state).float()[None].to(self.device)

    # ---- Inference ----

    def _infer_policy(self, obs):
        """Run a single forward pass through the model and return raw predictions."""
        img = self._proc_images(obs["images"])
        state = self._proc_state(obs["state"])

        with torch.no_grad():
            pred_action = self.agent(img, {"state": state})
            if torch.is_tensor(pred_action):
                pred_action = pred_action.cpu().numpy()

            # Handle variable output shapes from different model architectures
            if pred_action.ndim > 2:
                ac = pred_action[0, :self.args.pred_horizon]
            elif pred_action.ndim > 1:
                ac = pred_action[0]
            else:
                ac = pred_action
            ac = ac.astype(np.float32)

        assert len(ac) >= self.args.pred_horizon, (
            f"Model returned {len(ac)} predictions, fewer than required {self.args.pred_horizon}!"
        )
        return ac

    def _forward_chunked(self, obs):
        """Chunked inference with post-processing pipeline.

        When action history is empty, runs model inference and fills the queue:
          1. Model outputs normalized action sequence (pred_horizon, D)
          2. Unnormalize to get raw actions
          3. Convert relative -> absolute actions (if enabled)
          4. Store in act_history queue

        Then pops one action and applies temporal ensemble (EMA) smoothing:
          smoothed = gamma * new + (1 - gamma) * previous
        This reduces jitter in real robot execution.
        """
        if not len(self.act_history):
            LogTool.info("INFER", "Action history empty, starting new inference...")
            acs_normalized = self._infer_policy(obs)
            acs_denorm = self.normalizer.unnormalize_action(acs_normalized)

            if self.use_relative_action:
                acs_abs = self.relative_transformer.forward(
                    acs_denorm, obs["state"], backward=True,
                )
                for ac in acs_abs:
                    self.act_history.append(ac)
            else:
                for ac in acs_denorm:
                    self.act_history.append(ac)

        # Temporal ensemble (EMA): smooths consecutive actions to reduce jitter
        raw_ac = self.act_history.popleft()
        last_ac = self.last_ac if self.last_ac is not None else raw_ac
        self.last_ac = self.args.gamma * raw_ac + (1 - self.args.gamma) * last_ac
        LogTool.info("INFER", f"Executing action (remaining: {len(self.act_history)})")
        return self.last_ac.copy()

    def forward(self, obs):
        """Return a single smoothed action, with optional rate limiting.

        Used by the /control endpoint in default (non-batch) mode.
        Respects the configured Hz rate to avoid overwhelming the robot.
        """
        ac = self._forward_chunked(obs)
        if self._last_time is not None:
            delta = time.time() - self._last_time
            if delta < self.args.period:
                time.sleep(self.args.period - delta)
        self._last_time = time.time()
        return ac

    def forward_batch(self, obs):
        """Return all pred_horizon actions at once with EMA smoothing applied.

        Used by the /control endpoint when batch=true. Runs a single model
        inference, then drains the entire action queue while applying temporal
        ensemble smoothing to each step. Useful when the client wants to
        receive the full action sequence in one request.
        """
        # Clear history to force a fresh inference
        self.act_history.clear()
        self.last_ac = None

        # _forward_chunked triggers inference and pops the first action
        self._forward_chunked(obs)
        actions = [self.last_ac.copy()]

        # Drain remaining actions with EMA smoothing
        while self.act_history:
            raw_ac = self.act_history.popleft()
            self.last_ac = self.args.gamma * raw_ac + (1 - self.args.gamma) * self.last_ac
            actions.append(self.last_ac.copy())

        LogTool.info("INFER", f"Batch inference: returned {len(actions)} actions")
        return actions
