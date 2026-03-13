# Copyright (c) Fang Kaipeng, 2025
# Extension of the Diffusion Transformer to support dual-path Human-Robot training
# Key difference from hybrid: No prompts, separate encoder/decoder for state/action processing
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Origin
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from simhum.agent import BaseAgent
from simhum.models.diffusion import (
    _get_activation_fn,
    _with_pos_embed,
    _PositionalEncoding,
    _TimeNetwork,
    _ShiftScaleMod,
    _ZeroScaleMod,
    _TransformerEncoder,
    _TransformerDecoder,
    _DiTDecoder,
    _DiTNoiseNet,
    DiffusionTransformerAgent
)


class VisionAdaptor(nn.Module):
    """
    Vision Adaptor for domain-specific feature transformation.
    Used to adapt visual features from different domains (human vs sim).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)  # No residual connection




class _DualFinalLayer(nn.Module):
    """
    Dual-path final output layer without prompts.
    Separate outputs for robot and human based on action dimensions.
    """
    def __init__(self, hidden_size, robot_action_dim=16, human_action_dim=44):
        super().__init__()
        self.robot_action_dim = robot_action_dim
        self.human_action_dim = human_action_dim
        
        # Shared normalization
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Separate output heads for robot and human
        self.robot_action_linear = nn.Linear(hidden_size, robot_action_dim, bias=True)
        self.human_action_linear = nn.Linear(hidden_size, human_action_dim, bias=True)
        
        # Shared AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def forward(self, x, t, cond, action_dims=None):
        """
        Forward pass - returns structured predictions for dual loss calculation
        
        Args:
            x: input features [seq_len, batch_size, hidden_size]
            t: time embedding [batch_size, hidden_size]
            cond: conditioning [seq_len, batch_size, hidden_size]
            action_dims: action dimension indicator for each sample [batch_size]
        
        Returns:
            Dictionary containing separate predictions for robot and human components
        """
        # Process conditioning
        cond = torch.mean(cond, axis=0)
        cond = cond + t
        
        # Apply AdaLN
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        shift = shift.unsqueeze(0)
        scale = scale.unsqueeze(0)
        
        # Normalize
        x = self.norm_final(x)
        x = x * (1 + scale) + shift
        
        # Transpose for batch processing
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        

        if action_dims is not None:
            robot_mask = (action_dims == self.robot_action_dim)
            human_mask = (action_dims == self.human_action_dim)
            robot_pred = torch.zeros(x.shape[0], x.shape[1], self.robot_action_dim, device=x.device)
            human_pred = torch.zeros(x.shape[0], x.shape[1], self.human_action_dim, device=x.device)

            if robot_mask.any():
                robot_pred[robot_mask] = self.robot_action_linear(x[robot_mask])
            if human_mask.any():
                human_pred[human_mask] = self.human_action_linear(x[human_mask])
        else:
            robot_pred = self.robot_action_linear(x)
            human_pred = self.human_action_linear(x)
              
        return {
            'robot': robot_pred,      # [batch_size, seq_len, 16] - for robot loss 
            'human': human_pred,      # [batch_size, seq_len, 44] - for human loss
            'action_dims': action_dims  # Pass through for loss calculation
        }


class _DiTNoiseNet_Dual(_DiTNoiseNet):
    """
    Dual-path DiT Noise Network without prompts.
    Separate state encoders and action projectors for Human and Robot data.
    Shared transformer backbone for efficient processing.
    """
    def __init__(
        self,
        ac_dim,  # Max action dimension (human dimension)
        ac_chunk,
        robot_state_dim=16,
        human_state_dim=44,
        robot_action_dim=16,
        human_action_dim=44,
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
    ):
        # Initialize base class with max action dim
        super().__init__(
            ac_dim=ac_dim,
            ac_chunk=ac_chunk,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            activation=activation,
        )
        
        # Store dimensions
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.robot_action_dim = robot_action_dim
        self.human_action_dim = human_action_dim
        self.hidden_dim = hidden_dim
        # Replace the original ac_proj with separate action projectors
        delattr(self, 'ac_proj')  # Remove the base class projector
        
        # Add separate state encoders for robot and human (following hybrid pattern)
        # self.robot_state_encoder = nn.Sequential(
        #     nn.Linear(robot_state_dim, robot_state_dim * 2),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(robot_state_dim * 2, hidden_dim),
        # )
        
        # self.human_state_encoder = nn.Sequential(
        #     nn.Linear(human_state_dim, human_state_dim),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(human_state_dim, hidden_dim),
        # )
        
        # Add separate action projectors for robot and human (following hybrid pattern)
        self.robot_ac_proj = nn.Sequential(
            nn.Linear(robot_action_dim, robot_action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(robot_action_dim, hidden_dim),
        )
        
        self.human_ac_proj = nn.Sequential(
            nn.Linear(human_action_dim, human_action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(human_action_dim, hidden_dim),
        )
        
        # Replace final layer with dual output (following hybrid pattern)
        delattr(self, 'eps_out')  # Remove the base class final layer
        self.eps_out = _DualFinalLayer(hidden_dim, robot_action_dim, human_action_dim)
        
    def forward_enc(self, obs_enc):
        """
        Encoder forward pass without prompts - direct state encoding routing
        Args:
            obs_enc: observation encoding [batch_size, seq_len, hidden_dim]  
            action_dims: tensor indicating action dimension for each sample [batch_size]
        """
        obs_enc = obs_enc.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        pos = self.enc_pos(obs_enc)
        #obs_enc[:3,:,:] = 0
        enc_cache = self.encoder(obs_enc, pos)
        
        return enc_cache
    
    def forward_dec(self, noise_actions, time, enc_cache, action_dims=None):
        """
        Decoder forward pass with separate action processing
        
        Args:
            noise_actions: noisy actions [batch_size, seq_len, action_dim]
            time: timesteps [batch_size]
            enc_cache: encoder output cache
            action_dims: action dimension indicator for each sample [batch_size]
        
        Returns:
            Dictionary with robot and human predictions for dual loss calculation
        """
        time_enc = self.time_net(time)
       
        # Process actions based on their dimensions
        batch_size = noise_actions.shape[0]
        
        # Check if in robot_only_mode
        if hasattr(self, '_parent_robot_only_mode') and self._parent_robot_only_mode:
            # Robot-only mode: all samples use robot projection
            robot_actions = noise_actions[:, :, :self.robot_action_dim]
            ac_tokens = self.robot_ac_proj(robot_actions)
        else:
            # Dual mode: route based on action_dims
            ac_tokens = torch.zeros(batch_size, self.dec_pos.shape[0], self.dec_pos.shape[-1],
                                   device=noise_actions.device)
            
            if action_dims is not None:
                robot_mask = (action_dims == self.robot_action_dim)
                human_mask = (action_dims == self.human_action_dim)
                
                if robot_mask.any():
                    robot_actions = noise_actions[robot_mask, :, :self.robot_action_dim]
                    robot_tokens = self.robot_ac_proj(robot_actions)
                    ac_tokens[robot_mask] = robot_tokens.to(ac_tokens.dtype)

                if human_mask.any():
                    human_actions = noise_actions[human_mask, :, :self.human_action_dim]
                    human_tokens = self.human_ac_proj(human_actions)
                    ac_tokens[human_mask] = human_tokens.to(ac_tokens.dtype)
            else:
                raise NotImplementedError
        
        ac_tokens = ac_tokens.transpose(0, 1)
        dec_in = ac_tokens + self.dec_pos
        
        # Apply shared decoder
        dec_out = self.decoder(dec_in, time_enc, enc_cache)
         
        # Apply dual final layer - returns structured predictions
        return self.eps_out(dec_out, time_enc, enc_cache[-1], action_dims)
    
    def forward(self, noise_actions, time, obs_enc, action_dims=None, enc_cache=None):
        """
        Forward pass with dual-path support
        
        Args:
            noise_actions: noisy actions [batch_size, seq_len, action_dim]
            time: timesteps [batch_size]
            obs_enc: observation encoding [batch_size, seq_len, hidden_dim]
            action_dims: action dimension indicator for each sample [batch_size]
            enc_cache: optional pre-computed encoder cache
        
        Returns:
            Dictionary with robot and human predictions for dual loss calculation
        """
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        
        return self.forward_dec(noise_actions, time, enc_cache, action_dims)


class DiffusionTransformerAgent_Dual(DiffusionTransformerAgent):
    """
    Dual-path Diffusion Transformer Agent supporting both Human and Robot data.
    Key differences from Hybrid:
    - No learnable prompts
    - Separate encoder/decoder paths only for state/action processing
    - Shared transformer backbone for efficiency
    - Vision adaptors for domain-specific feature transformation
    """
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override to handle backward compatibility with checkpoints without adaptors.
        """
        # Check if the checkpoint has adaptor weights
        has_human_adaptor = any('human_vision_adaptor' in k for k in state_dict.keys())
        has_sim_adaptor = any('sim_vision_adaptor' in k for k in state_dict.keys())
        
        if not has_human_adaptor or not has_sim_adaptor:
            print("Warning: Loading checkpoint without vision adaptors.")
            print("  Initializing adaptors with random weights.")
            # Use strict=False to allow missing keys
            strict = False
        
        # Load the state dict
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            adaptor_keys = [k for k in missing_keys if 'vision_adaptor' in k]
            other_keys = [k for k in missing_keys if 'vision_adaptor' not in k]
            
            if adaptor_keys:
                print(f"  Missing adaptor keys (will be randomly initialized): {len(adaptor_keys)} keys")
            if other_keys:
                print(f"  Other missing keys: {other_keys}")
        
        return missing_keys, unexpected_keys
    def __init__(
        self,
        features,
        odim,
        n_cams,
        use_obs,
        ac_dim,  # Max action dimension (human dimension)
        ac_chunk,
        train_diffusion_steps,
        eval_diffusion_steps,
        robot_state_dim=16,
        human_state_dim=44,
        robot_action_dim=16,
        human_action_dim=44,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        early_fusion=False,
        feat_norm=None,
        token_dim=None,
        noise_net_kwargs=dict(),
        robot_only_mode=False,
        use_human_adaptor=True,
        dropout_wrist_rate=0.0
    ):
        # Initialize base agent (skip parent __init__ to use our noise net)
        BaseAgent.__init__(
            self,
            odim=odim,
            features=features,
            n_cams=n_cams,
            imgs_per_cam=imgs_per_cam,
            use_obs=use_obs,
            share_cam_features=share_cam_features,
            early_fusion=early_fusion,
            dropout=dropout,
            feat_norm=feat_norm,
            token_dim=token_dim,
        )
        # Store dimensions and mode configuration
        self.robot_only_mode = robot_only_mode
        self.dropout_wrist_rate = dropout_wrist_rate
        # Store n_cams for embed method
        self._n_cams = n_cams
   
        # if not share_cam_features:
        #     sim_vision_adaptor_list = [sim_vision_adaptor] + [copy.deepcopy(sim_vision_adaptor) for _ in range(1, n_cams)]
        #     self.sim_vision_adaptor = nn.ModuleList(sim_vision_adaptor_list)
        # else :
        #     self.sim_vision_adaptor = sim_vision_adaptor
        # Use dual noise network
        self.noise_net = _DiTNoiseNet_Dual(
            ac_dim=human_action_dim,  # Use human_action_dim as max dimension
            ac_chunk=ac_chunk,
            robot_state_dim=robot_state_dim,
            human_state_dim=human_state_dim,
            robot_action_dim=robot_action_dim,
            human_action_dim=human_action_dim,
            **noise_net_kwargs,
        )
        
        self.robot_state_encoder = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(robot_state_dim, self.noise_net.hidden_dim))
        self.human_state_encoder = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(human_state_dim, self.noise_net.hidden_dim))
        # Initialize vision adaptors for domain-specific feature transformation
        # These are used to adapt cam0 features for different data domains
        self.human_vision_adaptor = VisionAdaptor(features.embed_dim, self.noise_net.hidden_dim)
        sim_vision_adaptor = VisionAdaptor(features.embed_dim, self.noise_net.hidden_dim)
        if not share_cam_features:
            sim_vision_adaptor_list = [sim_vision_adaptor] + [copy.deepcopy(sim_vision_adaptor) for _ in range(1, n_cams)]
            self.sim_vision_adaptor = nn.ModuleList(sim_vision_adaptor_list)
        else :
             self.sim_vision_adaptor = sim_vision_adaptor
        self.use_human_adaptor = use_human_adaptor
        # In robot_only_mode, use robot_action_dim as max; otherwise use human_action_dim
        self._ac_dim = robot_action_dim if robot_only_mode else human_action_dim
        self._ac_chunk = ac_chunk
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.robot_action_dim = robot_action_dim
        self.human_action_dim = human_action_dim
        
        # Pass robot_only_mode flag to noise_net
        self.noise_net._parent_robot_only_mode = robot_only_mode
        
        # Initialize diffusion scheduler (same as base)
        assert eval_diffusion_steps <= train_diffusion_steps, "Can't eval with more steps!"
        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
  
        # Print detailed parameter statistics
        self._print_model_statistics()

    def _print_model_statistics(self):
        """
        Print clean and clear model parameter statistics with color.
        """
        # ANSI color codes - simple and clean
        CYAN = '\033[96m'
        YELLOW = '\033[93m'
        GREEN = '\033[92m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

        # Calculate parameters for each component
        visual_params = sum(p.numel() for p in self.visual_features.parameters())
        human_adaptor_params = sum(p.numel() for p in self.human_vision_adaptor.parameters())
        sim_adaptor_params = sum(p.numel() for p in self.sim_vision_adaptor.parameters())
        noise_net_params = sum(p.numel() for p in self.noise_net.parameters())
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Helper functions
        def format_size(n):
            return f"{n / 1e6:.2f}M"

        def format_pct(part, total):
            return f"{100 * part / total:.2f}%"

        # Print statistics table
        print("\n" + CYAN + "=" * 80)
        print(BOLD + "MODEL PARAMETER STATISTICS" + RESET)
        print(CYAN + "=" * 80 + RESET)
        print(f"{'Component':<30} {'Size':>15} {'Percentage':>12}")
        print("-" * 80)

        components = [
            ("Vision Encoder", visual_params),
            ("Human Vision Adaptor", human_adaptor_params),
            ("Sim Vision Adaptor", sim_adaptor_params),
            ("DiT Noise Network", noise_net_params),
        ]

        for name, params in components:
            print(f"{name:<30} {YELLOW}{format_size(params):>15}{RESET} {format_pct(params, total_params):>12}")

        print("-" * 80)
        print(f"{BOLD}{'TOTAL':<30}{RESET} {GREEN}{BOLD}{format_size(total_params):>15}{RESET} {BOLD}{'100.00%':>12}{RESET}")
        print(CYAN + "=" * 80 + RESET + "\n")

    def _embed_helper(self, net, im):
        """Helper function to process images through visual feature extractor"""
        if self.early_fusion and len(im.shape) == 5:
            T = im.shape[1]
            im = torch.cat([im[:, t] for t in range(T)], 1)
            return net(im)
        elif len(im.shape) == 5:
            B, T, C, H, W = im.shape
            embeds = net(im.reshape((B * T, C, H, W)))
            embeds = embeds.reshape((B, -1, net.embed_dim))
            return embeds

        assert len(im.shape) == 4
        return net(im)

    def _process_adaptor_output(self, output):
        """Ensure adaptor output is 2D [batch_size, embed_dim]"""
        if len(output.shape) == 3:
            return output.squeeze(1)
        return output

    def embed(self, imgs, action_dims=None):
        """
        Override BaseAgent's embed method with dual-path processing.

        Args:
            imgs: dictionary of camera images, keys: "cam0", "cam1", ..., "cam{n_cams-1}"
            action_dims: tensor [batch_size], indicating action dimension for each sample

        Returns:
            embeds: [batch_size, n_cams, embed_dim] - 3D tensor with separate camera features
        """
        if self.robot_only_mode:
            return self._embed_robot_only(imgs, action_dims)
        else:
            return self._embed_dual(imgs, action_dims)

    def _embed_dual(self, imgs, action_dims):
        """
        Dual mode embedding: separate processing for human and robot data.

        Human data:
          - cam0: visual_features[0] → human_vision_adaptor
          - cam1~camN: all zeros

        Robot data:
          - cam0: visual_features[0] → sim_vision_adaptor[0]
          - cam1~camN: 90% real features via visual_features[i] → sim_vision_adaptor[i]
                       10% all zeros (shared dropout for all cam1~camN)

        Args:
            imgs: dictionary of camera images
            action_dims: tensor [batch_size], indicating action dimension for each sample

        Returns:
            embeds: [batch_size, n_cams, embed_dim]
        """
        # Get batch info
        batch_size = imgs["cam0"].shape[0]
        device = imgs["cam0"].device
        
        # Validate action_dims
        if action_dims is None:
            raise ValueError("action_dims must be provided in dual mode")

        # Create masks
        robot_mask = (action_dims == self.robot_action_dim)
        human_mask = (action_dims == self.human_action_dim)


        embed_dim = self.noise_net.hidden_dim
        # Initialize output: [batch_size, n_cams, embed_dim]
        all_embeds = torch.zeros(batch_size, self._n_cams, embed_dim, device=device)

        # ============================================================
        # Process cam0 for all samples
        # ============================================================
        # import ipdb; ipdb.set_trace()
        # Process cam0 features
        if self._share_cam_features:
            cam0_features = self._embed_helper(self.visual_features, imgs["cam0"])
        else:
            cam0_features = self._embed_helper(self.visual_features[0], imgs["cam0"])
        
        # Apply adaptors to cam0 based on data type
        if human_mask.any():
            # Human data: apply human_vision_adaptor
            human_cam0_features = cam0_features[human_mask]
            human_cam0_adapted = self.human_vision_adaptor(human_cam0_features)
            all_embeds[human_mask, 0, :] = human_cam0_adapted.squeeze(1).to(all_embeds.dtype)

        if robot_mask.any():
            # Robot data: apply sim_vision_adaptor[0]
            robot_cam0_features = cam0_features[robot_mask]
            if self._share_cam_features:
                robot_cam0_adapted = self.sim_vision_adaptor(robot_cam0_features)
            else:
                robot_cam0_adapted = self.sim_vision_adaptor[0](robot_cam0_features)
            # robot_cam0_adapted = self.sim_vision_adaptor(robot_cam0_features)
            all_embeds[robot_mask, 0, :] = robot_cam0_adapted.squeeze(1).to(all_embeds.dtype)

        # ============================================================
        # Process cam1~camN
        # Human: keep as zeros (already initialized)
        # Robot: 90% real features, 10% zeros (shared dropout)
        # ============================================================

        if robot_mask.any():
            num_robot = robot_mask.sum()
            robot_indices = torch.where(robot_mask)[0]  # [num_robot]

            # Generate shared dropout mask for cam1~camN
            # 90% probability to compute features (True), 10% to keep zeros (False)
            dropout_mask = torch.rand(num_robot, device=device) > self.dropout_wrist_rate  # [num_robot]

            # Process each camera cam1~camN with the same dropout_mask
            for i in range(1, self._n_cams):
                if dropout_mask.any():
                    # Extract robot samples' cam{i} images
                    robot_imgs_cami = imgs[f"cam{i}"][robot_mask]  # [num_robot, C, H, W]

                    # Select samples with dropout_mask=True
                    selected_imgs = robot_imgs_cami[dropout_mask]  # [num_selected, C, H, W]

                    # Compute features
                    if self._share_cam_features:
                        cami_features = self._embed_helper(self.visual_features, selected_imgs)
                        cami_adapted = self.sim_vision_adaptor(cami_features)
                    else:
                        cami_features = self._embed_helper(self.visual_features[i], selected_imgs)
                        cami_adapted = self.sim_vision_adaptor[i](cami_features)

                    # Fill back to corresponding positions
                    selected_indices = robot_indices[dropout_mask]  # [num_selected]
                    all_embeds[selected_indices, i, :] = cami_adapted.squeeze(1).to(all_embeds.dtype)

        # Reshape and return: [batch_size, n_cams * embed_dim]
        return all_embeds

    def _embed_robot_only(self, imgs, action_dims):
        """
        Robot-only mode embedding:
           - cam0: vision_features → human_adaptor (if use_human_adaptor=True)
                   or sim_adaptor[0] (if use_human_adaptor=False)
           - cam1~camN: vision_features[i] → sim_adaptor[i] (no dropout)

        Args:
            imgs: dictionary of camera images
            action_dims: tensor [batch_size], indicating action dimension for each sample

        Returns:
            embeds: [batch_size, n_cams, embed_dim]
        """

        # ============================================================
        # Step 1: Extract visual features for all cameras
        # ============================================================
        cam_features = []
        for i in range(self._n_cams):
            if self._share_cam_features:
                net = self.visual_features
            else:
                net = self.visual_features[i]

            feat = self._embed_helper(net, imgs[f"cam{i}"])
            # Ensure 2D: [batch_size, embed_dim]
            feat = self._process_adaptor_output(feat)
            cam_features.append(feat)

        # ============================================================
        # Step 2: Apply adaptors based on mode
        # ============================================================
        cam_outputs = []

       
        cam0_feat = cam_features[0]
        if self.use_human_adaptor:
            cam0_output = self.human_vision_adaptor(cam0_feat)
        else:
            # cam0_output = self.sim_vision_adaptor(cam0_feat)
            if self._share_cam_features:
                cam0_output = self.sim_vision_adaptor(cam0_feat)
            else:
                cam0_output = self.sim_vision_adaptor[0](cam0_feat)
        cam0_output = self._process_adaptor_output(cam0_output)
        cam_outputs.append(cam0_output)

        # Process cam1~camN
        for i in range(1, self._n_cams):
            cami_feat = cam_features[i]
            if self._share_cam_features:
                cami_output = self.sim_vision_adaptor(cami_feat)
            else:
                cami_output = self.sim_vision_adaptor[i](cami_feat)
            cami_output = self._process_adaptor_output(cami_output)
            cam_outputs.append(cami_output)

        # ============================================================
        # Step 3: Stack all outputs
        # ============================================================
        all_embeds = torch.stack(cam_outputs, dim=1)  # [batch_size, n_cams, embed_dim]

        return all_embeds

    def _resolve_action_dims(self, obs):
        """
        Unified action dimension resolution and validation
        
        Args:
            obs: observation dictionary
            
        Returns:
            action_dims: validated action dimension tensor [batch_size]
        """
        action_dims = obs.get('action_dim', None)
        if action_dims is None:
            raise ValueError("obs must contain 'action_dim' key")
        
        if self.robot_only_mode:
            # Verify all inputs are robot data
            if not torch.all(action_dims == self.robot_action_dim):
                raise ValueError(f"robot_only_mode expects all action_dims={self.robot_action_dim}, got {action_dims}")
        else:
            # Dual mode validation
            valid_dims = (action_dims == self.robot_action_dim) | (action_dims == self.human_action_dim)
            if not torch.all(valid_dims):
                raise ValueError(f"Invalid action_dims in dual mode: expected {self.robot_action_dim} or {self.human_action_dim}, got {action_dims}")
        
        return action_dims
    
    def _get_obs_enc(self, imgs, obs, action_dims):
        """
        Get observation encoding with separate state processing based on data type.
        Uses BaseAgent's tokenize_obs but with custom state processing.
        """
        # Process images using our overridden embed method with action_dims
        img_tokens = self.embed(imgs, action_dims)
        
        # Extract state
        if isinstance(obs, dict):
            state = obs['state']
        else:
            state = obs
        
        batch_size = state.shape[0]
        state_tokens = torch.zeros(batch_size, 1, img_tokens.shape[-1], device=state.device)
        
        if self.robot_only_mode:
            # Robot-only mode: all samples use robot encoder
            robot_states = state[:, :self.robot_state_dim]
            state_tokens[:, 0] = self.robot_state_encoder(robot_states)
        else:
            # Dual mode: route based on action_dims
            robot_mask = (action_dims == self.robot_action_dim)
            human_mask = (action_dims == self.human_action_dim)
            
            if robot_mask.any():
                robot_states = state[robot_mask, :self.robot_state_dim]
                robot_state_tokens = self.robot_state_encoder(robot_states)
                state_tokens[robot_mask, 0] = robot_state_tokens.to(state_tokens.dtype)

            if human_mask.any():
                human_states = state[human_mask, :self.human_state_dim]
                human_state_tokens = self.human_state_encoder(human_states)
                state_tokens[human_mask, 0] = human_state_tokens.to(state_tokens.dtype)
        
        # Concatenate image and state tokens (following BaseAgent pattern)
        obs_enc = torch.cat([img_tokens, state_tokens], dim=1)
        
        # Apply post-processing (following BaseAgent pattern)
        obs_enc = self.post_proc(obs_enc)
        
        return obs_enc
    
    def forward(self, imgs, obs, ac_flat, mask, **kwargs):
        """
        Forward pass with dual-path support
        """
        # Unified action dimension resolution
        action_dims = self._resolve_action_dims(obs)
        
        # Get batch info and setup
        batch_size, device = obs['state'].shape[0], obs['state'].device
        
        # Get observation encoding with separate state processing
        obs_enc = self._get_obs_enc(imgs, obs, action_dims)
        
        # Standard diffusion forward pass
        time = torch.randint(
            0,
            self._train_diffusion_steps,
            (batch_size,),
            dtype=torch.long,
            device=ac_flat.device,
        )
        
        # Reshape actions and mask based on mode
        if self.robot_only_mode:
            # Robot-only: use robot_action_dim for reshaping
            actual_ac_dim = self.robot_action_dim
            mask = mask.reshape((batch_size, self._ac_chunk, actual_ac_dim))
            actions = ac_flat.reshape((batch_size, self._ac_chunk, actual_ac_dim))
        else:
            # Dual mode: use human_action_dim as max
            actual_ac_dim = self.human_action_dim
            mask = mask.reshape((batch_size, self._ac_chunk, actual_ac_dim))
            actions = ac_flat.reshape((batch_size, self._ac_chunk, actual_ac_dim))
        
        # Sample noise
        noise = torch.randn_like(actions, device=ac_flat.device)
        noise_actions = self.diffusion_schedule.add_noise(actions, noise, time)
        
        # Predict noise with dual routing - returns structured predictions
        pred_dict = self.noise_net(noise_actions, time, obs_enc, action_dims)
        
        # Calculate loss based on mode
        if self.robot_only_mode:
            return self._calculate_robot_loss(pred_dict, noise, mask, device)
        else:
            return self._calculate_dual_loss(pred_dict, noise, mask, action_dims, device)
    
    def _calculate_dual_loss(self, pred_dict, noise, mask, action_dims, device):
        """
        Calculate dual loss from structured predictions
        
        Args:
            pred_dict: Dictionary containing 'robot' and 'human' predictions
            noise: Ground truth noise [batch_size, seq_len, human_action_dim]
            mask: Loss mask [batch_size, seq_len, human_action_dim]
            action_dims: Action dimension indicators [batch_size]
            device: Device
        
        Returns:
            Mean loss across all samples
        """
        batch_size = noise.shape[0]
        loss_total = torch.zeros(batch_size, device=device)
        
        robot_pred = pred_dict['robot']  # [batch_size, seq_len, 16]
        human_pred = pred_dict['human']  # [batch_size, seq_len, 44]
        
        robot_mask = (action_dims == self.robot_action_dim)
        human_mask = (action_dims == self.human_action_dim)


        # Robot samples: only use robot prediction (16-dim)
        if robot_mask.any():
            robot_target = noise[robot_mask, :, :self.robot_action_dim]
            robot_mask_weights = mask[robot_mask, :, :self.robot_action_dim]
            robot_loss_batch = F.mse_loss(robot_pred[robot_mask], robot_target, reduction="none")
            robot_loss_batch = (robot_loss_batch * robot_mask_weights).sum(1)
            loss_total[robot_mask] = robot_loss_batch.mean()


        # Human samples: use human prediction (44-dim)
        if human_mask.any():
            human_target = noise[human_mask, :, :self.human_action_dim]
            human_mask_weights = mask[human_mask, :, :self.human_action_dim]
            human_loss_batch = F.mse_loss(human_pred[human_mask], human_target, reduction="none")
            human_loss_batch = (human_loss_batch * human_mask_weights).sum(1)
            loss_total[human_mask] = human_loss_batch.mean()
  
        
        return {'loss': loss_total.mean(), 'robot_loss': loss_total[robot_mask].mean(), 'human_robot_loss': torch.tensor(0.0, device=device), 'human_hands_loss': loss_total[human_mask].mean()}
    
    def _calculate_robot_loss(self, pred_dict, noise, mask, device):
        """
        Calculate loss for robot-only mode.
        Only uses robot prediction (16-dim).
        
        Args:
            pred_dict: Dictionary containing 'robot' predictions
            noise: Ground truth noise [batch_size, seq_len, robot_action_dim]
            mask: Loss mask [batch_size, seq_len, robot_action_dim]
            device: Device
        
        Returns:
            Mean loss for robot-only training
        """
        robot_pred = pred_dict['robot']  # [batch_size, seq_len, 16]
        
        # In robot-only mode, noise and mask are already robot_action_dim
        robot_loss = F.mse_loss(robot_pred, noise, reduction="none")
        robot_loss = (robot_loss * mask).sum(1)
        
        return robot_loss.mean()
    
    def _combine_predictions_for_diffusion(self, pred_dict, action_dims):
        """
        Combine structured predictions into unified format for diffusion step
        
        Args:
            pred_dict: Dictionary containing 'robot' and 'human' predictions
            action_dims: Action dimension indicators
        
        Returns:
            Unified noise prediction [batch_size, seq_len, action_dim]
        """
        robot_pred = pred_dict['robot']  # [batch_size, seq_len, 16]
        
        if self.robot_only_mode:
            # Robot-only mode: return robot predictions directly
            return robot_pred
        
        # Dual mode: combine based on action_dims
        human_pred = pred_dict['human']  # [batch_size, seq_len, 44]
        
        batch_size, seq_len = robot_pred.shape[:2]
        device = robot_pred.device
        
        # Create unified output with human_action_dim
        unified_pred = torch.zeros(batch_size, seq_len, self.human_action_dim, device=device)
        
        robot_mask = (action_dims == self.robot_action_dim)
        human_mask = (action_dims == self.human_action_dim)
        
        # Robot samples: use only robot prediction (first 16 dims)
        if robot_mask.any():
            unified_pred[robot_mask, :, :self.robot_action_dim] = robot_pred[robot_mask]
        
        # Human samples: use human prediction (44 dims)
        if human_mask.any():
            unified_pred[human_mask, :, :self.human_action_dim] = human_pred[human_mask]
        
        return unified_pred
    
    def get_actions(self, imgs, obs, n_steps=None):
        """
        Generate actions for inference with dual-path support.
        
        Args:
            imgs: Input images
            obs: Observations (should contain 'action_dim' key or will default to human mode)
            n_steps: Number of diffusion steps
        
        Returns:
            If robot_only_mode:
                tensor: [batch_size, ac_chunk, robot_action_dim]
            Else (dual mode):
                dict: {
                    'robot': [batch_size, ac_chunk, robot_action_dim] - Robot actions
                    'human': [batch_size, ac_chunk, human_action_dim] - Human actions  
                    'action_dims': [batch_size] - Action dimension indicators
                }
        """
        # Get batch info
        if isinstance(obs, dict):
            B, device = obs['state'].shape[0], obs['state'].device
        else:
            B, device = obs.shape[0], obs.device
        #import ipdb; ipdb.set_trace()  # Removed debug breakpoint
        # Handle action dimensions
        if self.robot_only_mode:
            # Robot-only mode: force robot dimensions
            action_dims = torch.full((B,), self.robot_action_dim, device=device)
        else:
            # Dual mode: use provided or default to human
            action_dims = obs.get('action_dim', None) if isinstance(obs, dict) else None
            if action_dims is None:
                # During inference, default to human action dimensions for maximum capability
                action_dims = torch.full((B,), self.human_action_dim, device=device)
        # import ipdb; ipdb.set_trace()
        # Get observation encoding
        obs_enc = self._get_obs_enc(imgs, obs, action_dims)
        # Initialize noise actions based on mode
        if self.robot_only_mode:
            noise_actions = torch.randn(B, self._ac_chunk, self.robot_action_dim, device=device)
        else:
            noise_actions = torch.randn(B, self._ac_chunk, self.human_action_dim, device=device)
        
        # Set number of diffusion steps
        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert n_steps <= self._train_diffusion_steps, f"Cannot be greater than {self._train_diffusion_steps}"
            eval_steps = n_steps
        
        # Begin diffusion process
        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alphas_cumprod = self.diffusion_schedule.alphas_cumprod.to(device)
        enc_cache = self.noise_net.forward_enc(obs_enc)
        for timestep in self.diffusion_schedule.timesteps:
            # Predict noise with dual routing
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            pred_dict = self.noise_net.forward_dec(noise_actions, batched_timestep, enc_cache, action_dims)
            
            # Convert structured predictions back to unified noise prediction
            noise_pred = self._combine_predictions_for_diffusion(pred_dict, action_dims)
            
            # Take diffusion step
            noise_actions = self.diffusion_schedule.step(
                model_output=noise_pred, timestep=timestep, sample=noise_actions
            ).prev_sample
        
        # Return actions based on mode
        if self.robot_only_mode:
            # Robot-only mode: return tensor directly
            return noise_actions
        else:
            robot_mask = (action_dims == self.robot_action_dim)
            # Dual mode: return structured dictionary
            return {
                'robot': noise_actions[robot_mask, :, :self.robot_action_dim],  # [B, ac_chunk, robot_action_dim]
                'human_hands': noise_actions[~robot_mask],                                 # [B, ac_chunk, human_action_dim]
                'action_dims': action_dims                              # [B]
            }