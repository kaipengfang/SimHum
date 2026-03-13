# Copyright (c) 2024
# DinoV2 Vision Encoder for DIT Policy
# Compatible with ResNet and ViT interfaces

import os
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DinoV2(nn.Module):
    """
    DinoV2 vision encoder compatible with ResNet/ViT interface.

    Args:
        size (str): Model size - 'small', 'base', 'large', 'giant'
        avg_pool (bool): If True, return only CLS token (1 token).
                        If False, return all patch tokens (256 tokens for 224x224)
        select_feature (str): 'cls', 'patch', or 'cls_patch'
    """

    def __init__(
        self,
        size='base',
        avg_pool=True,
        select_feature=None,
    ):
        super().__init__()

        # Model name mapping
        self.model_mapping = {
            'small': ('facebook/dinov2-small', 384),
            'base': ('facebook/dinov2-base', 768),
            'large': ('facebook/dinov2-large', 1024),
            'giant': ('facebook/dinov2-giant', 1536),
        }

        if size not in self.model_mapping:
            raise ValueError(f"Invalid size: {size}. Choose from {list(self.model_mapping.keys())}")

        self.size = size
        self.vision_tower_name, self._embed_dim = self.model_mapping[size]
        self._avg_pool = avg_pool

        # Determine feature selection mode
        if select_feature is None:
            # Default: match avg_pool behavior
            self.select_feature = 'cls' if avg_pool else 'patch'
        else:
            self.select_feature = select_feature

        # Load model
        self._load_model()

    def _load_model(self):
        """Load DinoV2 model from HuggingFace or local checkpoint."""

        # Load image processor (lightweight, config only)
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)

        # Load model structure
        self.model = AutoModel.from_pretrained(self.vision_tower_name)

        # Freeze model parameters (for feature extraction)
        self.model.requires_grad_(False)
        self.model.eval()

        # Report parameter statistics
        self._print_parameter_stats()

    def forward(self, x):
        """
        Forward pass through DinoV2.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            features: Shape (B, n_tokens, embed_dim)
                     - If avg_pool=True: (B, 1, embed_dim)
                     - If avg_pool=False: (B, 256, embed_dim) for 224x224 images
        """
        with torch.no_grad():
            outputs = self.model(x)
            features = outputs.last_hidden_state  # (B, num_tokens, embed_dim)

            # Select features based on mode
            if self.select_feature == 'cls':
                # Use only CLS token
                features = features[:, :1]  # (B, 1, embed_dim)
            elif self.select_feature == 'patch':
                # Use only patch tokens (exclude CLS)
                features = features[:, 1:]  # (B, num_patches, embed_dim)
            elif self.select_feature == 'cls_patch':
                # Use all tokens (CLS + patches)
                features = features  # (B, num_patches+1, embed_dim)
            else:
                raise ValueError(f"Invalid select_feature: {self.select_feature}")

            return features

    def _print_parameter_stats(self):
        """Print parameter statistics (total, trainable, frozen)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # ANSI color codes
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

        print(f"{BOLD}{CYAN}🖼️  DinoV2-{self.size} Vision Encoder Initialized:{RESET}")
        print(f"   🔢 Total parameters: {BOLD}{GREEN}{total_params:,}{RESET} ({BOLD}{GREEN}{total_params / 1e6:.2f}M{RESET})")
        print(f"   🎯 Trainable parameters: {BOLD}{YELLOW}{trainable_params:,}{RESET} ({BOLD}{YELLOW}{trainable_params / 1e6:.2f}M{RESET})")
        print(f"   ❄️  Frozen parameters: {BOLD}{BLUE}{frozen_params:,}{RESET} ({BOLD}{BLUE}{frozen_params / 1e6:.2f}M{RESET})")
        print(f"   📊 Embed dim: {BOLD}{GREEN}{self._embed_dim}{RESET}, Tokens: {BOLD}{GREEN}{self.n_tokens}{RESET}")
    
    @property
    def embed_dim(self):
        """Return embedding dimension (matches ResNet interface)."""
        return self._embed_dim

    @property
    def n_tokens(self):
        """
        Return number of output tokens (matches ResNet interface).

        Returns:
            int: Number of tokens (dynamically calculated based on model config)
                - 1 if select_feature='cls' (CLS token only)
                - num_patches if select_feature='patch' (patch tokens only)
                - num_patches+1 if select_feature='cls_patch' (CLS + patches)

        Note: The actual number depends on the model's config (image_size and patch_size).
        """
        if self.select_feature == 'cls':
            return 1
        elif self.select_feature == 'patch':
            # Dynamic calculation based on model config
            return self.num_patches
        elif self.select_feature == 'cls_patch':
            # CLS token + all patch tokens
            return self.num_patches + 1
        else:
            return 1 if self._avg_pool else self.num_patches

    @property
    def num_patches(self):
        """
        Return number of patches based on actual model config.

        Note: DinoV2 config has image_size=518, but most robotics datasets
        use 224x224 images. The actual number of patches depends on input size.
        """
        config = self.model.config
        # Return config value (518/14)^2 = 1369, but actual will be based on input
        return (config.image_size // config.patch_size) ** 2

    @property
    def device(self):
        """Return device of the model."""
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        """Return dtype of the model."""
        return next(self.model.parameters()).dtype


# Convenience functions for different sizes
def dinov2_small(avg_pool=True, **kwargs):
    """DinoV2-Small (embed_dim=384)"""
    return DinoV2(size='small', avg_pool=avg_pool, **kwargs)


def dinov2_base(avg_pool=True, **kwargs):
    """DinoV2-Base (embed_dim=768)"""
    return DinoV2(size='base', avg_pool=avg_pool, **kwargs)


def dinov2_large(avg_pool=True, **kwargs):
    """DinoV2-Large (embed_dim=1024)"""
    return DinoV2(size='large', avg_pool=avg_pool, **kwargs)


def dinov2_giant(avg_pool=True, **kwargs):
    """DinoV2-Giant (embed_dim=1536)"""
    return DinoV2(size='giant', avg_pool=avg_pool, **kwargs)