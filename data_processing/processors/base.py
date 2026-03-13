"""
Base data structures and abstract processor class

Core abstractions following KISS principle - simple, focused interfaces.
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from .common.image_utils import resize_and_encode, create_dummy_image
from .common.constants import (
    IMAGE_SIZE, CAM_NAMES, AGILEX_CAM_NAMES, ROBOT_DIM, HUMAN_DIM,
    HUMAN_MIN_DISTANCE, HUMAN_MAX_DISTANCE, HUMAN_RANGE
)


@dataclass
class DataInfo:
    """Data source information container"""
    data_type: str
    num_tasks: int
    task_names: List[str]
    total_episodes: int
    output_dimension: int
    has_video: bool = False


@dataclass
class EpisodeData:
    """Single episode data container"""
    states: np.ndarray  # State sequence (T, dim) - raw data, not normalized
    actions: np.ndarray  # Action sequence (T-1, dim) - raw data, not normalized
    images: Dict[str, np.ndarray]  # Image sequences per camera
    episode_length: int


class BaseDataProcessor(ABC):
    """
    Abstract base class for all data processors.
    
    Simplified to pure interface following KISS principle.
    Common functionality moved to utility modules following DRY principle.
    
    Args:
        config_section: Configuration dictionary for this data source
        task_id_offset: Starting task ID for this processor (for multi-source mixing)
    """
    
    def __init__(self, config_section: Dict[str, Any], task_id_offset: int = 0):
        self.config = config_section
        self.task_id_offset = task_id_offset
        self.data_path = os.path.expanduser(config_section['path'])
        self.tasks = config_section.get('tasks', [])
        self.max_episodes = config_section.get('max_episodes_per_task', None)
        
        # Human output format configuration (46, 44, or 16 dimensions)
        self.human_output_format = config_section.get('human_output_format', 46)
        if self.human_output_format not in [46, 44, 16]:
            raise ValueError(f"Invalid human_output_format: {self.human_output_format}. Supported: 46, 44, 16")
        
        # Validate data path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
    
    def pad_to_target_dim(self, data: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Pad data to target dimension.
        
        For robot data (16->46): pad with zeros
        For human data (46->46): return as-is
        
        Args:
            data: Input data array
            target_dim: Target dimension
            
        Returns:
            Padded data array
        """
        if data.shape[-1] == target_dim:
            return data
        elif data.shape[-1] < target_dim:
            # Pad with zeros
            if len(data.shape) == 1:
                padded = np.zeros(target_dim, dtype=data.dtype)
                padded[:data.shape[0]] = data
            else:
                padded = np.zeros((*data.shape[:-1], target_dim), dtype=data.dtype)
                padded[..., :data.shape[-1]] = data
            return padded
        else:
            raise ValueError(f"Cannot pad from {data.shape[-1]} to {target_dim} dimensions")

    def _resize_and_encode(self, bgr_image: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
        """
        Resize image to target size and encode as JPEG.
        Wrapper for common image utility function.
        """
        return resize_and_encode(bgr_image, size)

    def create_dummy_image(self, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
        """
        Create a dummy black image for missing camera data.
        Wrapper for common image utility function.
        """
        return create_dummy_image(size)

    @abstractmethod
    def get_data_info(self) -> DataInfo:
        """Get basic information about this data source"""
        pass

    @abstractmethod
    def load_episodes(self) -> List[Tuple]:
        """
        Find all episode files for this data source.
        
        Returns:
            List of episode identifier tuples
        """
        pass

    @abstractmethod  
    def load_single_episode(self, episode_info: Tuple) -> Optional[EpisodeData]:
        """
        Load data from a single episode file.
        
        Args:
            episode_info: Episode identifier tuple
            
        Returns:
            EpisodeData object or None if loading failed
        """
        pass

    @abstractmethod
    def get_output_dimension(self) -> int:
        """Get the natural output dimension for this data source"""
        pass