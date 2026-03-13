"""
Unified Data Processors Package

Provides modular data processors for different robotics datasets.
Follows KISS, DRY, LOD principles for maintainable code architecture.

"""

from .factory import create_processor
from .base import (
    BaseDataProcessor, DataInfo, EpisodeData, 
    ROBOT_DIM, HUMAN_DIM, IMAGE_SIZE, CAM_NAMES, AGILEX_CAM_NAMES,
    HUMAN_MIN_DISTANCE, HUMAN_MAX_DISTANCE, HUMAN_RANGE
)

# Import all processors for direct access
from .robot import SimRobotProcessor, AgilexRobotProcessor  
from .human import HumanCollectProcessor

__all__ = [
    'create_processor',
    'BaseDataProcessor', 'DataInfo', 'EpisodeData',
    'SimRobotProcessor', 'AgilexRobotProcessor', 
    'HumanCollectProcessor',
    'ROBOT_DIM', 'HUMAN_DIM', 'IMAGE_SIZE', 'CAM_NAMES', 'AGILEX_CAM_NAMES',
    'HUMAN_MIN_DISTANCE', 'HUMAN_MAX_DISTANCE', 'HUMAN_RANGE'
]