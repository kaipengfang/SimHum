"""
Data processor factory

Factory function for creating appropriate data processors.
Following KISS principle - simple factory pattern implementation.
"""

from typing import Dict, Any

from .base import BaseDataProcessor
from .robot.sim_processor import SimRobotProcessor
from .robot.agilex_processor import AgilexRobotProcessor
from .human.collect_processor import HumanCollectProcessor


def create_processor(data_type: str, config_section: Dict[str, Any], task_id_offset: int = 0) -> BaseDataProcessor:
    """
    Factory function to create appropriate data processor.
    
    Args:
        data_type: Type of data processor ("SimRobot", "AgilexRobot", "HumanCollect")
        config_section: Configuration dictionary for the processor
        task_id_offset: Starting task ID offset for this processor
        
    Returns:
        Instantiated data processor
        
    Raises:
        ValueError: If data_type is not recognized
    """
    processor_map = {
        "SimRobot": SimRobotProcessor,
        "AgilexRobot": AgilexRobotProcessor,
        "HumanCollect": HumanCollectProcessor
    }
    
    if data_type not in processor_map:
        raise ValueError(f"Unknown data type: {data_type}. Supported types: {list(processor_map.keys())}")
    
    processor_class = processor_map[data_type]
    return processor_class(config_section, task_id_offset)