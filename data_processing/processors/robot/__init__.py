"""
Robot data processors

Handles simulated and real robot data processing.
"""

from .sim_processor import SimRobotProcessor
from .agilex_processor import AgilexRobotProcessor

__all__ = ['SimRobotProcessor', 'AgilexRobotProcessor']