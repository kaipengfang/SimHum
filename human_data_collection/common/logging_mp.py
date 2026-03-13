"""
Simple logging_mp replacement module
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR

def get_logger(name, level=logging.INFO):
    """Get a logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
