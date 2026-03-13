"""
Configuration Module
Configuration management and initialization
"""
import os
import re
from pathlib import Path
from datetime import datetime


class CaptureConfig:
    """Capture system configuration"""

    def __init__(
        self,
        description,
        args,
        freq=30,
        path="data/integrated_recordings",
        image_server_address="localhost",
        image_server_port=5555
    ):
        self.description = description
        self.args = args
        self.freq = freq
        self.base_path = path
        self.image_server_address = image_server_address
        self.image_server_port = image_server_port

        # VR settings
        self.resolution = (720, 1280)
        self.crop_size_w = 160
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w
        )

        # Recording settings
        self.if_record = not args.get('no_record', False)

        # Create task-specific path
        self.task_path = self.create_task_path(path, description)

    def clean_task_name(self, task_name):
        """
        Clean task name, remove special characters, replace spaces with underscores

        Args:
            task_name: Original task name

        Returns:
            str: Cleaned task name
        """
        # Replace spaces with underscores
        cleaned = task_name.replace(' ', '_')
        # Remove special characters, keep only letters, numbers, underscores, and hyphens
        cleaned = re.sub(r'[^\w\-]', '', cleaned)
        return cleaned

    def create_task_path(self, base_path, description):
        """
        Create task-specific save path

        Args:
            base_path: Base path
            description: Task description

        Returns:
            str: Task-specific path
        """
        cleaned_desc = self.clean_task_name(description)
        task_path = os.path.join(base_path, cleaned_desc)
        Path(task_path).mkdir(parents=True, exist_ok=True)
        return task_path

    def get_image_shape(self):
        """Get image shape for shared memory"""
        return (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
