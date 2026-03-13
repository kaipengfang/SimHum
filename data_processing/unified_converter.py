#!/usr/bin/env python3
"""
Unified Data Converter for Robotics Tasks

This tool converts different robotics datasets into a unified format for training.
It supports mixing robot and human data with automatic dimension alignment.

Supported Data Sources:
1. SimRobot: Simulated robot data with endpose actions (16-dim)
2. RealRobot: Real robot (AgileX) data with eef_pose (16-dim)  
3. HumanCollect: Self-collected human data (44-dim)

Output Data Structure:
- buf.pkl: Main trajectory buffer (OPTIMIZED)
  * List[Tuple[obs, action, reward]] - flattened trajectory steps, no episode grouping
  * obs: {'state': np.array, 'enc_cam_0': bytes, 'enc_cam_1': bytes, 
          'enc_cam_2': bytes, 'instruction_id': int, 'action_dim': int}
  * action: np.array (16-dim or 46-dim based on data mix)
  * reward: float (always 0 for imitation learning)


- Language files:
  * task_embeddings.pkl: List[np.ndarray] - language embeddings
  * task_instruction.txt: Human-readable task descriptions
  * task_mapping.json: Task name to ID mapping

Usage Examples:
    # Single robot data source (16-dim output)
    python unified_converter.py --config robot_only.yaml
    
    # Mixed robot+human data (46-dim output, robot data padded)
    python unified_converter.py --config mixed_data.yaml
    
    # Custom output path
    python unified_converter.py --config config.yaml --output /custom/path
    
    # Verbose processing with progress
    python unified_converter.py --config config.yaml --verbose --progress

Dependencies:
    - data_processors: Contains all data processor classes
    - rich: For beautiful terminal output
    - yaml: For configuration file parsing

"""

import os
import sys
import yaml
import pickle as pkl
import numpy as np
import argparse
from typing import Dict, List, Any, Optional



# Import our custom processors
from processors import (
    create_processor, BaseDataProcessor, DataInfo, EpisodeData,
    ROBOT_DIM, HUMAN_DIM
)
from display_helper import DisplayHelper


# Normalization functions removed - normalization now handled at training time


class UnifiedConverter:

    def __init__(self, config_path: str, output_override: Optional[str] = None):
        """
        Initialize converter with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            output_override: Optional override for output path
        """
        self.display = DisplayHelper()
        self.config = self._load_config(config_path)
        self.processors: List[BaseDataProcessor] = []
        
        # Simple statistics variables
        self.total_episodes = 0
        self.total_samples = 0
        self.processing_time = 0.0
        
        # Override output path if specified
        if output_override:
            self.config['output']['target_path'] = output_override
        
        self._validate_config()
        self._create_processors()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self):
        """Validate configuration structure and required fields"""
        required_fields = ['data_sources', 'output']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        if not self.config['data_sources']:
            raise ValueError("At least one data source must be specified")
            
        # Validate each data source
        for i, source in enumerate(self.config['data_sources']):
            if 'type' not in source or 'path' not in source:
                raise ValueError(f"Data source {i} missing required fields 'type' or 'path'")
    
    def _create_processors(self):
        """Create processor instances for each data source"""
        task_id_offset = 0
        coord_aligned_sources = []
        
        for source_config in self.config['data_sources']:
            processor = create_processor(
                data_type=source_config['type'],
                config_section=source_config,
                task_id_offset=task_id_offset
            )
            self.processors.append(processor)
            
            # Check if coordinate alignment is enabled
            if source_config.get('type') == 'SimRobot' and source_config.get('align_to_human_coords', False):
                coord_aligned_sources.append(source_config.get('type', 'Unknown'))
            
            # Update task ID offset for next processor
            data_info = processor.get_data_info()
            task_id_offset += data_info.num_tasks
        
        # This is now handled in process_all_data
    
    def _detect_target_dimension(self) -> int:
        """
        Detect target dimension based on data sources.
        
        Logic:
        - Return the maximum dimension among all data sources
        - This allows for flexible human data formats (46, 44, 16)
        
        Returns:
            Target dimension (max dimension from all sources)
        """
        max_dimension = ROBOT_DIM  # Default to robot dimension
        
        for processor in self.processors:
            output_dim = processor.get_output_dimension()
            max_dimension = max(max_dimension, output_dim)
        
        return max_dimension
    
    def _create_output_directory(self) -> str:
        """Create and return output directory path"""
        target_path = os.path.expanduser(self.config['output']['target_path'])
        dataset_name = self.config['output'].get('dataset_name', f'mixed_{len(self.processors)}')
        
        final_path = os.path.join(target_path, dataset_name)
        os.makedirs(final_path, exist_ok=True)
        
        return final_path
    
    def process_all_data(self):
        """Main processing pipeline - simplified approach similar to converter"""
        import time
        start_time = time.time()
        
        # 1. Display processing header
        self.display.print_header()
        self.display.print_data_source_summary(self.processors)
        
        # 2. Detect target dimension and show info
        target_dim = self._detect_target_dimension()
        self.display.print_target_dimension(target_dim, ROBOT_DIM, HUMAN_DIM)
        
        # Check coordinate alignment
        coord_aligned_sources = []
        for processor in self.processors:
            processor_config = getattr(processor, 'config', {})
            if processor_config.get('align_to_human_coords', False):
                coord_aligned_sources.append(processor.get_data_info().data_type)
        self.display.print_coordinate_alignment(coord_aligned_sources)
        
        # 3. Create output directory
        output_path = self._create_output_directory()
        self.display.print_output_directory(output_path)

        # 4. Single pass processing - collect data and generate trajectories
        processor_data = self._process_all_episodes(target_dim)

        # 5. Combine all trajectories
        all_trajs = []
        for data in processor_data.values():
            all_trajs.extend(data['trajs'])

        self.total_episodes = len(all_trajs)
        self.total_samples = sum(len(traj) for traj in all_trajs)

        # 6. Save trajectory data (normalization now handled at training time)
        self._save_all_data(output_path, all_trajs)
        
        # 8. Display final summary
        end_time = time.time()
        self.processing_time = end_time - start_time
        self.display.print_final_summary(
            output_path, self.total_episodes, self.total_samples, 
            self.processing_time, target_dim
        )
    
    def _process_all_episodes(self, target_dim):
        """Single pass processing - collect data by processor and generate trajectories"""
        self.display.print_data_processing_start()

        processor_data = {}

        for i, processor in enumerate(self.processors):
            processor_data[i] = {'trajs': []}
            data_info = processor.get_data_info()
            episodes = processor.load_episodes()

            self.display.print_processor_info(data_info.data_type, len(episodes))

            # Process episodes with progress bar
            pbar = self.display.create_progress_bar(episodes, f"Processing {data_info.data_type}")

            for episode_info in pbar:
                try:
                    # Load raw episode data
                    episode_data = processor.load_single_episode(episode_info)
                    if episode_data is None:
                        continue

                    # Generate episode trajectory (without normalization)
                    episode_traj = self._convert_episode_to_trajectory(
                        episode_data, episode_info, processor, target_dim
                    )

                    if episode_traj:
                        processor_data[i]['trajs'].append(episode_traj)

                except Exception as e:
                    self.display.print_error(f"Processing episode: {e}")
                    continue

        return processor_data

    def _convert_episode_to_trajectory(self, episode_data, episode_info, processor, target_dim):
        """Convert episode data to trajectory format (similar to converter approach)"""
        if len(episode_info) == 3:
            # Human format: (task_id, hdf5_path, mp4_path)
            task_id = episode_info[0]
        else:
            # Other formats: (task_id, file_path)
            task_id = episode_info[0]
        
        trajectory = []
        
        dummy_left = processor.create_dummy_image() if hasattr(processor, 'create_dummy_image') else None
        dummy_right = processor.create_dummy_image() if hasattr(processor, 'create_dummy_image') else None
        
        # Process each timestep
        for t in range(len(episode_data.actions)):
            try:
                # Prepare observation
                obs = {}
                
                # Handle images
                for idx, cam_name in enumerate(['head_camera', 'left_camera', 'right_camera']):
                    if cam_name in episode_data.images and episode_data.images[cam_name] is not None:
                        # Real camera data
                        obs[f'enc_cam_{idx}'] = processor._resize_and_encode(episode_data.images[cam_name][t])
                    else:
                        # Use dummy image for missing cameras
                        if idx == 1 and dummy_left is not None:
                            obs[f'enc_cam_{idx}'] = dummy_left
                        elif idx == 2 and dummy_right is not None:
                            obs[f'enc_cam_{idx}'] = dummy_right
                        else:
                            obs[f'enc_cam_{idx}'] = processor.create_dummy_image()
                
                # Handle state with dimension padding (data NOT normalized yet)
                state = processor.pad_to_target_dim(episode_data.states[t], target_dim)
                obs['state'] = state  # Use direct reference, not copy
                obs['instruction_id'] = task_id
                obs['action_dim'] = episode_data.states[t].shape[0]
                
                # Handle action with dimension padding (data NOT normalized yet)
                action = processor.pad_to_target_dim(episode_data.actions[t], target_dim)
                
                # Reward is always 0 for imitation learning
                reward = 0.0
                
                trajectory.append((obs, action, reward))  # Use direct reference for action
                
            except Exception as e:
                print(f"Error processing timestep {t}: {e}")
                continue
        
        return trajectory if trajectory else None

    def _save_all_data(self, output_path, all_trajs):
        """Save trajectory data (normalization handled at training time)"""
        self.display.print_saving_data()

        # Save trajectory data
        with open(os.path.join(output_path, 'buf.pkl'), 'wb') as f:
            pkl.dump(all_trajs, f)

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Unified Data Converter for Robotics Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python unified_converter.py --config robot_only.yaml
        python unified_converter.py --config mixed_data.yaml --output /custom/path
        python unified_converter.py --config config.yaml --verbose
        """
    )
    
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--output', '-o', type=str,
                       help='Override output directory path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Create and run converter
        converter = UnifiedConverter(args.config, args.output)
        converter.process_all_data()
        
    except Exception as e:
        try:
            from rich.console import Console
            console = Console()
            console.print(f"❌ [bold red]Error:[/bold red] {e}")
        except ImportError:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()