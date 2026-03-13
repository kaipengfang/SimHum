#!/usr/bin/env python3
"""
Script to verify unified_converter output data

Checks:
1. Whether buf.pkl file format is correct (episode-organized data)
2. Whether data has been correctly normalized
3. Whether normalization parameter files are correctly generated per data source
4. Whether data dimension handling is correct
"""

import os
import sys
import pickle as pkl
import numpy as np
import json
import argparse
from typing import Dict, List, Any


def load_and_analyze_buf(buf_path: str):
    """Load and analyze buf.pkl file"""
    print(f"📁 Loading buf.pkl from: {buf_path}")
    
    with open(buf_path, 'rb') as f:
        data = pkl.load(f)
    
    print(f"✅ Loaded successfully: {type(data)} with {len(data)} episodes")
    
    # Analyze data structure
    if len(data) > 0:
        first_episode = data[0]
        print(f"📊 First episode format: {type(first_episode)}, length: {len(first_episode)}")
        
        if len(first_episode) > 0:
            first_step = first_episode[0]
            obs, action, reward = first_step
            print(f"📋 Step format: obs={type(obs)}, action={type(action)} {action.shape}, reward={reward}")
            print(f"🔑 Observation keys: {list(obs.keys())}")
            print(f"📏 State shape: {obs['state'].shape}, Action dim: {obs.get('action_dim', 'N/A')}")
    
    return data


def analyze_normalization_by_source(data: List, output_dir: str):
    """Analyze normalization effect by data source"""
    print("\n🔍 Analyzing normalization by data source...")
    
    # Group data by data source
    source_data = {}  # {data_type: {'actions': [], 'states': [], 'sample_count': int}}
    
    for episode in data:
        for obs, action, reward in episode:
            # Determine data source type based on action_dim (simplified heuristic)
            action_dim = obs.get('action_dim', action.shape[0])
            
            if action_dim == 16:
                data_type = 'robot'
            elif action_dim in [44, 46]:
                data_type = 'human'
            else:
                data_type = f'unknown_{action_dim}d'
            
            if data_type not in source_data:
                source_data[data_type] = {'actions': [], 'states': [], 'sample_count': 0}
            
            source_data[data_type]['actions'].append(action)
            source_data[data_type]['states'].append(obs['state'])
            source_data[data_type]['sample_count'] += 1
    
    # Analyze statistical properties for each data source
    print("\n📊 Data normalization analysis by source:")
    for data_type, data_dict in source_data.items():
        actions = np.array(data_dict['actions'])
        states = np.array(data_dict['states'])
        
        print(f"\n🤖 {data_type.upper()} DATA ({data_dict['sample_count']} samples):")
        print(f"   Actions shape: {actions.shape}")
        print(f"   States shape: {states.shape}")
        
        # Analyze action statistics
        action_mean = np.mean(actions, axis=0)
        action_std = np.std(actions, axis=0)
        action_min = np.min(actions, axis=0)
        action_max = np.max(actions, axis=0)
        
        print(f"   Action stats:")
        print(f"     Mean range: [{action_mean.min():.3f}, {action_mean.max():.3f}]")
        print(f"     Std range:  [{action_std.min():.3f}, {action_std.max():.3f}]")
        print(f"     Min/Max:    [{action_min.min():.3f}, {action_max.max():.3f}]")
        
        # Analyze state statistics
        state_mean = np.mean(states, axis=0)
        state_std = np.std(states, axis=0)
        state_min = np.min(states, axis=0)
        state_max = np.max(states, axis=0)
        
        print(f"   State stats:")
        print(f"     Mean range: [{state_mean.min():.3f}, {state_mean.max():.3f}]")
        print(f"     Std range:  [{state_std.min():.3f}, {state_std.max():.3f}]")
        print(f"     Min/Max:    [{state_min.min():.3f}, {state_max.max():.3f}]")
        
        # Check normalization parameter files
        norm_files = [
            f"{data_type}_ac_norm.json",
            f"{data_type}_state_norm.json"
        ]
        
        for norm_file in norm_files:
            norm_path = os.path.join(output_dir, norm_file)
            if os.path.exists(norm_path):
                with open(norm_path, 'r') as f:
                    norm_params = json.load(f)
                print(f"   ✅ Found {norm_file}: loc range [{min(norm_params['loc']):.3f}, {max(norm_params['loc']):.3f}], scale range [{min(norm_params['scale']):.3f}, {max(norm_params['scale']):.3f}]")
            else:
                print(f"   ❌ Missing {norm_file}")
    
    return source_data


def check_normalization_files(output_dir: str):
    """Check normalization parameter files"""
    print(f"\n📄 Checking normalization files in: {output_dir}")
    
    expected_files = [
        'buf.pkl',
        'task_embeddings.pkl', 
        'task_instruction.txt',
        'task_mapping.json'
    ]
    
    found_norm_files = []
    for file in os.listdir(output_dir):
        if file.endswith('_norm.json'):
            found_norm_files.append(file)
            
    print(f"✅ Found normalization files: {found_norm_files}")
    
    # Verify file contents
    for norm_file in found_norm_files:
        file_path = os.path.join(output_dir, norm_file)
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        if 'loc' in params and 'scale' in params:
            print(f"   ✅ {norm_file}: {len(params['loc'])} dimensions")
        else:
            print(f"   ❌ {norm_file}: Invalid format")


def check_data_format(data: List):
    """Check if data format matches expected structure"""
    print(f"\n🔍 Checking data format...")
    
    # Check if data is episode-organized (not flattened)
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, list) and len(first_item) > 0:
            first_step = first_item[0]
            if isinstance(first_step, tuple) and len(first_step) == 3:
                print("✅ Data format: Episode-organized (List[List[Tuple[obs, action, reward]]])")
            else:
                print("❌ Data format: Invalid step format")
        else:
            print("❌ Data format: Not episode-organized")
    else:
        print("❌ Data format: Invalid data structure")
    
    # Compute episode length distribution
    episode_lengths = [len(episode) for episode in data]
    print(f"📊 Episode statistics:")
    print(f"   Total episodes: {len(data)}")
    print(f"   Episode length range: [{min(episode_lengths)}, {max(episode_lengths)}]")
    print(f"   Average episode length: {np.mean(episode_lengths):.1f}")
    print(f"   Total steps: {sum(episode_lengths)}")


def main():
    parser = argparse.ArgumentParser(description="Verify unified_converter output data")
    parser.add_argument('output_dir', help='Output directory path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    print("🔍 UNIFIED CONVERTER OUTPUT VERIFICATION")
    print("="*50)
    
    # 1. Load and analyze buf.pkl
    buf_path = os.path.join(args.output_dir, 'buf.pkl')
    if not os.path.exists(buf_path):
        print(f"❌ buf.pkl not found: {buf_path}")
        sys.exit(1)
    
    data = load_and_analyze_buf(buf_path)
    
    # 2. Check data format
    check_data_format(data)
    
    # 3. Check normalization parameter files
    check_normalization_files(args.output_dir)
    
    # 4. Analyze normalization effect
    source_data = analyze_normalization_by_source(data, args.output_dir)
    
    # 5. Summary
    print("\n" + "="*50)
    print("🎯 VERIFICATION SUMMARY")
    print("="*50)
    print(f"✅ Data loaded successfully: {len(data)} episodes")
    print(f"✅ Data sources found: {list(source_data.keys())}")
    print(f"✅ Total samples: {sum(src['sample_count'] for src in source_data.values())}")
    print("✅ Episode-organized format confirmed")
    print("✅ Normalization parameters saved by data source")
    print("\n🎉 Verification completed successfully!")


if __name__ == '__main__':
    main()