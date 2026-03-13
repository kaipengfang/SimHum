"""
HDF5 Writer Module
Responsible for writing data to HDF5 files
"""
import cv2
import h5py
import numpy as np
import datetime
from pathlib import Path
from scipy.spatial.transform import Rotation
from ..quality import EpisodeDataQualityChecker


class HDF5Writer:
    """HDF5 file writer with quality checking and data processing"""
    
    def __init__(self, freq=30):
        self.freq = freq
    
    def save_to_hdf5(self, path, data_dict, description, embodiment, log_callback=None):
        """
        Save data to HDF5 file, automatically discard last 3 seconds to avoid saving end gesture
        
        Args:
            path: Save path
            data_dict: Data dictionary
            description: Task description
            embodiment: Embodiment type
            log_callback: Logging callback function
            
        Returns:
            Tuple[bool, dict]: (success, quality_result)
        """
        def log_msg(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        # Step 1: Quality check
        success, quality_result = self._perform_quality_check(data_dict, log_msg)
        if not success:
            return False, quality_result
        
        # Step 2: Process data (discard last 3 seconds, calculate gripper, etc.)
        processed_data = self._process_data(data_dict, log_msg)
        
        # Step 3: Write to HDF5
        self._write_hdf5_file(path, processed_data, description, embodiment, log_msg)
        
        # Step 4: Save preview video
        frames_to_keep = processed_data['frames_to_keep']
        self._save_preview_video(path, data_dict, frames_to_keep)
        
        log_msg("")
        log_msg("✅ Data saved successfully!")
        log_msg("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        return True, quality_result
    
    def _perform_quality_check(self, data_dict, log_msg):
        """Perform data quality check"""
        log_msg("")
        log_msg("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log_msg("🔍 Starting data quality check...")

        quality_checker = EpisodeDataQualityChecker(fps=self.freq)
        quality_result = quality_checker.check_episode_quality(data_dict)
        
        # Display check details
        self._log_quality_details(quality_result, log_msg)
        
        if not quality_result['is_valid']:
            self._log_quality_failure(quality_result, log_msg)
            return False, quality_result
        
        log_msg("")
        log_msg("✅ Data quality check passed!")
        log_msg("┌─────────────────────────────────────────────────────────")
        log_msg(f"│ ✓ All {quality_result['details']['checked_frames']} frames have good quality")
        log_msg("│ ✓ No continuous static segments found")
        log_msg("│ ✓ Hand movement is normal")
        log_msg("│ ✓ Safe to save")
        log_msg("└─────────────────────────────────────────────────────────")
        
        return True, quality_result
    
    def _log_quality_details(self, quality_result, log_msg):
        """Log quality check details"""
        log_msg(f"📊 Check details:")
        total_frames = quality_result['details'].get('total_frames', 0)
        log_msg(f"  ├─ Total frames: {total_frames} frames ({total_frames/self.freq:.1f}s)")
        
        start_frame = quality_result['details'].get('start_frame', 0)
        end_frame = quality_result['details'].get('end_frame', 0)
        start_time = start_frame / self.freq
        end_time = (end_frame-1) / self.freq if end_frame > 0 else 0
        log_msg(f"  ├─ Check range: Frame {start_frame} - {end_frame-1} ({start_time:.1f}s - {end_time:.1f}s)")
        log_msg(f"  ├─ Check note: Skip first 1s gesture + skip last 4s (including 3s to discard)")
        log_msg(f"  ├─ Valid frames: {quality_result['details'].get('checked_frames', 'N/A')} frames")
        log_msg(f"  ├─ Static threshold: 0.5 mm (all axes)")
        log_msg(f"  └─ Continuous threshold: 30 frames ({30/self.freq:.2f} seconds)")
    
    def _log_quality_failure(self, quality_result, log_msg):
        """Log quality check failure details"""
        log_msg("")
        log_msg("❌ Data quality check failed")
        log_msg("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log_msg("📋 Issue details:")
        for i, issue in enumerate(quality_result['issues'], 1):
            log_msg(f"  {i}. {issue}")

        # Display static segment details
        left_hand = quality_result['details'].get('left_hand', {})
        right_hand = quality_result['details'].get('right_hand', {})
        
        for hand_name, hand_info in [('Left', left_hand), ('Right', right_hand)]:
            static_segments = hand_info.get('static_segments', [])
            if static_segments:
                log_msg("")
                log_msg(f"🔍 {hand_name} hand - Found {len(static_segments)} static segments:")
                log_msg("┌─────────────────────────────────────────────────────────")
                for i, segment in enumerate(static_segments, 1):
                    start_frame = segment['start']
                    end_frame = segment['end']
                    duration = segment['duration']
                    start_time = start_frame/self.freq
                    end_time = end_frame/self.freq

                    log_msg(f"│")
                    log_msg(f"│ 【Static Segment {i}】")
                    log_msg(f"│   ▸ Frame range: Frame {start_frame:4d} - {end_frame:4d} (total {duration} frames)")
                    log_msg(f"│   ▸ Timestamp: {start_time:6.2f}s - {end_time:6.2f}s")
                    log_msg(f"│   ▸ Duration: {duration/self.freq:.2f} seconds")
                    log_msg(f"│   ▸ Verdict: Hand almost static")
                log_msg("└─────────────────────────────────────────────────────────")
    
    def _process_data(self, data_dict, log_msg):
        """Process data: discard frames, calculate gripper values, etc."""
        log_msg("")
        log_msg("🔄 Processing data...")
        
        # Calculate frames to discard (last 3 seconds)
        total_frames = len(data_dict['/obs/timestamp'])
        frames_to_discard = int(3 * self.freq)
        frames_to_keep = total_frames - frames_to_discard
        
        log_msg(f"  ├─ Total frames: {total_frames}")
        log_msg(f"  ├─ Frames to discard: {frames_to_discard} (last 3 seconds)")
        log_msg(f"  └─ Frames to keep: {frames_to_keep}")
        
        # Extract data to keep
        processed_data = {
            'frames_to_keep': frames_to_keep,
            'timestamp': np.array(data_dict['/obs/timestamp'][:frames_to_keep]),
            'head_mat': np.array(data_dict['/action/cmd/head_mat'][:frames_to_keep]),
            'left_wrist_mat': np.array(data_dict['/action/cmd/rel_left_wrist_mat'][:frames_to_keep]),
            'right_wrist_mat': np.array(data_dict['/action/cmd/rel_right_wrist_mat'][:frames_to_keep]),
            'left_keypoints': np.array(data_dict['/action/cmd/rel_left_hand_keypoints'][:frames_to_keep]),
            'right_keypoints': np.array(data_dict['/action/cmd/rel_right_hand_keypoints'][:frames_to_keep]),
            'head_images': data_dict['/observation/image/head'][:frames_to_keep]
        }
        
        # Calculate gripper values
        log_msg("")
        log_msg("🤏 Calculating gripper values...")
        processed_data['left_gripper'], processed_data['right_gripper'] = self._calculate_gripper_values(
            processed_data['left_keypoints'],
            processed_data['right_keypoints'],
            log_msg
        )
        
        # Calculate EEF poses
        log_msg("")
        log_msg("🎯 Calculating EEF poses...")
        processed_data['left_eef'], processed_data['right_eef'] = self._calculate_eef_poses(
            processed_data['left_wrist_mat'],
            processed_data['right_wrist_mat'],
            log_msg
        )
        
        return processed_data
    
    def _calculate_gripper_values(self, left_keypoints, right_keypoints, log_msg):
        """Calculate gripper values from keypoints"""
        quality_checker = EpisodeDataQualityChecker(fps=self.freq)
        
        # Extract fingertips (indices 4, 8, 12, 16, 20)
        fingertip_indices = [4, 8, 12, 16, 20]
        left_fingertips = left_keypoints[:, fingertip_indices, :]
        right_fingertips = right_keypoints[:, fingertip_indices, :]
        
        # Calculate gripper values
        left_gripper, right_gripper = quality_checker.calculate_gripper_dual_hands(
            left_fingertips,
            right_fingertips
        )
        
        log_msg(f"  ├─ Left gripper range: [{left_gripper.min():.3f}, {left_gripper.max():.3f}]")
        log_msg(f"  └─ Right gripper range: [{right_gripper.min():.3f}, {right_gripper.max():.3f}]")
        
        return left_gripper, right_gripper
    
    def _calculate_eef_poses(self, left_wrist_mat, right_wrist_mat, log_msg):
        """Calculate end-effector poses (position + quaternion)"""
        def mat_to_pose(mat_array):
            """Convert 4x4 matrix to 7D pose (x, y, z, qx, qy, qz, qw)"""
            poses = []
            for mat in mat_array:
                pos = mat[:3, 3]
                rot_mat = mat[:3, :3]
                quat = Rotation.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
                pose = np.concatenate([pos, quat])
                poses.append(pose)
            return np.array(poses)
        
        left_eef = mat_to_pose(left_wrist_mat)
        right_eef = mat_to_pose(right_wrist_mat)
        
        log_msg(f"  ├─ Left EEF shape: {left_eef.shape}")
        log_msg(f"  └─ Right EEF shape: {right_eef.shape}")
        
        return left_eef, right_eef
    
    def _write_hdf5_file(self, path, processed_data, description, embodiment, log_msg):
        """Write processed data to HDF5 file"""
        log_msg("")
        log_msg("💾 Writing to HDF5 file...")
        
        # Create HDF5 file
        hdf5_path = path if path.endswith('.hdf5') else path + '.hdf5'
        with h5py.File(hdf5_path, 'w') as f:
            # Metadata
            f.attrs['description'] = description
            f.attrs['embodiment'] = embodiment
            f.attrs['created_at'] = datetime.datetime.now().isoformat()
            f.attrs['freq'] = self.freq
            
            # Observation data
            obs_group = f.create_group('observation')
            obs_group.create_dataset('timestamp', data=processed_data['timestamp'])
            
            # Images (JPEG bytes -> vlen uint8, write per-frame to avoid bytes lacking dtype attribute)
            image_group = obs_group.create_group('image')
            head_images = processed_data['head_images']
            dt = h5py.vlen_dtype(np.uint8)
            ds = image_group.create_dataset('head', shape=(len(head_images),), dtype=dt)
            for i, img in enumerate(head_images):
                ds[i] = np.frombuffer(img, dtype=np.uint8) if isinstance(img, bytes) else img
            
            # Action data
            action_group = f.create_group('action')
            action_group.create_dataset('left_eef', data=processed_data['left_eef'])
            action_group.create_dataset('right_eef', data=processed_data['right_eef'])
            action_group.create_dataset('left_gripper', data=processed_data['left_gripper'])
            action_group.create_dataset('right_gripper', data=processed_data['right_gripper'])
            
            # Raw data (for reference)
            raw_group = f.create_group('raw')
            raw_group.create_dataset('head_mat', data=processed_data['head_mat'])
            raw_group.create_dataset('left_wrist_mat', data=processed_data['left_wrist_mat'])
            raw_group.create_dataset('right_wrist_mat', data=processed_data['right_wrist_mat'])
            raw_group.create_dataset('left_keypoints', data=processed_data['left_keypoints'])
            raw_group.create_dataset('right_keypoints', data=processed_data['right_keypoints'])
        
        log_msg(f"  └─ Saved to: {path}")
    
    def _save_preview_video(self, path, data_dict, frames_to_keep):
        """Save preview video from head camera images"""
        video_path = Path(path).with_suffix('.mp4')
        
        # Decode first image to get dimensions
        first_image_bytes = data_dict['/observation/image/head'][0]
        first_image = cv2.imdecode(np.frombuffer(first_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if first_image is None:
            print("Warning: Cannot decode first image, skipping preview video")
            return
        
        height, width = first_image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.freq, (width, height))
        
        # Write frames
        for i in range(frames_to_keep):
            image_bytes = data_dict['/observation/image/head'][i]
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                out.write(image)
        
        out.release()
        print(f"Preview video saved to: {video_path}")
