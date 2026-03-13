"""
Image processing utilities

Shared image processing functions for data processors.
Following DRY principle - used by multiple processors.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from .constants import IMAGE_SIZE


def resize_and_encode(bgr_image: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Resize image to target size and encode as JPEG.

    Uses width-based scaling followed by padding for height.
    Maintains aspect ratio during initial resize, then pads to exact size.
    If width-based scaling results in height exceeding target, switches to height-based scaling and pads width.

    Args:
        bgr_image: Input image as numpy array or encoded bytes
        size: Target (height, width)

    Returns:
        JPEG-encoded image as bytes array
    """
    resize_height, resize_width = size

    # Decode input image - handle different input formats
    if isinstance(bgr_image, np.ndarray):
        if bgr_image.ndim == 0:
            # Scalar array containing encoded bytes - convert to bytes first
            frame = cv2.imdecode(np.frombuffer(bgr_image.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        elif bgr_image.ndim == 1:
            # 1D array of encoded bytes
            frame = cv2.imdecode(bgr_image, cv2.IMREAD_COLOR)
        elif bgr_image.ndim == 3:
            # Already decoded image array
            frame = bgr_image
        else:
            # Unsupported format - create dummy
            frame = None
    else:
        # Non-numpy input - try to decode as bytes
        try:
            frame = cv2.imdecode(np.frombuffer(bgr_image, np.uint8), cv2.IMREAD_COLOR)
        except:
            frame = None

    # Handle decoding failures
    if frame is None:
        print(f"Warning: Failed to decode image, creating dummy image")
        frame = np.zeros((resize_height, resize_width, 3), dtype=np.uint8)

    original_height, original_width = frame.shape[:2]

    # Calculate scale based on width first
    scale_width = resize_width / original_width
    new_height_if_scale_by_width = int(original_height * scale_width)

    if new_height_if_scale_by_width <= resize_height:
        # Scale by width, then pad height
        scale = scale_width
        new_width = resize_width
        new_height = new_height_if_scale_by_width

        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Pad height if needed
        if new_height < resize_height:
            pad_top = (resize_height - new_height) // 2
            pad_bottom = resize_height - new_height - pad_top
            resized = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        # Scale by height, then pad width
        scale = resize_height / original_height
        new_width = int(original_width * scale)
        new_height = resize_height

        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Pad width if needed
        if new_width < resize_width:
            pad_left = (resize_width - new_width) // 2
            pad_right = resize_width - new_width - pad_left
            resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Encode to JPEG
    _, encoded = cv2.imencode(".jpg", resized)
    return encoded


def create_dummy_image(size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Create a dummy (all-black) image for missing camera feeds.
    
    Used when human data has fewer cameras than the standard 3-camera setup.
    
    Args:
        size: Image dimensions (height, width)
        
    Returns:
        JPEG-encoded black image
    """
    height, width = size
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy)
    return encoded


def load_video_frames(video_path: str) -> Optional[np.ndarray]:
    """
    Load all frames from MP4 video file.
    
    Args:
        video_path: Path to MP4 file
        
    Returns:
        Array of video frames (T, H, W, 3) or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        print(f"Error: No frames found in {video_path}")
        return None
    
    return np.array(frames)


def process_head_images(head_images: np.ndarray) -> np.ndarray:
    """
    Process encoded head camera images into a standardized format.
    
    Handles the decoding of stored image data from HDF5 object arrays. The images
    are typically stored as encoded bytes (JPEG/PNG) to save disk space and need
    to be decoded for processing.
    
    Args:
        head_images: Array of encoded images of shape (T,) where each element
                    is an encoded image as bytes or numpy array.
                    
    Returns:
        np.ndarray: Decoded images of shape (T, H, W, 3) ready for further processing
                   by the base class image processing pipeline.
                   
    Note:
        This method performs minimal processing since the base class resize_and_encode
        method will handle final resizing and encoding for training.
    """
    decoded_images = []
    
    for i, encoded_image in enumerate(head_images):
        try:
            # Handle different storage formats
            if isinstance(encoded_image, np.ndarray):
                if encoded_image.ndim == 1:
                    # Encoded as 1D byte array - decode using cv2
                    decoded = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
                else:
                    # Already decoded as image array
                    decoded = encoded_image
            elif isinstance(encoded_image, bytes):
                # Convert bytes to numpy array for cv2
                img_array = np.frombuffer(encoded_image, dtype=np.uint8)
                decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                print(f"Warning: Unknown image format at index {i}: {type(encoded_image)}")
                # Create dummy image as fallback
                decoded = np.zeros((480, 640, 3), dtype=np.uint8)
            
            if decoded is None:
                print(f"Warning: Failed to decode image at index {i}")
                # Create dummy image as fallback
                decoded = np.zeros((480, 640, 3), dtype=np.uint8)
            
            decoded_images.append(decoded)
            
        except Exception as e:
            print(f"Error processing image at index {i}: {e}")
            # Create dummy image as fallback
            decoded_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    return np.array(decoded_images)