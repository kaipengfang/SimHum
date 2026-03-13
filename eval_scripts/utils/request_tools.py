"""
HTTP request parsing and image processing utilities.

Provides helper functions shared by the Flask evaluation server
(eval_agilex.py) for decoding incoming image payloads, extracting
robot state vectors, and resizing images to match training data format.
"""
import base64

import cv2
import numpy as np

# Target image size, must match the resolution used during training
IMGSIZE = (224, 224)


class Tools:
    """Utility class for request payload parsing and image preprocessing."""

    @staticmethod
    def decode_image(img_dict):
        """Decode a base64-encoded image dict to a BGR numpy array.

        Supported formats:
          - {"format": "jpeg", "data": "<base64>"}
          - {"format": "raw", "shape": [H, W, C], "data": "<base64>"}
        """
        if not isinstance(img_dict, dict):
            return None
        fmt = img_dict.get("format", "jpeg")
        data_b64 = img_dict.get("data", "")
        if not data_b64:
            return None
        arr = np.frombuffer(base64.b64decode(data_b64), dtype=np.uint8)
        if fmt == "jpeg":
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            shape = img_dict.get("shape", [])
            if not (isinstance(shape, (list, tuple)) and len(shape) == 3):
                return None
            img = np.frombuffer(arr, dtype=np.uint8).reshape(shape)
        return img

    @staticmethod
    def extract_eef_state(payload):
        """Extract 16D dual-arm EEF state vector from request payload.

        Supports two field formats (both must contain EEF data, not joint angles):
          - "eef_state": flat 16D array [left_7D + gripper, right_7D + gripper]
          - "eef_states": {"left_arm": [8D], "right_arm": [8D]} concatenated
        """
        if "eef_state" in payload:
            return np.asarray(payload.get("eef_state", {}), dtype=np.float32)
        elif "eef_states" in payload:
            eef = payload.get("eef_states", {})
            left = eef.get("left_arm", [])
            right = eef.get("right_arm", [])
            return np.asarray(left + right, dtype=np.float32)
        else:
            return None

    @staticmethod
    def resize_like_dataset(bgr_image, size=IMGSIZE):
        """Resize image to match training data preprocessing.

        Mirrors the logic in dataset/process_data._resize_and_encode:
        1) Scale proportionally to fit target size
        2) Pad shorter dimension with black borders
        """
        resize_height, resize_width = size
        frame = Tools._decode_to_frame(bgr_image, resize_height, resize_width)

        original_height, original_width = frame.shape[:2]
        scale_width = resize_width / original_width
        new_height_if_scale_by_width = int(original_height * scale_width)

        if new_height_if_scale_by_width <= resize_height:
            # Scale by width, pad height if needed
            resized = cv2.resize(frame, (resize_width, new_height_if_scale_by_width),
                                 interpolation=cv2.INTER_AREA)
            if new_height_if_scale_by_width < resize_height:
                pad_top = (resize_height - new_height_if_scale_by_width) // 2
                pad_bottom = resize_height - new_height_if_scale_by_width - pad_top
                resized = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0,
                                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            # Scale by height, pad width if needed
            new_width = int(original_width * (resize_height / original_height))
            resized = cv2.resize(frame, (new_width, resize_height),
                                 interpolation=cv2.INTER_AREA)
            if new_width < resize_width:
                pad_left = (resize_width - new_width) // 2
                pad_right = resize_width - new_width - pad_left
                resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right,
                                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return resized

    @staticmethod
    def _decode_to_frame(bgr_image, resize_height, resize_width):
        """Decode various input formats into a BGR numpy array.

        Handles: 0D scalar array, 1D encoded bytes, 3D decoded array,
        and raw bytes. Falls back to a black dummy image on failure.
        """
        frame = None
        if isinstance(bgr_image, np.ndarray):
            if bgr_image.ndim == 0:
                frame = cv2.imdecode(
                    np.frombuffer(bgr_image.tobytes(), np.uint8), cv2.IMREAD_COLOR)
            elif bgr_image.ndim == 1:
                frame = cv2.imdecode(bgr_image, cv2.IMREAD_COLOR)
            elif bgr_image.ndim == 3:
                frame = bgr_image
        else:
            try:
                frame = cv2.imdecode(
                    np.frombuffer(bgr_image, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                pass

        if frame is None:
            print("Warning: Failed to decode image, creating dummy image")
            frame = np.zeros((resize_height, resize_width, 3), dtype=np.uint8)
        return frame
