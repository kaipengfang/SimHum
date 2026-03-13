"""
Test client for the Agilex evaluation server (eval_agilex.py).

Reads episode data from an HDF5 file, sends observations to the inference
server frame by frame, collects predicted actions, and generates a comparison
plot of predicted actions vs ground truth states with per-dimension error metrics.
"""
import argparse
import base64

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests

# Maps HDF5 dataset keys to server-expected camera names.
# Must stay in sync with obs_config.yaml used by the server.
CAMERA_MAPPING = {
    "observations/images/cam_high": "head_cam",
    "observations/images/cam_left_wrist": "left_cam",
    "observations/images/cam_right_wrist": "right_cam",
}


# ---------- Encoding & HTTP Helpers ----------

def encode_image(img_arr):
    """Encode a numpy BGR image to a base64 jpeg dict for the server."""
    _, img_encoded = cv2.imencode('.jpg', img_arr)
    return {
        "format": "jpeg",
        "data": base64.b64encode(img_encoded.tobytes()).decode('utf-8'),
        "shape": img_arr.shape,
    }


def server_request(method, url, **kwargs):
    """Send an HTTP request with unified error handling.

    Returns (success: bool, response_or_None).
    """
    try:
        resp = method(url, **kwargs)
        if resp.status_code == 200:
            return True, resp
        print(f"  Server error {resp.status_code}: {resp.text}")
        return False, None
    except requests.exceptions.RequestException as e:
        print(f"  Network error: {e}")
        return False, None


def reset_server(server_url):
    """POST /reset to clear server policy state between episodes."""
    print("Resetting server state...")
    ok, _ = server_request(requests.post, f"{server_url}/reset", timeout=10)
    print("Server reset successful" if ok else "Server reset failed")
    return ok


def check_server_connection(server_url):
    """GET /connect to verify the server is reachable."""
    ok, _ = server_request(requests.get, f"{server_url}/connect", timeout=5)
    print("Server connection OK" if ok else "Server connection failed")
    return ok


# ---------- Data Loading ----------

def load_episode_states(h5_file, episode_length):
    """Load all ground truth dual-arm EEF states from an HDF5 episode."""
    states = []
    for i in range(episode_length):
        left = h5_file['eef_pose/puppet_eef_pose/left_eef_4D'][i]
        right = h5_file['eef_pose/puppet_eef_pose/right_eef_4D'][i]
        states.append(np.concatenate([left, right]))
    return states


def build_payload(h5_file, frame_idx):
    """Build a /control request payload from a single HDF5 frame.

    Encodes all camera images and extracts dual-arm EEF states.
    """
    images = {}
    for h5_key, server_key in CAMERA_MAPPING.items():
        raw = h5_file[h5_key][frame_idx]
        decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        images[server_key] = encode_image(decoded)

    left_arm = h5_file['eef_pose/puppet_eef_pose/left_eef_4D'][frame_idx]
    right_arm = h5_file['eef_pose/puppet_eef_pose/right_eef_4D'][frame_idx]

    return {
        "images": images,
        "eef_states": {
            "left_arm": left_arm.tolist(),
            "right_arm": right_arm.tolist(),
        },
    }


# ---------- Inference Loop ----------

def run_episode(h5_path, server_url, max_frames):
    """Run the inference loop over one episode.

    Sends observations to the server only when the local action buffer is
    exhausted, reusing cached actions otherwise. This reduces the number
    of network round-trips compared to per-frame requests.

    Returns (raw_actions, real_states) for downstream comparison.
    """
    with h5py.File(h5_path, "r") as f:
        episode_length = min(
            len(f['eef_pose/puppet_eef_pose/left_eef_4D']), max_frames,
        )
        print(f"Episode length: {episode_length}")

        real_states = load_episode_states(f, episode_length)
        raw_actions = []
        request_count = 0

        for frame in range(episode_length):
            actions_available = len(raw_actions) - frame

            if actions_available <= 0:
                # Action buffer exhausted, request new prediction from server
                print(f"Frame {frame+1}: Sending request...")
                payload = build_payload(f, frame)
                request_count += 1

                ok, resp = server_request(
                    requests.post, f"{server_url}/control",
                    json=payload, timeout=30,
                )
                if not ok:
                    break

                result = resp.json()
                # Compatible with two response formats:
                # - "action": single action (current eval_agilex.py format)
                # - "actions": batch action sequence (reserved for future batch inference server)
                actions = result.get('actions') or []
                action = result.get('action')
                if actions:
                    raw_actions.extend(actions)
                elif action is not None:
                    raw_actions.append(action)
                else:
                    print("  Warning: Server did not return any action")

                print(f"  Request #{request_count}: buffer size {len(raw_actions)}")
            else:
                print(f"Frame {frame+1}: Using cached action (remaining {actions_available})")

    print(f"\nEpisode complete: {request_count} requests, {len(raw_actions)} actions collected")
    if request_count > 0:
        print(f"Efficiency: {episode_length // request_count:.1f}x vs per-frame")

    return raw_actions, real_states


# ---------- Visualization ----------

def plot_comparison(raw_actions, real_states, output_path):
    """Plot predicted actions vs ground truth states with error metrics.

    Generates one subplot per action dimension (16D for dual-arm EEF).
    Each subplot shows predicted (blue solid) vs ground truth (red dashed)
    with MAE and MSE annotations.
    """
    min_len = min(len(raw_actions), len(real_states))
    actions_arr = np.array(raw_actions[:min_len])
    states_arr = np.array(real_states[:min_len])
    num_dims = actions_arr.shape[1]

    print(f"Plotting: actions {actions_arr.shape}, states {states_arr.shape}")

    fig, axes = plt.subplots(num_dims, 1, figsize=(15, 3 * num_dims))
    if num_dims == 1:
        axes = [axes]

    for i in range(num_dims):
        ax = axes[i]
        ax.plot(actions_arr[:, i], 'b-', label='Predicted', linewidth=2, alpha=0.8)
        if i < states_arr.shape[1]:
            ax.plot(states_arr[:, i], 'r--', label='Ground Truth', linewidth=2, alpha=0.8)
            mae = np.mean(np.abs(actions_arr[:, i] - states_arr[:, i]))
            mse = np.mean((actions_arr[:, i] - states_arr[:, i]) ** 2)
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nMSE: {mse:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)
        ax.set_title(f'Dimension {i}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved: {output_path}")

    # Print overall error summary
    if actions_arr.shape[1] == states_arr.shape[1]:
        overall_mae = np.mean(np.abs(actions_arr - states_arr))
        overall_rmse = np.sqrt(np.mean((actions_arr - states_arr) ** 2))
        print(f"Overall MAE: {overall_mae:.6f}, RMSE: {overall_rmse:.6f}")


# ---------- Entry Point ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Test client for Agilex eval server")
    parser.add_argument("--episode", "-e", required=True,
                        help="Path to episode HDF5 file")
    parser.add_argument("--server", "-s", default="http://127.0.0.1:5000",
                        help="Server URL")
    parser.add_argument("--max-frames", "-m", type=int, default=700,
                        help="Max frames to process per episode")
    parser.add_argument("--output", "-o", default="action_vs_state_comparison.png",
                        help="Output chart file path")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Episode: {args.episode}")
    print(f"Server:  {args.server}")
    print(f"Max frames: {args.max_frames}")
    print("-" * 60)

    if not check_server_connection(args.server):
        print("Cannot connect to server")
        return

    # Reset before and after episode to ensure clean state
    reset_server(args.server)

    try:
        raw_actions, real_states = run_episode(args.episode, args.server, args.max_frames)
    finally:
        reset_server(args.server)

    if raw_actions and real_states:
        plot_comparison(raw_actions, real_states, args.output)
    else:
        print("Not enough data for comparison")


if __name__ == "__main__":
    main()
