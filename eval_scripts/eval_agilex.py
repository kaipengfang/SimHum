"""
Real-time evaluation server for Agilex robot.

Starts a Flask HTTP server that loads a trained SimHum policy and serves
action predictions. The robot client sends camera images + EEF state,
and receives predicted actions in response.

Endpoints:
  GET  /connect  - Health check
  GET  /meta     - Model metadata (camera keys, prediction horizon)
  POST /reset    - Reset policy state (clear action history)
  POST /control  - Predict action(s) from observation
"""
import argparse
import os
import sys
import threading

# Add project root to path so that `simhum` package can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from flask import Flask, request, jsonify

from utils import LogTool, Tools
from policy import Policy

# Enable TensorCore acceleration for faster matmul on supported GPUs
torch.set_float32_matmul_precision('high')

# ---------- Flask & Globals ----------
app = Flask(__name__)
g_lock = threading.Lock()  # Serializes policy access across requests
g_policy = None
g_batch_mode = False  # Set via --batch flag


# ---------- Flask Routes ----------

@app.route("/connect", methods=["GET"])
def connect():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/meta", methods=["GET"])
def meta():
    """Return model metadata for client configuration."""
    if g_policy is None:
        return jsonify({"error": "model not loaded"}), 503
    return jsonify({
        "img_keys": g_policy.img_keys,
        "pred_horizon": g_policy.pred_horizon,
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Reset policy state. Call between episodes to clear action history."""
    if g_policy is None:
        return jsonify({"error": "model not loaded"}), 503
    with g_lock:
        g_policy.reset()
        LogTool.success("HTTP", "Policy state has been reset")
    return jsonify({"status": "reset"})


@app.route("/control", methods=["POST"])
def control():
    """Predict action(s) from observation.

    Request body:
      {
        "images": {"head_cam": {...}, "left_cam": {...}, "right_cam": {...}},
        "eef_state": [16D]  OR  "eef_states": {"left_arm": [8D], "right_arm": [8D]}
      }

    Response (controlled by server --batch flag):
      default    -> {"action": [16D float array]}
      --batch    -> {"actions": [[16D], [16D], ...]}  (pred_horizon actions)
    """
    if g_policy is None:
        return jsonify({"error": "model not loaded"}), 503

    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "invalid json"}), 401

    # Parse images
    images_payload = payload.get("images", {})
    if not isinstance(images_payload, dict):
        return jsonify({"error": "images must be a dict"}), 402

    images = {}
    for k in g_policy.img_keys:
        img_obj = images_payload.get(k)
        img = Tools.decode_image(img_obj)
        if img is None:
            return jsonify({"error": f"missing or invalid image for key: {k}"}), 403
        images[k] = img

    # Parse robot state
    state = Tools.extract_eef_state(payload)
    if state is None:
        return jsonify({"error": "missing eef_state/eef_states"}), 404

    obs = {"images": images, "state": state}

    with g_lock:
        if g_batch_mode:
            actions = g_policy.forward_batch(obs)
            return jsonify({"actions": [ac.tolist() for ac in actions]})
        else:
            action = g_policy.forward(obs)
            return jsonify({"action": action.tolist()})


# ---------- Startup ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Agilex real-time evaluation server")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--pred_horizon", default=40, type=int,
                        help="Number of future actions to predict per inference")
    parser.add_argument("--hz", default=40.0, type=float,
                        help="Target inference rate in Hz")
    parser.add_argument("--gamma", default=0.85, type=float,
                        help="Temporal ensemble weight (higher = less smoothing)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch", action="store_true",
                        help="Return full action sequence per request instead of single action")
    return parser.parse_args()


def main():
    global g_policy, g_batch_mode
    args = parse_args()
    g_batch_mode = args.batch
    args.period = 1.0 / args.hz

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = os.path.basename(args.checkpoint)
    g_policy = Policy(agent_path, model_name, args)

    LogTool.section("SERVER STARTUP")
    LogTool.success("SERVER", "Real robot inference service ready")
    LogTool.info("SERVER", f"Address: http://{args.host}:{args.port}")
    LogTool.info("SERVER", f"Device: {args.device}")
    LogTool.info("SERVER", f"Inference rate: {args.hz} Hz")
    print()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
