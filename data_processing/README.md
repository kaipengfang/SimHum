# Data Processing

Converts raw HDF5 episodes into replay buffer (`buf.pkl`) for training.

## Quick Start

```bash
python data_processing/unified_converter.py --config data_processing/configs/<config>.yaml
```

**Available configs:**

| Config | Description |
|--------|-------------|
| `config_sim_robot.yaml` | Simulated robot data ([RobotWin 2.0](https://robotwin-platform.github.io/)) |
| `config_agilex_robot.yaml` | Real robot data (AgileX COBOT Magic) |
| `config_human.yaml` | Human hand demonstrations |
| `config_hybrid.yaml` | Mixed sim + human (for SimHum co-training) |
| `_template.yaml` | Template with all options documented |

## Config Format

```yaml
data_sources:
  - type: "SimRobot"              # SimRobot | AgilexRobot | HumanCollect
    path: "/path/to/data"
    tasks: ["click_bell"]         # Omit to process all tasks
    max_episodes_per_task: 500
    align_to_human_coords: true   # SimRobot only: align to human coordinate system

  - type: "HumanCollect"
    path: "/path/to/human_data"
    tasks: ["click_bell"]
    max_episodes_per_task: 500
    human_output_format: 44       # HumanCollect only: 44 (no gripper) or 46

output:
  target_path: "/path/to/output"
  dataset_name: "my_dataset"
  normalization: "max_min"
```

**Type-specific options:**
- `SimRobot`: `align_to_human_coords` — rotate to human coordinates (recommended for mixed training)
- `AgilexRobot`: `use_qpos` — use joint positions instead of EEF pose (default: `false`)
- `HumanCollect`: `human_output_format` — `44` (no gripper) or `46` (with gripper)

## Output Format (`buf.pkl`)

```python
import pickle
with open("buf.pkl", "rb") as f:
    data = pickle.load(f)

obs, action, reward = data[0]
```

| Key | Type | Description |
|-----|------|-------------|
| `obs["state"]` | `np.ndarray` | State vector (same dim as action) |
| `obs["enc_cam_0/1/2"]` | `bytes` | JPEG-encoded images (head / left / right) |
| `obs["action_dim"]` | `int` | Original action dimension (16 or 44) |
| `action` | `np.ndarray` | Action vector |

When mixing different data sources, 16D robot actions are zero-padded to 44D. The `action_dim` field preserves the original dimension for the model's dual-path routing.

## Input Data Sources

### SimRobot (16D)

Simulated dual-arm data from [RobotWin 2.0](https://robotwin-platform.github.io/). Action: `[left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)]`.

| HDF5 Field | Shape | Description |
|------------|-------|-------------|
| `endpose/left_endpose` | `(T, 7)` | Left arm pose `[x,y,z,qw,qx,qy,qz]` |
| `endpose/right_endpose` | `(T, 7)` | Right arm pose |
| `endpose/left_gripper` | `(T,)` | Left gripper |
| `endpose/right_gripper` | `(T,)` | Right gripper |
| `observation/{head,left,right}_camera/rgb` | `(T,)` | Camera images |

### AgilexRobot (16D)

Real-world dual-arm data from AgileX COBOT Magic. Same 16D action space as SimRobot.

| HDF5 Field | Shape | Description |
|------------|-------|-------------|
| `eef_pose/puppet_eef_pose/{left,right}_eef` | `(T, 7)` | EEF pose `[x,y,z,qw,qx,qy,qz]` |
| `action` | `(T, 14)` | Joint actions (for gripper extraction) |
| `observations/images/cam_{high,left_wrist,right_wrist}` | `(T,)` | Camera images |

### HumanCollect (44D)

Human demonstrations via wrist tracking and hand keypoints. Action: `[left_eef(7) + left_fingertips(15) + right_eef(7) + right_fingertips(15)]`.

| HDF5 Field | Shape | Description |
|------------|-------|-------------|
| `action/cmd/rel_{left,right}_wrist_mat` | `(T, 4, 4)` | Wrist transformation matrix |
| `action/cmd/rel_{left,right}_hand_keypoints` | `(T, 25, 3)` | Hand keypoints (25 joints) |
| `action/cmd/head_mat` | `(T, 4, 4)` | Head transformation matrix |

Each episode has a companion `.mp4` from the head-mounted camera, frame-aligned with HDF5.
