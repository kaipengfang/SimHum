<h1 align="center">Sim-and-Human Co-training for Data-Efficient<br>and Generalizable Robotic Manipulation</h1>

<p align="center">
  <a href="https://kaipengfang.github.io/"><b>Kaipeng Fang</b></a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://github.com/weiqing11/"><b>Weiqing Liang</b></a><sup>1</sup>&nbsp;&nbsp;
  <b>Yuyang Li</b><sup>1</sup>&nbsp;&nbsp;
  <a href="https://jimzai.github.io/"><b>Ji Zhang</b></a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://ppengzeng.github.io/"><b>Pengpeng Zeng</b></a><sup>3</sup>&nbsp;&nbsp;
  <br>
  <a href="https://lianligao.github.io/"><b>Lianli Gao</b></a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://cfm.uestc.edu.cn/~shenht/"><b>Heng Tao Shen</b></a><sup>3</sup>&nbsp;&nbsp;
  <a href="https://jingkuansong.github.io/"><b>Jingkuan Song</b></a><sup>3,4</sup>
</p>

<p align="center">
  <sup>1</sup>UESTC&nbsp;&nbsp;
  <sup>2</sup>SWJTU&nbsp;&nbsp;
  <sup>3</sup>Tongji University&nbsp;&nbsp;
  <sup>4</sup>Shanghai Innovation Institute
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.19406"><img src="https://img.shields.io/badge/arXiv-2601.19406-b31b1b.svg" alt="arXiv"></a>
  <a href="https://kaipengfang.github.io/sim-and-human/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://github.com/kaipengfang/SimHum"><img src="https://img.shields.io/badge/Code-GitHub-black.svg" alt="Code"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

<p align="center">
  <img src="https://kaipengfang.github.io/sim-and-human/resources/images/Introduction.png" width="80%">
</p>

SimHum co-trains on **simulated robot trajectories** and **real-world human demonstrations** to learn manipulation policies that are both data-efficient and generalizable. It outperforms the real-only baseline by up to **40%** under the same data budget, and achieves **62.5% OOD success with only 80 real demonstrations** (a **7.1x** data efficiency gain).

---

## Installation

```bash
git clone https://github.com/kaipengfang/SimHum.git
cd SimHum
conda env create -f env.yml
conda activate dit
```

## Data Preparation

Convert raw HDF5 episodes to replay buffer format:

```bash
python data_processing/unified_converter.py --config data_processing/configs/config_sim_robot.yaml
python data_processing/unified_converter.py --config data_processing/configs/config_agilex_robot.yaml
python data_processing/unified_converter.py --config data_processing/configs/config_human.yaml
python data_processing/unified_converter.py --config data_processing/configs/config_hybrid.yaml
```

See [`data_processing/README.md`](data_processing/README.md) for data format details.

## Human Data Collection

We provide a complete data collection pipeline for capturing egocentric hand demonstrations using Meta Quest 3 and a depth camera. Launch the Web GUI, connect the VR headset, and press `Space` to start recording episodes in HDF5 format.

```bash
cd human_data_collection
bash scripts/setup_env.sh          # One-time environment setup
conda activate H_data_col
bash scripts/start_gui_web.sh      # Launch Web GUI at http://localhost:8000
```

See [`human_data_collection/README.md`](human_data_collection/README.md) for hardware setup, camera configuration, and data format details.

## Training

Training is organized into two stages. All scripts auto-select the best GPU.

**Arguments:**
- `buffer_path` — path to the replay buffer file (`.pkl`)
- `checkpoint_path` — path to the pre-trained checkpoint (`.ckpt`, fine-tuning only)
- `exp_name` — experiment name for logging (optional)
- `wandb` — set to `true` to enable [W&B](https://wandb.ai) logging (default: `false`). First-time setup: run `wandb login` and set `wandb.entity` in `experiments/finetune.yaml`

### Stage 1: Pre-training

```bash
# Sim-only
./scripts/train_sim.sh <buffer_path> [exp_name] [wandb]

# Human-only
./scripts/train_human.sh <buffer_path> [exp_name] [wandb]

# SimHum co-training (ours)
./scripts/train_simhum.sh <human_ratio> <buffer_path> [exp_name] [wandb]

# Robot-only baseline
./scripts/train_real.sh <buffer_path> [exp_name] [wandb]
```

### Stage 2: Fine-tuning on Real Robot Data

```bash
# Sim → Robot
./scripts/finetune_sim.sh <buffer_path> <checkpoint_path> [exp_name] [wandb]

# Human → Robot
./scripts/finetune_human.sh <buffer_path> <checkpoint_path> [exp_name] [wandb]

# SimHum → Robot (ours)
./scripts/finetune_simhum.sh <buffer_path> <checkpoint_path> [exp_name] [wandb]
```

### Configuration Summary

All configs use **batch size 250**, **action chunk 40**, and **3 cameras**.

| Config | Action Dim | LR | Iterations | Batch Size |
|--------|------------|-----|------------|------------|
| Sim pre-train | 16 | 5e-4 | 200K | 250 |
| Human pre-train | 44 | 5e-4 | 200K | 250 |
| SimHum pre-train | 44 | 5e-4 | 200K | 250 |
| Robot-only | 16 | 5e-4 | 60K | 250 |
| Sim → Robot FT | 16 | 5e-5 | 60K | 250 |
| Human → Robot FT | 16 | 5e-5 | 60K | 250 |
| SimHum → Robot FT | 16 | 5e-5 | 60K | 250 |

### Pipeline Overview

```
Stage 1: Pre-training                     Stage 2: Fine-tuning
┌─────────────────────┐                  ┌─────────────────────┐
│  Sim Data (500 ep)  │──► train_sim.sh ──► finetune_sim.sh ──►│                     │
│ Human Data (500 ep) │──► train_human.sh─► finetune_human.sh─►│  Real Robot Policy  │
│ Mixed Data (1000 ep)│──► train_simhum.sh► finetune_simhum.sh►│                     │
│ Robot Data (80 ep)  │──► train_real.sh ─────────────────────►│  (Baseline)         │
└─────────────────────┘                  └─────────────────────┘
```

### Hardware Requirements

- **GPU**: 1x NVIDIA RTX 4090 (24GB) or equivalent
- **RAM**: 32GB+
- **Time**: Pre-training ~40-100h, fine-tuning ~10-25h (single RTX 4090)

## Evaluation

### Real-time evaluation (server + client)

Start the inference server, then run the test client against it:

```bash
# 1. Start the evaluation server
python eval_scripts/eval_agilex.py \
    --checkpoint path/to/checkpoint.ckpt \
    --batch

# 2. Run the test client on a saved episode
python eval_scripts/test_client_agilex.py \
    --episode path/to/episode.hdf5

# If you have http_proxy set and get 403 errors, bypass proxy for localhost:
no_proxy=127.0.0.1 python eval_scripts/test_client_agilex.py \
    --episode path/to/episode.hdf5
```

### Real robot deployment

Use `test_client_agilex.py` as a reference to write your robot control code:

1. **`GET /connect`** — verify the server is running
2. **`POST /reset`** — clear policy state before each episode
3. **`POST /control`** — send `{"images": {...}, "eef_states": {...}}`, receive predicted action(s)
4. **Execute** — send the action to the robot, repeat from step 3

See `build_payload()` and `CAMERA_MAPPING` in `test_client_agilex.py` for the exact request format.

## Architecture

<p align="center">
  <img src="https://kaipengfang.github.io/sim-and-human/resources/images/model.png" width="80%">
</p>

SimHum uses a **dual-path architecture** with separate vision adaptors for simulation and human data streams. During fine-tuning, the real-world vision adaptor (learned from human demonstrations) is reused to process robot camera inputs, transferring photorealistic visual priors.

**Data dimensions:**
- Robot: 16D — `[x, y, z, qw, qx, qy, qz, gripper]` x2 arms
- Human: 44D — `[eef_pose(7), fingertips(15)]` x2 hands

## Project Structure

```
├── simhum/                    # Core library
│   ├── models/                # DiffusionTransformerAgent, DiffusionTransformerAgent_Dual, ResNet
│   ├── trainers/              # Behavior cloning trainer
│   └── replay_buffer.py       # Data loading and sampling
├── data_processing/           # HDF5 → replay buffer conversion
├── human_data_collection/     # VR-based human demonstration capture
├── experiments/               # Hydra configs (task / agent / trainer)
├── eval_scripts/              # Real-robot evaluation
├── scripts/                   # Training & fine-tuning shell scripts
└── finetune.py                # Main entry point
```

## Citation

```bibtex
@article{fang2025simhum,
  title={Sim-and-Human Co-training for Data-Efficient and Generalizable Robotic Manipulation},
  author={Fang, Kaipeng and Liang, Weiqing and Li, Yuyang and Zhang, Ji and Zeng, Pengpeng and Gao, Lianli and Shen, Heng Tao and Song, Jingkuan},
  journal={arXiv preprint arXiv:2601.19406},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built upon [dit-policy](https://github.com/SudeepDasari/dit-policy).
