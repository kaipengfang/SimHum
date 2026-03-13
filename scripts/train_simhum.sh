#!/bin/bash
# Train SimHum: Sim-and-Human co-training (dual-path method, our approach)
# Usage: ./scripts/train_simhum.sh <human_ratio> <buffer_path> [exp_name] [wandb]

source "$(dirname "$0")/utils.sh"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <human_ratio> <buffer_path> [exp_name] [wandb]"
    echo "  human_ratio  Fraction of human data per batch (e.g., 0.5)"
    echo "  buffer_path  Path to mixed replay buffer (.pkl)"
    echo "  exp_name     Experiment name (default: simhum)"
    echo "  wandb        Set 'true' to enable W&B logging (default: false/disabled)"
    exit 1
fi

human_ratio=$1
buffer_path=$2
exp_name=${3:-simhum}
wandb_debug=$(process_wandb "${4:-false}")

validate_file "$buffer_path" "Buffer file"

gpu_id=$(find_best_gpu)
gpu_info=$(get_gpu_info $gpu_id)

print_header "SimHum Co-training (Dual-path)"
print_config "Buffer path" "$buffer_path"
print_config "Experiment" "$exp_name"
print_config "Human ratio" "$human_ratio"
print_config "GPU" "$gpu_id ($gpu_info)"
print_config "WandB" "$([ "$wandb_debug" == "True" ] && echo 'disabled' || echo 'enabled')"
print_config "Max iterations" "200000"
print_config "LR" "0.0005"
print_header ""

if [[ "$wandb_debug" == "True" ]]; then
    print_warn "WandB logging is disabled!"
    echo ""
fi

print_launch "Starting SimHum co-training..."
echo ""

CUDA_VISIBLE_DEVICES=$gpu_id python finetune.py \
    exp_name=$exp_name \
    agent=diffusion_dual \
    agent/features=resnet_gn \
    task=mixed_44_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$buffer_path \
    human_ratio=$human_ratio \
    wandb.debug=$wandb_debug \
    wandb.name=${exp_name}_simhum \
    max_iterations=200000 \
    eval_freq=1000 \
    save_freq=50000
