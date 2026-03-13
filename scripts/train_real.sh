#!/bin/bash
# Train Real-only baseline (robot data only, from scratch)
# Usage: ./scripts/train_real.sh <buffer_path> [exp_name] [wandb]

source "$(dirname "$0")/utils.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <buffer_path> [exp_name] [wandb]"
    echo "  buffer_path  Path to robot replay buffer (.pkl)"
    echo "  exp_name     Experiment name (default: real_only)"
    echo "  wandb        Set 'true' to enable W&B logging (default: false/disabled)"
    exit 1
fi

buffer_path=$1
exp_name=${2:-real_only}
wandb_debug=$(process_wandb "${3:-false}")

validate_file "$buffer_path" "Buffer file"

gpu_id=$(find_best_gpu)
gpu_info=$(get_gpu_info $gpu_id)

print_header "Real-only Training (Robot data)"
print_config "Buffer path" "$buffer_path"
print_config "Experiment" "$exp_name"
print_config "GPU" "$gpu_id ($gpu_info)"
print_config "WandB" "$([ "$wandb_debug" == "True" ] && echo 'disabled' || echo 'enabled')"
print_config "Max iterations" "60000"
print_config "LR" "0.0005"
print_header ""

if [[ "$wandb_debug" == "True" ]]; then
    print_warn "WandB logging is disabled!"
    echo ""
fi

print_launch "Starting Real-only training..."
echo ""

CUDA_VISIBLE_DEVICES=$gpu_id python finetune.py \
    exp_name=$exp_name \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$buffer_path \
    wandb.debug=$wandb_debug \
    wandb.name=${exp_name}_real_only \
    max_iterations=60000 \
    eval_freq=1000 \
    save_freq=20000
