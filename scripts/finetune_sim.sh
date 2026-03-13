#!/bin/bash
# Fine-tune: Sim-pretrained model -> Robot data
# Usage: ./scripts/finetune_sim.sh <buffer_path> <checkpoint_path> [exp_name] [wandb]

source "$(dirname "$0")/utils.sh"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <buffer_path> <checkpoint_path> [exp_name] [wandb]"
    echo "  buffer_path      Path to robot replay buffer (.pkl)"
    echo "  checkpoint_path  Path to Sim-pretrained checkpoint (.ckpt)"
    echo "  exp_name         Experiment name (default: ft_sim)"
    echo "  wandb            Set 'true' to enable W&B logging (default: false/disabled)"
    exit 1
fi

buffer_path=$1
checkpoint_path=$2
exp_name=${3:-ft_sim}
wandb_debug=$(process_wandb "${4:-false}")

validate_file "$buffer_path" "Buffer file"
validate_file "$checkpoint_path" "Checkpoint file"

gpu_id=$(find_best_gpu)
gpu_info=$(get_gpu_info $gpu_id)

print_header "Sim -> Robot Fine-tuning"
print_config "Buffer path" "$buffer_path"
print_config "Checkpoint" "$checkpoint_path"
print_config "Experiment" "$exp_name"
print_config "GPU" "$gpu_id ($gpu_info)"
print_config "WandB" "$([ "$wandb_debug" == "True" ] && echo 'disabled' || echo 'enabled')"
print_config "Max iterations" "60000"
print_config "LR" "0.00005"
print_config "Robot-only mode" "true"
print_header ""

if [[ "$wandb_debug" == "True" ]]; then
    print_warn "WandB logging is disabled!"
    echo ""
fi

print_launch "Starting Sim -> Robot fine-tuning..."
echo ""

CUDA_VISIBLE_DEVICES=$gpu_id python finetune.py \
    mode=finetune \
    checkpoint_path=$checkpoint_path \
    robot_only_mode=true \
    exp_name=$exp_name \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$buffer_path \
    lr=0.00005 \
    wandb.debug=$wandb_debug \
    wandb.name=${exp_name}_sim_ft \
    max_iterations=60000 \
    eval_freq=1000 \
    save_freq=20000
