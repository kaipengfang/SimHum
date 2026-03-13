#!/bin/bash
# Quick training test: Run 2 iterations to verify pipeline
# Usage: ./scripts/test_train_quick.sh <buffer_path>

buffer_path=${1:?Usage: $0 <buffer_path>}

echo "=== Quick Training Test ==="
echo "Buffer: $buffer_path"
echo "Iterations: 2 (just to verify pipeline)"
echo ""

CUDA_VISIBLE_DEVICES=4 conda run -n dit python finetune.py \
    exp_name=test_quick \
    agent=diffusion_dual \
    agent/features=resnet_gn \
    task=mixed_44_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$buffer_path \
    human_ratio=0.5 \
    max_iterations=2 \
    eval_freq=999999 \
    save_freq=1 \
    wandb.mode=disabled
