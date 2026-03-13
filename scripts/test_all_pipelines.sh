#!/bin/bash
# Quick pipeline test: run each training config for 2 iterations
# Usage: ./scripts/test_all_pipelines.sh
# This script verifies all 7 training configurations can start and produce loss.

source "$(dirname "$0")/utils.sh"

DATA=${SIMHUM_DATA_DIR:?Set SIMHUM_DATA_DIR to processed data directory}
CKPT=${SIMHUM_CKPT_DIR:?Set SIMHUM_CKPT_DIR to checkpoint directory}

# Override: only 2 iterations, no wandb, minimal eval
OVERRIDE="max_iterations=2 eval_freq=999999 save_freq=1 wandb.mode=disabled"

gpu_id=$(find_best_gpu)
echo "Using GPU: $gpu_id"
echo ""

PASS=0
FAIL=0
RESULTS=""

run_test() {
    local name=$1
    shift
    local cmd="$@"

    print_header "TEST: $name"
    echo "Command: CUDA_VISIBLE_DEVICES=$gpu_id $cmd $OVERRIDE"
    echo ""

    output=$(CUDA_VISIBLE_DEVICES=$gpu_id $cmd $OVERRIDE 2>&1)
    exit_code=$?

    # Extract train_loss lines
    loss_lines=$(echo "$output" | grep -E "train_loss|train/loss" | tail -3)
    error_lines=$(echo "$output" | grep -iE "error|exception|traceback" | tail -5)

    if [ $exit_code -eq 0 ] && [ -n "$loss_lines" ]; then
        echo -e "\033[1;32mPASS\033[0m"
        echo "$loss_lines"
        PASS=$((PASS + 1))
        RESULTS="${RESULTS}\nPASS: $name"
    else
        echo -e "\033[1;31mFAIL (exit=$exit_code)\033[0m"
        if [ -n "$error_lines" ]; then
            echo "$error_lines"
        fi
        # Save full output for debugging
        echo "$output" > "/tmp/test_fail_${name}.log"
        echo "Full log: /tmp/test_fail_${name}.log"
        FAIL=$((FAIL + 1))
        RESULTS="${RESULTS}\nFAIL: $name"
    fi
    echo ""
}

# =============================================
# PT-1: Sim pre-training
# =============================================
run_test "PT1_sim" python finetune.py \
    exp_name=test_pt1_sim \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/robotwin_random_click_bell_500/buf.pkl

# =============================================
# PT-2: Human pre-training
# =============================================
run_test "PT2_human" python finetune.py \
    exp_name=test_pt2_human \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_44_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/human_click_bell_500/buf.pkl \
    task.train_buffer.use_relative_action=true

# =============================================
# PT-3: SimHum pre-training
# =============================================
run_test "PT3_simhum" python finetune.py \
    exp_name=test_pt3_simhum \
    agent=diffusion_dual \
    agent/features=resnet_gn \
    task=mixed_44_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/hybrid_click_bell_SH1000/buf.pkl \
    human_ratio=0.5

# =============================================
# PT-4: Robot-only
# =============================================
run_test "PT4_real" python finetune.py \
    exp_name=test_pt4_real \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/agilex_click_bell_80/buf.pkl

# =============================================
# FT-1: Sim -> Robot fine-tune
# =============================================
run_test "FT1_sim_ft" python finetune.py \
    mode=finetune \
    checkpoint_path=$CKPT/S500_click_bell/wandb_S500_click_bell_task:robotwin_random_click_bell_500_end_effector_16_3_resnet_gn_2025-12-04_11-30-40/S500_click_bell_final.ckpt \
    robot_only_mode=true \
    exp_name=test_ft1_sim \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/agilex_click_bell_80/buf.pkl \
    lr=0.00005

# =============================================
# FT-2: Human -> Robot fine-tune
# =============================================
run_test "FT2_human_ft" python finetune.py \
    mode=finetune \
    checkpoint_path=$CKPT/H500_place_bread_basket/wandb_H500_place_bread_basket_task:human_place_bread_basket_500_end_effector_44_3_resnet_gn_2025-12-05_21-35-09/H500_place_bread_basket_final.ckpt \
    robot_only_mode=true \
    exp_name=test_ft2_human \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/agilex_click_bell_80/buf.pkl \
    lr=0.00005

# =============================================
# FT-3: SimHum -> Robot fine-tune
# =============================================
run_test "FT3_simhum_ft" python finetune.py \
    mode=finetune \
    checkpoint_path=$CKPT/SH1000_click_bell/wandb_SH1000_click_bell_dual_task:hybrid_click_bell_SH1000_mixed_44_3_resnet_gn_2025-12-07_20-20-37/SH1000_click_bell_final.ckpt \
    robot_only_mode=true \
    exp_name=test_ft3_simhum \
    agent=diffusion_dual \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/agilex_click_bell_80/buf.pkl \
    lr=0.00005 \
    use_human_adaptor=true \

# =============================================
# Summary
# =============================================
echo ""
print_header "TEST SUMMARY"
echo -e "$RESULTS"
echo ""
echo "PASS: $PASS / $((PASS + FAIL))"
if [ $FAIL -gt 0 ]; then
    echo -e "\033[1;31m$FAIL test(s) FAILED\033[0m"
else
    echo -e "\033[1;32mAll tests PASSED\033[0m"
fi
