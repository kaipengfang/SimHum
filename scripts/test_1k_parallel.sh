#!/bin/bash
# Parallel 1000-step test for all 7 training pipelines
# Each experiment runs on a separate GPU with full paper config
# Usage: ./scripts/test_1k_parallel.sh

set -e
cd "$(dirname "$0")/.."

DATA=${SIMHUM_DATA_DIR:?Set SIMHUM_DATA_DIR to processed data directory}
CKPT=${SIMHUM_CKPT_DIR:?Set SIMHUM_CKPT_DIR to checkpoint directory}
LOGDIR=logs/test_1k_$(date +%Y%m%d_%H%M%S)
mkdir -p $LOGDIR

# Override: 1000 iterations, no wandb, no save overhead
OVERRIDE="max_iterations=1000 eval_freq=999999 save_freq=999999 +wandb.mode=disabled"

echo "============================================="
echo "  7-Pipeline Parallel Test (1000 steps each)"
echo "  Logs: $LOGDIR"
echo "============================================="
echo ""

# PT-1: Sim pretrain (GPU 0)
echo "[PT1] Sim pretrain → GPU 0"
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    exp_name=test_pt1_sim \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/robotwin_random_click_bell_500/buf.pkl \
    $OVERRIDE \
    > $LOGDIR/PT1_sim.log 2>&1 &
PID_PT1=$!

# PT-2: Human pretrain (GPU 1)
echo "[PT2] Human pretrain → GPU 1"
CUDA_VISIBLE_DEVICES=1 python finetune.py \
    exp_name=test_pt2_human \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_44_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/human_click_bell_500/buf.pkl \
    task.train_buffer.use_relative_action=true \
    $OVERRIDE \
    > $LOGDIR/PT2_human.log 2>&1 &
PID_PT2=$!

# PT-3: SimHum pretrain (GPU 2)
echo "[PT3] SimHum pretrain → GPU 2"
CUDA_VISIBLE_DEVICES=2 python finetune.py \
    exp_name=test_pt3_simhum \
    agent=diffusion_dual \
    agent/features=resnet_gn \
    task=mixed_44_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/hybrid_click_bell_SH1000/buf.pkl \
    human_ratio=0.5 \
    $OVERRIDE \
    > $LOGDIR/PT3_simhum.log 2>&1 &
PID_PT3=$!

# PT-4: Robot-only (GPU 3)
echo "[PT4] Robot-only → GPU 3"
CUDA_VISIBLE_DEVICES=3 python finetune.py \
    exp_name=test_pt4_real \
    agent=diffusion \
    agent/features=resnet_gn \
    task=end_effector_16_3 \
    trainer=bc_cos_sched \
    ac_chunk=40 \
    buffer_path=$DATA/agilex_click_bell_80/buf.pkl \
    $OVERRIDE \
    > $LOGDIR/PT4_real.log 2>&1 &
PID_PT4=$!

# FT-1: Sim → Robot (GPU 4)
echo "[FT1] Sim→Robot finetune → GPU 4"
CUDA_VISIBLE_DEVICES=4 python finetune.py \
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
    lr=0.00005 \
    $OVERRIDE \
    > $LOGDIR/FT1_sim_ft.log 2>&1 &
PID_FT1=$!

# FT-2: Human → Robot (GPU 5)
echo "[FT2] Human→Robot finetune → GPU 5"
CUDA_VISIBLE_DEVICES=5 python finetune.py \
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
    lr=0.00005 \
    $OVERRIDE \
    > $LOGDIR/FT2_human_ft.log 2>&1 &
PID_FT2=$!

# FT-3: SimHum → Robot (GPU 6)
echo "[FT3] SimHum→Robot finetune → GPU 6"
CUDA_VISIBLE_DEVICES=6 python finetune.py \
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
    $OVERRIDE \
    > $LOGDIR/FT3_simhum_ft.log 2>&1 &
PID_FT3=$!

echo ""
echo "All 7 experiments launched. Waiting..."
echo ""

# Wait and collect results
PIDS=("$PID_PT1" "$PID_PT2" "$PID_PT3" "$PID_PT4" "$PID_FT1" "$PID_FT2" "$PID_FT3")
NAMES=("PT1_sim" "PT2_human" "PT3_simhum" "PT4_real" "FT1_sim_ft" "FT2_human_ft" "FT3_simhum_ft")

PASS=0
FAIL=0

for i in "${!PIDS[@]}"; do
    name=${NAMES[$i]}
    pid=${PIDS[$i]}

    if wait $pid; then
        status="PASS"
        PASS=$((PASS + 1))
    else
        status="FAIL (exit=$?)"
        FAIL=$((FAIL + 1))
    fi

    # Extract loss info
    logfile=$LOGDIR/${name}.log
    loss_lines=$(grep -E "train/bc_loss|train_loss|bc_loss" "$logfile" 2>/dev/null | tail -5)

    echo "[$status] $name"
    if [ -n "$loss_lines" ]; then
        echo "$loss_lines" | head -3
    fi
    echo "  Log: $logfile"
    echo ""
done

echo "============================================="
echo "  SUMMARY: $PASS PASS / $((PASS + FAIL)) total"
if [ $FAIL -gt 0 ]; then
    echo "  $FAIL FAILED"
fi
echo "  Logs: $LOGDIR"
echo "============================================="
