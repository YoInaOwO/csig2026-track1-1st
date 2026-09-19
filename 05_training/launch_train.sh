#!/bin/bash
# 用法: bash launch_train.sh <exp_tag> <list> <nr_w> <dists_w> <gan_w> <lpips_w> <rgb_w> [gen_ckpt] [dis_ckpt] [save_iters]
cd /workspace
TAG=$1; LIST=$2; NRW=${3:-1.0}; DW=${4:-0.5}; GW=${5:-0.05}; LW=${6:-1.0}; RW=${7:-0.1}
GEN=${8:-/workspace/models/ODTSR/weight.pth}; DIS=${9:-/workspace/models/ODTSR/net_dis_iter_5001.pth}; SAVE=${10:-250}
CFG=/workspace/ODTSR/examples/qwen_image/configs/$TAG.yaml
cat > $CFG <<EOF
exp_tag: $TAG
pretrained_ckpt_path_gen: $GEN
pretrained_ckpt_path_dis: $DIS
learning_rate: !!float 2e-5
learning_rate_dis: !!float 5e-6
accumulate_grad_batches: 4
max_epochs: 1
use_gradient_checkpointing: True
viz_iters: 25
save_ckpt_iters: $SAVE
relaxed_mean_size: 100
use_dual_condition_flag: False
rgb_w: $RW
lpips_w: $LW
dists_w: $DW
ssim_w: ${SSIM_W:-0.0}
nr_w: $NRW
nr_metrics: ${NR_METRICS:-clipiqa,musiq}
nr_scales: ${NR_SCALES:-1.0}
gan_type: vanilla
gan_loss_weight: $GW
real_label_val: 1.0
fake_label_val: 0.0
d_iters: 1
r1_regularization: 5.0
r2_regularization: 0.0
variance: 0.005
lq_no_noise_prob: 0.7
EOF
sed -i 's/\r$//' /workspace/run_train_1gpu.sh
tmux kill-session -t train 2>/dev/null; sleep 3
rm -rf /workspace/ODTSR/experiments/$TAG
tmux new-session -d -s train "bash /workspace/run_train_1gpu.sh 0 $LIST /workspace/ODTSR/examples/qwen_image/deg_tianchi.yaml $CFG 29501 2>&1 | tee /workspace/logs/train_$TAG.log"
echo "launched $TAG  list=$(wc -l < $LIST) nr_w=$NRW dists_w=$DW gan=$GW lpips=$LW rgb=$RW metrics=${NR_METRICS:-clipiqa,musiq} scales=${NR_SCALES:-1.0} gen=$GEN"
