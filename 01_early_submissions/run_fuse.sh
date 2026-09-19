#!/bin/bash
cd /workspace
source venv_odtsr/bin/activate
tmux kill-session -t train 2>/dev/null; sleep 8
python fuse_eval.py ${1:-v4_1500} ${2:-base_reflect} 2>&1 | grep -E 'MEAN|Error'
python sharp_report.py base_reflect,v4_1500,fuse_mix0.5,fuse_hf3.0,fuse_hf3_half 2>&1 | grep -v Warning
# 恢复 v4
G=$(ls -t /workspace/ODTSR/experiments/tianchi_v4/*/checkpoints/net_gen_iter_*.pth | grep -v iter_1.pth | head -1); D=${G/net_gen/net_dis}
sed -e "s#pretrained_ckpt_path_gen: .*#pretrained_ckpt_path_gen: $G#" -e "s#pretrained_ckpt_path_dis: .*#pretrained_ckpt_path_dis: $D#" /workspace/ODTSR/examples/qwen_image/configs/train_v4.yaml > /workspace/ODTSR/examples/qwen_image/configs/train_resume.yaml
tmux new-session -d -s train "bash /workspace/run_train.sh /workspace/data/train/list_full.txt /workspace/ODTSR/examples/qwen_image/deg_tianchi.yaml /workspace/ODTSR/examples/qwen_image/configs/train_resume.yaml 2>&1 | tee -a /workspace/logs/train_v4.log"
echo "resumed from $G"
echo RUN_DONE
