#!/bin/bash
cd /workspace
source venv_odtsr/bin/activate
# 1) 原始 ODTSR 输出直接导出（sigma=0, alpha=1）
rm -rf /workspace/submit_raw && mkdir -p /workspace/submit_raw
CUDA_VISIBLE_DEVICES=1 python finalize.py --model_dir output/test_odtsr_f1.0 --lq_dir data/test --out_dir /workspace/submit_raw/output_dir --sigma 0 --alpha 1.0 --quality 95
cd /workspace/submit_raw && zip -q -r submission_raw.zip output_dir && ls -la submission_raw.zip && unzip -l submission_raw.zip | tail -1
cd /workspace
# 2) 验证集上跑更激进的 fidelity（GPU0: f=0.6, GPU1: f=0.4）
tmux kill-session -t exp_f06 2>/dev/null; tmux kill-session -t exp_f04 2>/dev/null
tmux new-session -d -s exp_f06 "bash /workspace/run_odtsr_generic.sh 0 /workspace/output/val_odtsr_f0.6 --fidelity 0.6 2>&1 | tee /workspace/logs/odtsr_f0.6.log"
tmux new-session -d -s exp_f04 "bash /workspace/run_odtsr_generic.sh 1 /workspace/output/val_odtsr_f0.4 --fidelity 0.4 2>&1 | tee /workspace/logs/odtsr_f0.4.log"
tmux ls
