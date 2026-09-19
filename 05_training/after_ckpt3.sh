#!/bin/bash
# 等 pair1c iter_251（累计 750）-> 停训 -> 并行：评 pair1_750 + TTO e8d on pair1_500 -> TTO e8d on pair1_750
cd /workspace
TTO="--eps 8 --steps 100 --crop 1024 --crops 4 --w_musiq 3 --w_maniqa 3 --w_clip 1 --w_lpips 2 --w_dists 2"
CK=""
while [ -z "$CK" ]; do CK=$(ls /workspace/ODTSR/experiments/pair1c/*/checkpoints/net_gen_iter_251.pth 2>/dev/null | head -1); sleep 20; done
sleep 60; tmux kill-session -t train; sleep 8
echo "ckpt: $CK"
tmux new-session -d -s tto "bash /workspace/run_tto_par.sh output/sval_pair1_500 output/sval_pair1_500_tto_e8d sval_pair1_500_tto_e8d /workspace/data/sval 3 $TTO 2>&1 | tee logs/tto_pair1_500_e8d.log"
bash /workspace/sval_infer.sh pair1_750 $CK 1.0 2>&1 | tee logs/sval_pair1_750.log
while ! grep -q RUN_TTO_PAR_DONE logs/tto_pair1_500_e8d.log; do sleep 30; done
bash /workspace/run_tto_par.sh output/sval_pair1_750 output/sval_pair1_750_tto_e8d sval_pair1_750_tto_e8d /workspace/data/sval 4 $TTO 2>&1 | tee logs/tto_pair1_750_e8d.log
echo AFTER_CKPT3_DONE
