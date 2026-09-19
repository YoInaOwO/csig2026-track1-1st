#!/bin/bash
# 补跑缺失的 10 张（run_tto3_par 会跳过已存在的输出），然后打包
cd /workspace
bash /workspace/run_tto3_par.sh output/test_pair1_750 output/test_pair1_750_tto3_b test_pair1_750_tto3_b_fix /workspace/data/sval 4 --steps 150 --adapt 0.03 --w_clip 4 2>&1 | grep -v -E 'missing|shape mismatch' | tee logs/test_tto3_b_fix.log
n=$(ls output/test_pair1_750_tto3_b | wc -l); echo "files=$n"
if [ "$n" -ge 100 ]; then bash pack_submit.sh output/test_pair1_750_tto3_b submission_pair1_750_tto3_b 2>&1 | tail -2; fi
echo TEST3_FIX_DONE
