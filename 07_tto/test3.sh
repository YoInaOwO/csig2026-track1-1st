#!/bin/bash
# 测试集 100 张：tto3_b 配置 -> 打包
cd /workspace
bash /workspace/run_tto3_par.sh output/test_pair1_750 output/test_pair1_750_tto3_b test_pair1_750_tto3_b /workspace/data/sval 4 --steps 150 --adapt 0.03 --w_clip 4 2>&1 | grep -v -E 'missing|shape mismatch' | tee logs/test_tto3_b.log
n=$(ls output/test_pair1_750_tto3_b | wc -l); echo "files=$n"
if [ "$n" -ge 100 ]; then bash pack_submit.sh output/test_pair1_750_tto3_b submission_pair1_750_tto3_b 2>&1 | tail -2; fi
echo TEST3_DONE
