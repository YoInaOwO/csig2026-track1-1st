#!/bin/bash
cd /workspace; source venv_odtsr/bin/activate
rm -f output/test_pair1_750_tto3_b/case64.jpg
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python tto3.py output/test_pair1_750 output/test_pair1_750_tto3_b --only case64 --steps 150 --adapt 0.03 --w_clip 4 2>&1 | grep -E '^case|warn|Error|Traceback' | tail -3
python sanity.py output/test_pair1_750 output/test_pair1_750_tto3_b
rm -f output/submission_pair1_750_tto3_b.zip; rm -rf output/submission_pair1_750_tto3_b
bash pack_submit.sh output/test_pair1_750_tto3_b submission_pair1_750_tto3_b 2>&1 | tail -2
echo FIX64_DONE
