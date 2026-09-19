#!/bin/bash
# 测试集推理：用法 bash test_infer.sh <tag> <ckpt> <fidelity> [start_end]
cd /workspace; source venv_odtsr/bin/activate
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image; export qwen_path=/workspace/models/Qwen-Image
TAG=$1; CK=$2; FID=$3; SE=${4:-}
OUT=/workspace/output/test_$TAG
cd /workspace/ODTSR/examples/qwen_image
EXTRA=""; if [ -n "$SE" ]; then EXTRA="--start_end $SE"; fi
CUDA_VISIBLE_DEVICES=0 python infer_odtsr.py --input_path /workspace/data/test --output_path $OUT --scale 1.0 --tile 64 --ext png --trained_ckpt $CK --fidelity $FID $EXTRA 2>&1 | grep -E '^\[[0-9]|Error|Traceback'
cd /workspace; echo "TEST_INFER_DONE $TAG $(ls $OUT | wc -l)"
