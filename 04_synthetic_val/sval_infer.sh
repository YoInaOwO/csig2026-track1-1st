#!/bin/bash
# 用法: bash sval_infer.sh <tag> <ckpt> <fidelity> [cfg] [neg]   在 data/sval_lq 上推理并 eval2
cd /workspace; source venv_odtsr/bin/activate
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image; export qwen_path=/workspace/models/Qwen-Image
TAG=$1; CK=$2; FID=$3; CFG=${4:-1.0}; NEG=${5:-}
OUT=/workspace/output/sval_$TAG
cd /workspace/ODTSR/examples/qwen_image
EXTRA=""; if [ "$CFG" != "1.0" ]; then EXTRA="--cfg $CFG --negative_prompt"; fi
if [ -n "$EXTRA" ]; then
  CUDA_VISIBLE_DEVICES=0 python infer_odtsr.py --input_path /workspace/data/sval_lq --output_path $OUT --scale 1.0 --tile 64 --ext png --trained_ckpt $CK --fidelity $FID $EXTRA "$NEG" 2>&1 | grep -E '^\[[0-9]|Error|Traceback' | tail -3
else
  CUDA_VISIBLE_DEVICES=0 python infer_odtsr.py --input_path /workspace/data/sval_lq --output_path $OUT --scale 1.0 --tile 64 --ext png --trained_ckpt $CK --fidelity $FID 2>&1 | grep -E '^\[[0-9]|Error|Traceback' | tail -3
fi
cd /workspace; ls $OUT | wc -l
CUDA_VISIBLE_DEVICES=0 python eval2.py $OUT --tag sval_$TAG --val_dir /workspace/data/sval 2>&1 | grep MEAN | cut -c1-260
echo SVAL_INFER_DONE $TAG
