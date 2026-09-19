#!/bin/bash
# 合成验证集可靠性校验：LQ 公式分 vs 2.18；cfg2_f0.4 流程输出公式分 vs 3.71
cd /workspace; source venv_odtsr/bin/activate
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image
export qwen_path=/workspace/models/Qwen-Image
while ! grep -q "sval written" logs/synth_val3.log; do sleep 10; done
mkdir -p data/sval_lq; rm -f data/sval_lq/*
for f in data/sval/case*_lq.png; do b=$(basename $f _lq.png); ln -sf /workspace/$f data/sval_lq/$b.png; done
echo "== LQ vs GT (synthetic) =="
CUDA_VISIBLE_DEVICES=0 python eval2.py lq --tag sval_LQ --val_dir /workspace/data/sval 2>&1 | grep MEAN
NEG="blurry, out of focus, low resolution, noisy, jpeg artifacts, oversmoothed, low quality"
OUT=/workspace/output/sval_cfg2_f0.4; rm -rf $OUT
cd /workspace/ODTSR/examples/qwen_image
CUDA_VISIBLE_DEVICES=0 python infer_odtsr.py --input_path /workspace/data/sval_lq --output_path $OUT --scale 1.0 --tile 64 --ext png \
  --trained_ckpt /workspace/models/ODTSR/weight.pth --fidelity 0.4 --cfg 2.0 --negative_prompt "$NEG" 2>&1 | grep -E '^\[[0-9]|Error|Traceback' | tail -4
cd /workspace; ls $OUT | wc -l
echo "== cfg2_f0.4 vs GT (synthetic) =="
CUDA_VISIBLE_DEVICES=0 python eval2.py $OUT --tag sval_cfg2_f0.4 --val_dir /workspace/data/sval 2>&1 | grep MEAN
echo "== 真实 5 张参考 =="
python score_pred.py | grep -E "tag|^LQ |cfg2_f0.4 |^GT "
echo SVAL_CHECK_DONE
