#!/bin/bash
# 官方 5 张验证集：pair1_750 推理 -> TTO v7 -> eval2 打分
cd /workspace; source venv_odtsr/bin/activate
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image; export qwen_path=/workspace/models/Qwen-Image
CK=$(ls /workspace/ODTSR/experiments/pair1c/*/checkpoints/net_gen_iter_251.pth | head -1); echo "CK=$CK"
OUT=/workspace/output/val_pair1_750
if [ "$(ls $OUT 2>/dev/null | wc -l)" -lt 5 ]; then
  cd /workspace/ODTSR/examples/qwen_image
  CUDA_VISIBLE_DEVICES=0 python infer_odtsr.py --input_path /workspace/output/val_LQ --output_path $OUT --scale 1.0 --tile 64 --ext png --trained_ckpt $CK --fidelity 1.0 2>&1 | grep -E 'Error|Traceback' | tail -3
  cd /workspace
fi
echo "INFER files=$(ls $OUT | wc -l)"
CUDA_VISIBLE_DEVICES=0 python eval2.py $OUT --tag val_pair1_750 --val_dir /workspace/data/val 2>&1 | grep -A1 MEAN | cut -c1-260
bash /workspace/run_tto_par.sh $OUT ${OUT}_tto_v7 val_pair1_750_tto_v7 /workspace/data/val 5 --eps 12 --steps 200 --crop 1024 --crops 6 --w_musiq 3 --w_maniqa 4 --w_clip 2 --w_lpips 2 --w_dists 2 --w_niqe 6 --w_ssim 1 --adapt 0.03 2>&1 | grep -A1 -E 'MEAN|Error'
echo VAL_CHECK_DONE
