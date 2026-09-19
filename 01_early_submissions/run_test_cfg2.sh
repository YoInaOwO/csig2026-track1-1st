#!/bin/bash
# 测试集：base ODTSR, cfg 2.0 + 负提示, fidelity 0.4，双卡各半
cd /workspace; source venv_odtsr/bin/activate
export qwen_path=/workspace/models/Qwen-Image
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image
cd /workspace/ODTSR/examples/qwen_image
BASE=/workspace/models/ODTSR/weight.pth
NEG="blurry, out of focus, low resolution, noisy, jpeg artifacts, oversmoothed, low quality"
FID=${1:-0.4}; OUT=/workspace/output/test_cfg2_f$FID
C="--input_path /workspace/data/test --output_path $OUT --scale 1.0 --tile 64 --ext png --trained_ckpt $BASE --fidelity $FID --cfg 2.0 --negative_prompt"
CUDA_VISIBLE_DEVICES=0 python infer_odtsr.py $C "$NEG" --start_end 0,50 > /workspace/logs/test_cfg2_a.log 2>&1 &
sleep 90
CUDA_VISIBLE_DEVICES=1 python infer_odtsr.py $C "$NEG" --start_end 50, > /workspace/logs/test_cfg2_b.log 2>&1 &
wait
echo "files: $(ls $OUT | wc -l)"
grep -h -E 'Error|Traceback' /workspace/logs/test_cfg2_*.log | head -3
# 打包 JPEG q95 4:4:4
cd /workspace; python - <<PY
import os, zipfile, numpy as np
from PIL import Image
src = "$OUT"; out = "/workspace/output/submission_cfg2_f$FID"; os.makedirs(out, exist_ok=True)
for i in range(1, 101):
    im = Image.open(f"{src}/case{i}.png").convert("RGB"); lq = Image.open(f"/workspace/data/test/case{i}.jpg")
    assert im.size == lq.size, (i, im.size, lq.size)
    im.save(f"{out}/case{i}.jpg", quality=95, subsampling=0)
zp = out + ".zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
    for i in range(1, 101): z.write(f"{out}/case{i}.jpg", f"case{i}.jpg")
print("zip", zp, os.path.getsize(zp) / 1e6, "MB")
PY
echo TEST_CFG2_DONE
