#!/bin/bash
# 打包：<src_dir>（含 case{i}.jpg 或 .png）-> /workspace/output/<name>.zip（根目录 case{i}.jpg，q95 4:4:4，尺寸校验）
cd /workspace; source venv_odtsr/bin/activate
SRC=$1; NAME=$2
python - "$SRC" "$NAME" <<'PY'
import os, sys, zipfile, glob
from PIL import Image
src, name = sys.argv[1], sys.argv[2]
out = f"/workspace/output/{name}"; os.makedirs(out, exist_ok=True)
bad = []
for i in range(1, 101):
    c = glob.glob(f"{src}/case{i}.jpg") + glob.glob(f"{src}/case{i}.png")
    if not c: bad.append(i); continue
    im = Image.open(c[0]).convert("RGB"); lq = Image.open(f"/workspace/data/test/case{i}.jpg")
    if im.size != lq.size: bad.append((i, im.size, lq.size)); continue
    if c[0].endswith(".jpg"):
        os.system(f"cp '{c[0]}' '{out}/case{i}.jpg'")
    else:
        im.save(f"{out}/case{i}.jpg", quality=95, subsampling=0)
if bad: print("BAD:", bad); sys.exit(1)
zp = out + ".zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
    for i in range(1, 101): z.write(f"{out}/case{i}.jpg", f"case{i}.jpg")
print("ZIP_OK", zp, round(os.path.getsize(zp) / 1e6, 1), "MB")
PY
