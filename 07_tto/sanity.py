"""检查 TTO 输出与基线的差异是否异常（均值差 > 8 或 std 差 > 15 视为异常）。用法: python sanity.py <base_dir> <tto_dir>"""
import sys, glob, os, numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
base, tto = sys.argv[1], sys.argv[2]
def chk(p):
    n = os.path.splitext(os.path.basename(p))[0]
    a = np.asarray(Image.open(p).convert("L").resize((512, 384))).astype(float)
    b = np.asarray(Image.open(f"{tto}/{n}.jpg").convert("L").resize((512, 384))).astype(float)
    return n, a.mean(), b.mean(), a.std(), b.std(), np.abs(a - b).mean()
with ProcessPoolExecutor(32) as ex: rs = list(ex.map(chk, sorted(glob.glob(f"{base}/*.png"))))
bad = [r for r in rs if abs(r[1] - r[2]) > 8 or abs(r[3] - r[4]) > 15]
print("n =", len(rs), "bad =", [(r[0], round(r[1], 1), round(r[2], 1)) for r in bad])
print("mean |diff| over all:", round(float(np.mean([r[5] for r in rs])), 3), "max:", round(max(r[5] for r in rs), 3))
