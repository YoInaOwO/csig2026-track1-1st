"""按类别融合 ft 与 base 测试输出 -> JPEG q95 -> zip"""
import json, os, sys, numpy as np, cv2, zipfile
FT, BASE, OUT = "/workspace/output/test_ft", "/workspace/output/test_base", "/workspace/output/submission_fuse"
route = json.load(open("/workspace/output/route_test.json"))
ALPHA = {"clock": 0.5, "text_cn": 0.6, "text": 0.6}   # base 权重；其余 0.7
DEFAULT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
os.makedirs(OUT, exist_ok=True); missing = []
for i in range(1, 101):
    n = f"case{i}"; pf, pb = f"{FT}/{n}.png", f"{BASE}/{n}.png"
    if not (os.path.exists(pf) and os.path.exists(pb)): missing.append(n); continue
    ft = cv2.imread(pf).astype(np.float32); base = cv2.imread(pb).astype(np.float32)
    a = ALPHA.get(route.get(n, ""), DEFAULT)
    out = np.clip(ft * (1 - a) + base * a, 0, 255).astype(np.uint8)
    lq = cv2.imread(f"/workspace/data/test/{n}.jpg")
    assert out.shape == lq.shape, (n, out.shape, lq.shape)
    from PIL import Image
    Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(f"{OUT}/{n}.jpg", quality=95, subsampling=0)
print("missing:", missing)
if not missing:
    zp = "/workspace/output/submission_fuse.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for i in range(1, 101): z.write(f"{OUT}/case{i}.jpg", f"case{i}.jpg")
    print("zip:", zp, os.path.getsize(zp) / 1e6, "MB")
from collections import Counter
print("alpha usage:", Counter(ALPHA.get(route.get(f"case{i}", ""), DEFAULT) for i in range(1, 101)))
