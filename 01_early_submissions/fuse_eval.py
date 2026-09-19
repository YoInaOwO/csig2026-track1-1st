"""融合实验：ft(保真) 与 base(纹理) 的像素混合 / 频域融合，输出到 /workspace/output/val_fuse_* 并评测"""
import numpy as np, cv2, os, subprocess, sys
FT = sys.argv[1] if len(sys.argv) > 1 else "v4_1500"; BASE = sys.argv[2] if len(sys.argv) > 2 else "base_reflect"
def load(t, i): return cv2.imread(f"/workspace/output/val_{t}/case{i}.png").astype(np.float32)
def save(tag, i, im):
    os.makedirs(f"/workspace/output/val_{tag}", exist_ok=True); cv2.imwrite(f"/workspace/output/val_{tag}/case{i}.png", np.clip(im, 0, 255).astype(np.uint8))
def hf_merge(low_src, hf_src, sigma):
    """低频来自 low_src，高频来自 hf_src（高斯分频）"""
    return cv2.GaussianBlur(low_src, (0, 0), sigma) + (hf_src - cv2.GaussianBlur(hf_src, (0, 0), sigma))
variants = {}
for i in range(1, 6):
    ft, base = load(FT, i), load(BASE, i)
    for a in [0.3, 0.5, 0.7]:
        save(f"fuse_mix{a}", i, ft * (1 - a) + base * a)
    for s in [1.5, 3.0, 6.0]:
        save(f"fuse_hf{s}", i, hf_merge(ft, base, s))
    # 高频取一半强度
    save("fuse_hf3_half", i, hf_merge(ft, ft * 0.5 + base * 0.5, 3.0))
tags = [f"fuse_mix{a}" for a in [0.3, 0.5, 0.7]] + [f"fuse_hf{s}" for s in [1.5, 3.0, 6.0]] + ["fuse_hf3_half"]
for t in tags:
    out = subprocess.run(f"cd /workspace && CUDA_VISIBLE_DEVICES=1 python eval.py /workspace/output/val_{t} --tag {t} 2>&1 | grep MEAN", shell=True, capture_output=True, text=True)
    print(out.stdout.strip(), flush=True)
print("FUSE_DONE")
