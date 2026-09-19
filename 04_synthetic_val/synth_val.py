"""合成验证集：从未进训练清单的 4K 原图取 30 张（按测试集配比），裁 1536x1152 内容最丰富区域为 GT，
用训练同款退化（deg_tianchi.yaml + tianchi_extra_degrade）生成 LQ。写到 /workspace/data/sval/case{i}_{gt,lq}.png
同时用 5 对真实图校验合成退化统计。用法: python synth_val.py [seed]"""
import os, sys, glob, random, json, numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, "/workspace/ODTSR")
from diffsynth.extensions.realesrgan.realesrgan import RealESRGAN_degradation
from diffsynth.extensions.realesrgan.dataset import tianchi_extra_degrade
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
OUT = "/workspace/data/sval"; os.makedirs(OUT, exist_ok=True)
CW, CH = 1536, 1152
deg = RealESRGAN_degradation("/workspace/ODTSR/examples/qwen_image/deg_tianchi.yaml", device="cpu")

def degrade(gt_rgb_u8, reject=True):
    g0 = cv2.Laplacian(cv2.cvtColor(gt_rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F).var()
    for _ in range(8):
        _, lq = deg.degrade_process(gt_rgb_u8.astype(np.float32) / 255.0, resize_bak=True)
        lq = tianchi_extra_degrade(lq)
        out = (lq[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        r = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F).var() / (g0 + 1e-6)
        if not reject or 0.01 <= r <= 0.7: break   # 拒绝噪声主导/几乎无退化的抽样
    return out

def stats(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F).var()
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = g.shape; yy, xx = np.mgrid[-h // 2:h - h // 2, -w // 2:w - w // 2]; r = np.sqrt((yy / h) ** 2 + (xx / w) ** 2)
    return lap, float(f[r > 0.25].sum() / (f.sum() + 1e-9))

# ---- 1) 用真实 5 对校验 ----
print("== real vs synth ==")
for i in range(1, 6):
    gt = cv2.imread(f"/workspace/data/val/case{i}_gt.jpg"); lq = cv2.imread(f"/workspace/data/val/case{i}_lq.jpg")
    # 只在中心 1536x1152 上算，省时间
    y, x = (gt.shape[0] - CH) // 2, (gt.shape[1] - CW) // 2
    gtc, lqc = gt[y:y + CH, x:x + CW], lq[y:y + CH, x:x + CW]
    lg, hg = stats(gtc); ll, hl = stats(lqc)
    ps, rat, hfs = [], [], []
    for k in range(3):
        s = degrade(cv2.cvtColor(gtc, cv2.COLOR_BGR2RGB)); s = cv2.cvtColor(s, cv2.COLOR_RGB2BGR)
        ls, hs = stats(s); rat.append(ls / lg); hfs.append(hs)
        ps.append(cv2.PSNR(s, lqc))
    print(f"case{i}: real lap_ratio={ll/lg:.3f} hf={hl:.4f} | synth lap_ratio={np.mean(rat):.3f}±{np.std(rat):.3f} hf={np.mean(hfs):.4f} | PSNR(synth,real)={np.mean(ps):.2f}  PSNR(gt,real)={cv2.PSNR(gtc, lqc):.2f}", flush=True)

# ---- 2) 选未进训练清单的原图 ----
used = set()
for f in glob.glob("/workspace/data/train/list_tiles_*.txt"):
    for l in open(f):
        b = os.path.basename(l.strip()); used.add(b.rsplit("_", 2)[0])
Q = {"text_cn": 9, "bird": 6, "plant": 5, "clock": 5, "flower": 2, "light": 2, "building": 1}
picked = []
for cat, n in Q.items():
    cands = []
    for src in ["pexels", "unsplash"]:
        cands += glob.glob(f"/workspace/data/train/{src}/{cat}/*.jpg")
    random.shuffle(cands); got = 0
    for p in cands:
        if os.path.splitext(os.path.basename(p))[0] in used: continue
        im = cv2.imread(p)
        if im is None or min(im.shape[:2]) < 3000: continue
        # 内容最丰富的窗口
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); best = (-1, 0, 0)
        for y in range(0, im.shape[0] - CH + 1, 384):
            for x in range(0, im.shape[1] - CW + 1, 512):
                v = cv2.Laplacian(g[y:y + CH:4, x:x + CW:4], cv2.CV_32F).var()
                if v > best[0]: best = (v, x, y)
        v, x, y = best
        if v < 20: continue
        picked.append((cat, p, x, y)); got += 1
        if got >= n: break
    print(f"{cat}: picked {got}", flush=True)
meta = []
for i, (cat, p, x, y) in enumerate(picked, 1):
    im = cv2.imread(p)[y:y + CH, x:x + CW]
    lq = cv2.cvtColor(degrade(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{OUT}/case{i}_gt.png", im); cv2.imwrite(f"{OUT}/case{i}_lq.png", lq)
    print(f"case{i} {cat} lap_ratio={stats(lq)[0]/stats(im)[0]:.3f} psnr={cv2.PSNR(im, lq):.2f}", flush=True)
    print(f"case{i} {cat} lap_ratio={stats(lq)[0]/stats(im)[0]:.3f} psnr={cv2.PSNR(im, lq):.2f}", flush=True)
    meta.append({"case": f"case{i}", "cat": cat, "src": p, "x": x, "y": y})
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
open("/workspace/data/train/holdout_images.txt", "w").write("\n".join(os.path.splitext(os.path.basename(m["src"]))[0] for m in meta) + "\n")
print("sval written:", len(meta))
