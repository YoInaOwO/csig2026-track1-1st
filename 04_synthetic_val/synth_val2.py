"""合成验证集 v2：简单退化（先模糊→下采样→上采样→对比度/亮度→轻噪声→JPEG），参数先在 5 对真实图上按 PSNR 拟合，
再从拟合结果附近抽样。全 4K 裁剪（4096x3072 或 3072x4096），20 张，来源为未进训练清单的原图。
输出 /workspace/data/sval/case{i}_gt.png, case{i}_lq.jpg"""
import os, sys, glob, random, json, itertools, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
random.seed(seed); np.random.seed(seed)
OUT = "/workspace/data/sval"; os.makedirs(OUT, exist_ok=True)

def degrade(g, pre, down, post, a, b, noise, q):
    x = cv2.GaussianBlur(g, (0, 0), pre) if pre > 0 else g.copy()
    h, w = x.shape[:2]
    if down > 1:
        x = cv2.resize(x, (int(round(w / down)), int(round(h / down))), interpolation=cv2.INTER_AREA)
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    if post > 0: x = cv2.GaussianBlur(x, (0, 0), post)
    x = x.astype(np.float32) * a + b
    if noise > 0: x = x + np.random.normal(0, noise, x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, 1)

def color_fit(d, l):   # 拟合 l ≈ a*d + b
    dm, lm = d.astype(np.float64).ravel(), l.astype(np.float64).ravel()
    a = ((dm - dm.mean()) * (lm - lm.mean())).sum() / (((dm - dm.mean()) ** 2).sum() + 1e-9); b = lm.mean() - a * dm.mean()
    return a, b

def fit_case(i):
    g = cv2.imread(f"/workspace/data/val/case{i}_gt.jpg"); l = cv2.imread(f"/workspace/data/val/case{i}_lq.jpg")   # 全图拟合，平滑块分辨不出模糊强度
    res = []
    for pre, down, post in itertools.product([0, 0.75, 1.5, 2.25, 3, 4], [1, 2, 3, 4, 6], [0, 1.5]):
        d = degrade(g, pre, down, post, 1.0, 0.0, 0, 100)
        a, b = color_fit(d, l); q = 90
        d2 = degrade(g, pre, down, post, a, b, 0, q)
        res.append((cv2.PSNR(d2, l), pre, down, post, round(a, 3), round(b, 1), q))
    res.sort(reverse=True)
    return i, res[:3], cv2.PSNR(g, l)

if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex:
        fits = list(ex.map(fit_case, range(1, 6)))
    params = []
    for i, top, base in fits:
        print(f"case{i}: PSNR(gt,lq)={base:.2f}  best " + " | ".join(f"psnr={r[0]:.2f} pre={r[1]} down={r[2]} post={r[3]} a={r[4]} b={r[5]} q={r[6]}" for r in top), flush=True)
        params.append(top[0][1:])
    json.dump(params, open(f"{OUT}/fit_params.json", "w"))
    # ---- 选未进训练清单的原图（全 4K 裁剪）----
    used = set()
    for f in glob.glob("/workspace/data/train/list_tiles_*.txt"):
        for l in open(f): used.add(os.path.basename(l.strip()).rsplit("_", 2)[0])
    Q = {"text_cn": 6, "bird": 4, "plant": 4, "clock": 3, "flower": 1, "light": 1, "building": 1}
    # 目标纹理量：真实 GT（::4 采样）的拉普拉斯方差中位数，窗口选最接近者而非最大者
    tv = [cv2.Laplacian(cv2.cvtColor(cv2.imread(f"/workspace/data/val/case{i}_gt.jpg"), cv2.COLOR_BGR2GRAY)[::4, ::4], cv2.CV_32F).var() for i in range(1, 6)]
    target = float(np.median(tv)); print("real GT lap(::4):", [round(t, 1) for t in tv], "target", round(target, 1), flush=True)
    picked = []
    for cat, n in Q.items():
        cands = sum([glob.glob(f"/workspace/data/train/{s}/{cat}/*.jpg") for s in ["pexels", "unsplash"]], [])
        random.shuffle(cands); got = 0
        for p in cands:
            if os.path.splitext(os.path.basename(p))[0] in used: continue
            im = cv2.imread(p)
            if im is None: continue
            H, W = im.shape[:2]
            CW, CH = (4096, 3072) if W >= H else (3072, 4096)
            if W < CW or H < CH: continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); best = (1e9, 0, 0, 0)
            for y in range(0, H - CH + 1, max(1, (H - CH) // 3 or 1)):
                for x in range(0, W - CW + 1, max(1, (W - CW) // 3 or 1)):
                    v = cv2.Laplacian(g[y:y + CH:4, x:x + CW:4], cv2.CV_32F).var()
                    d = abs(np.log((v + 1e-3) / target))
                    if d < best[0]: best = (d, x, y, v)
            d, x, y, v = best
            if d > np.log(3.0): continue   # 纹理量与真实差 2.5 倍以上的图不用
            picked.append((cat, p, x, y, CW, CH)); got += 1
            if got >= n: break
        print(f"{cat}: picked {got}", flush=True)
    meta = []
    for i, (cat, p, x, y, CW, CH) in enumerate(picked, 1):
        im = cv2.imread(p)[y:y + CH, x:x + CW]
        pre, down, post, a, b, q = random.choice(params)
        pre = max(0, pre + random.uniform(-0.5, 0.5)); a = float(np.clip(a + random.uniform(-0.03, 0.03), 0.8, 1.05)); b = b + random.uniform(-4, 4)
        q = int(np.clip(q + random.randint(-6, 6), 70, 98)); noise = random.uniform(0, 1.0)
        lq = degrade(im, pre, down, post, a, b, noise, q)
        cv2.imwrite(f"{OUT}/case{i}_gt.png", im); cv2.imwrite(f"{OUT}/case{i}_lq.png", lq)
        print(f"case{i} {cat} {CW}x{CH} lap={cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[::4, ::4], cv2.CV_32F).var():.1f} pre={pre:.2f} down={down} post={post} a={a:.3f} b={b:.1f} q={q} psnr={cv2.PSNR(im, lq):.2f}", flush=True)
        meta.append({"case": f"case{i}", "cat": cat, "src": p, "x": x, "y": y, "pre": pre, "down": down, "post": post, "a": a, "b": b, "q": q})
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    open("/workspace/data/train/holdout_images.txt", "w").write("\n".join(os.path.splitext(os.path.basename(m["src"]))[0] for m in meta) + "\n")
    print("sval written:", len(meta))
