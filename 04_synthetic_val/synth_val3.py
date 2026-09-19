"""合成验证集 v3：GT 软化到官方 GT 统计（k∈[1.6,2.4] 下采样-上采样 + JPEG92），LQ 用 calib.json 里 5 组真实拟合参数抽样+抖动。
全 4K 裁剪，来源为未进训练清单的原图，CPU 并行。输出 /workspace/data/sval/case{i}_gt.png / case{i}_lq.png"""
import os, sys, glob, random, json, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 23
random.seed(seed); np.random.seed(seed)
OUT = "/workspace/data/sval"; os.makedirs(OUT, exist_ok=True)
FITS = json.load(open("/workspace/data/calib.json"))["fits"]

def soften(im, k, q=92):
    h, w = im.shape[:2]
    x = cv2.resize(im, (int(round(w / k)), int(round(h / k))), interpolation=cv2.INTER_AREA)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q]); return cv2.imdecode(buf, 1)
def degrade(g, pre, down, a, b, noise, q, rng):
    x = cv2.GaussianBlur(g, (0, 0), pre) if pre > 0 else g.copy(); h, w = x.shape[:2]
    if down > 1:
        x = cv2.resize(x, (int(round(w / down)), int(round(h / down))), interpolation=cv2.INTER_AREA); x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    x = x.astype(np.float32) * a + b
    if noise > 0: x = x + rng.normal(0, noise, x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(q)]); return cv2.imdecode(buf, 1)
def lap4(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[::4, ::4], cv2.CV_32F).var())

def pick_window(p, target):
    im = cv2.imread(p)
    if im is None: return None
    H, W = im.shape[:2]; CW, CH = (4096, 3072) if W >= H else (3072, 4096)
    if W < CW or H < CH: return None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); best = (1e9, 0, 0, 0)
    for y in range(0, H - CH + 1, max(1, (H - CH) // 3 or 1)):
        for x in range(0, W - CW + 1, max(1, (W - CW) // 3 or 1)):
            v = cv2.Laplacian(g[y:y + CH:4, x:x + CW:4], cv2.CV_32F).var(); d = abs(np.log((v + 1e-3) / target))
            if d < best[0]: best = (d, x, y, v)
    d, x, y, v = best
    return None if d > np.log(4.0) else (p, x, y, CW, CH, v)

def gen(job):
    i, cat, p, x, y, CW, CH, sd = job
    rng = np.random.default_rng(sd); r = random.Random(sd)
    im = cv2.imread(p)[y:y + CH, x:x + CW]
    k = r.uniform(1.6, 2.4); gt = soften(im, k)
    f = r.choice(FITS)
    pre = max(0.0, f["pre"] + r.uniform(-1, 1)) if f["pre"] > 0 else (0.0 if r.random() < 0.7 else r.uniform(0.5, 1.5))
    down = max(1.0, f["down"] * r.uniform(0.8, 1.25)); noise = max(0.0, f["noise"] + r.uniform(-1, 1)) if f["noise"] > 0 else (0.0 if r.random() < 0.7 else r.uniform(0.5, 2))
    a = float(np.clip(f["a"] + r.uniform(-0.02, 0.02), 0.9, 1.06)); b = f["b"] + r.uniform(-3, 3); q = r.randint(85, 95)
    lq = degrade(gt, pre, down, a, b, noise, q, rng)
    cv2.imwrite(f"{OUT}/case{i}_gt.png", gt); cv2.imwrite(f"{OUT}/case{i}_lq.png", lq)
    return {"case": f"case{i}", "cat": cat, "src": p, "x": x, "y": y, "k": k, "fit_case": f["case"], "pre": pre, "down": down, "noise": noise, "a": a, "b": b, "q": q,
            "lap4_gt": lap4(gt), "lap4_lq": lap4(lq), "psnr": cv2.PSNR(gt, lq)}

if __name__ == "__main__":
    used = set()
    for f in glob.glob("/workspace/data/train/list_tiles_*.txt"):
        for l in open(f): used.add(os.path.basename(l.strip()).rsplit("_", 2)[0])
    target = 656.4   # 官方 GT lap(::4) 中位数
    Q = {"text_cn": 7, "bird": 5, "plant": 5, "clock": 4, "flower": 1, "light": 1, "building": 1}
    picked = []
    with ProcessPoolExecutor(32) as ex:
        for cat, n in Q.items():
            cands = sum([glob.glob(f"/workspace/data/train/{s}/{cat}/*.jpg") for s in ["pexels", "unsplash"]], [])
            cands = [p for p in cands if os.path.splitext(os.path.basename(p))[0] not in used]; random.shuffle(cands)
            got = 0
            for chunk_start in range(0, len(cands), 48):
                res = [r for r in ex.map(pick_window, cands[chunk_start:chunk_start + 48], [target] * 48) if r]
                for r in res:
                    if got < n: picked.append((cat,) + r); got += 1
                if got >= n: break
            print(f"{cat}: picked {got}", flush=True)
        jobs = [(i, cat, p, x, y, CW, CH, seed * 1000 + i) for i, (cat, p, x, y, CW, CH, v) in enumerate(picked, 1)]
        meta = list(ex.map(gen, jobs))
    for m in meta: print(f"{m['case']} {m['cat']} k={m['k']:.2f} fit=case{m['fit_case']} pre={m['pre']:.1f} down={m['down']:.1f} noise={m['noise']:.1f} a={m['a']:.3f} b={m['b']:.1f} q={m['q']} lap4 gt={m['lap4_gt']:.0f} lq={m['lap4_lq']:.0f} psnr={m['psnr']:.2f}")
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    open("/workspace/data/train/holdout_images.txt", "w").write("\n".join(os.path.splitext(os.path.basename(m["src"]))[0] for m in meta) + "\n")
    print("sval written:", len(meta))
