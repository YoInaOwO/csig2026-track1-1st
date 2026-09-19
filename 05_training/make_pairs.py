"""按标定退化造训练配对：源图（排除 holdout）→ 随机 1024x1024 块 → GT 软化(k∈[1.6,2.4]+JPEG92) → LQ(5 组拟合参数轮转+抖动+空间变化对比度场)。
输出 /workspace/data/train/pairs/{cat}/{id}_{n}_gt.png / _lq.png 与列表 /workspace/data/train/list_pairs.txt（gt|lq 每行）。
用法: python make_pairs.py <per_image_tiles> <max_images_per_cat>"""
import os, sys, glob, json, random, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
TILE = 1024
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MAXC = int(sys.argv[2]) if len(sys.argv) > 2 else 600
OUT = "/workspace/data/train/pairs"
FITS = json.load(open("/workspace/data/calib.json"))["fits"]
hold = set()
for f in glob.glob("/workspace/data/train/holdout_images*.txt"): hold |= set(l.strip() for l in open(f))

def soften(im, k, q=92):
    h, w = im.shape[:2]
    x = cv2.resize(im, (int(round(w / k)), int(round(h / k))), interpolation=cv2.INTER_AREA); x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q]); return cv2.imdecode(buf, 1)
def degrade(g, pre, down, a, b, noise, q, rng):
    x = cv2.GaussianBlur(g, (0, 0), pre) if pre > 0 else g.copy(); h, w = x.shape[:2]
    if down > 1:
        x = cv2.resize(x, (int(round(w / down)), int(round(h / down))), interpolation=cv2.INTER_AREA); x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    ga = cv2.resize(rng.uniform(-0.05, 0.05, (2, 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    gb = cv2.resize(rng.uniform(-6, 6, (2, 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    x = x.astype(np.float32) * (a + ga)[..., None] + (b + gb)[..., None]
    if noise > 0: x = x + rng.normal(0, noise, x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(q)]); return cv2.imdecode(buf, 1)

def job(args):
    cat, p, idx = args
    r = random.Random(idx); rng = np.random.default_rng(idx)
    im = cv2.imread(p)
    if im is None or min(im.shape[:2]) < TILE + 64: return []
    H, W = im.shape[:2]; g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); out = []
    base = os.path.splitext(os.path.basename(p))[0]; d = f"{OUT}/{cat}"; os.makedirs(d, exist_ok=True)
    tries = 0
    while len(out) < PER and tries < PER * 6:
        tries += 1
        y, x = r.randint(0, H - TILE), r.randint(0, W - TILE)
        if cv2.Laplacian(g[y:y + TILE:4, x:x + TILE:4], cv2.CV_32F).var() < 60: continue   # 跳过平坦块
        tile = im[y:y + TILE, x:x + TILE]
        gt = soften(tile, r.uniform(1.6, 2.4))
        f = FITS[(idx + len(out)) % len(FITS)]
        pre = max(0.0, f["pre"] + r.uniform(-1, 1)) if f["pre"] > 0 else (0.0 if r.random() < 0.7 else r.uniform(0.5, 1.5))
        down = max(1.0, f["down"] * r.uniform(0.9, 1.4)); noise = max(0.0, f["noise"] + r.uniform(-1, 1)) if f["noise"] > 0 else (0.0 if r.random() < 0.7 else r.uniform(0.5, 2))
        a = float(np.clip(f["a"] + r.uniform(-0.02, 0.02), 0.9, 1.06)); b = f["b"] + r.uniform(-3, 3); q = r.randint(75, 90)
        lq = degrade(gt, pre, down, a, b, noise, q, rng)
        gp, lp = f"{d}/{base}_{len(out)}_gt.png", f"{d}/{base}_{len(out)}_lq.png"
        cv2.imwrite(gp, gt); cv2.imwrite(lp, lq); out.append(f"{gp}|{lp}")
    return out

if __name__ == "__main__":
    CATS = ["text_cn", "bird", "plant", "clock", "flower", "light", "building"]
    jobs = []; idx = 0
    for cat in CATS:
        srcs = sum([glob.glob(f"/workspace/data/train/{s}/{cat}/*.jpg") for s in ["pexels", "unsplash"]], [])
        srcs = [p for p in srcs if os.path.splitext(os.path.basename(p))[0] not in hold]
        random.Random(0).shuffle(srcs); srcs = srcs[:MAXC]
        for p in srcs: jobs.append((cat, p, idx)); idx += 1
        print(f"{cat}: {len(srcs)} source images", flush=True)
    lines = []
    with ProcessPoolExecutor(96) as ex:
        for i, res in enumerate(ex.map(job, jobs, chunksize=4)):
            lines += res
            if i % 200 == 0: print(f"  {i}/{len(jobs)} pairs={len(lines)}", flush=True)
    open("/workspace/data/train/list_pairs.txt", "w").write("\n".join(lines) + "\n")
    print("PAIRS_DONE", len(lines))
