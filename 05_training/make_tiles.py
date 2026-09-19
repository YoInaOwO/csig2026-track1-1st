"""GT 切块 + 锐度筛选。
  python make_tiles.py stats                 -> 官方 GT / LQ 的 tile 级锐度分布，写 /workspace/data/tile_stats.json
  python make_tiles.py tile <src> <cat> <dir> [max_per_img]  -> 切 768 块到 /workspace/data/tiles/<src>_<cat>/，按阈值筛
  python make_tiles.py uhd                   -> UHD-IQA 按标签/MOS 分类后切块
锐度指标: lap = Laplacian 方差(gray, 3x3)；hf = 频谱中半径 > N/4 的能量占比。都在 512 中心块上算以与训练裁剪一致。"""
import os, sys, glob, json, csv, collections, numpy as np, cv2
from multiprocessing import Pool
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
T = 768; OUT = "/workspace/data/tiles"; STATS = "/workspace/data/tile_stats.json"

def metrics(tile_bgr):
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c = (g.shape[0] - 512) // 2; g = g[c:c + 512, c:c + 512]
    std = float(g.std())
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    yy, xx = np.mgrid[-256:256, -256:256]; r = np.sqrt(yy ** 2 + xx ** 2)
    tot = f.sum() + 1e-9; hf = float(f[r > 128].sum() / tot); mf = float(f[(r > 64) & (r <= 128)].sum() / tot)
    return {"std": std, "lap": lap, "hf": hf, "mf": mf}

def grid(im):
    h, w = im.shape[:2]
    for y in range(0, h - T + 1, T):
        for x in range(0, w - T + 1, T):
            yield x, y

def stats():
    res = {"gt": [], "lq": []}
    for p in sorted(glob.glob("/workspace/data/val/case*_gt.jpg")):
        gt = cv2.imread(p); lq = cv2.imread(p.replace("_gt", "_lq"))
        for x, y in grid(gt):
            mg = metrics(gt[y:y + T, x:x + T]); ml = metrics(lq[y:y + T, x:x + T])
            if mg["std"] < 8: continue
            res["gt"].append(mg); res["lq"].append(ml)
    out = {}
    for k in ["gt", "lq"]:
        for m in ["std", "lap", "hf", "mf"]:
            v = np.array([r[m] for r in res[k]]); out[f"{k}_{m}"] = {str(q): float(np.percentile(v, q)) for q in [5, 10, 25, 50, 75, 90, 95, 99]}
    out["n"] = len(res["gt"])
    json.dump(out, open(STATS, "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, dict): print(k, {q: round(x, 4) for q, x in v.items()})
    print("tiles:", out["n"])

def load_thr():
    s = json.load(open(STATS))
    # 保留区间：lap 在 GT 的 [p10, 2.5*p95]，hf 在 [p10, 3*p95]（允许略锐于 GT，砍掉极端过锐/过软）
    return {"lap_lo": s["gt_lap"]["10"], "lap_hi": 2.5 * s["gt_lap"]["95"], "hf_lo": s["gt_hf"]["10"], "hf_hi": 3.0 * s["gt_hf"]["95"], "std_lo": 10.0}

def tile_one(a):
    path, outdir, maxn, thr = a
    try:
        im = cv2.imread(path)
        if im is None or min(im.shape[:2]) < 2200: return (path, 0, "small")
        cands = []
        for x, y in grid(im):
            t = im[y:y + T, x:x + T]; m = metrics(t)
            if m["std"] < thr["std_lo"]: continue
            if not (thr["lap_lo"] <= m["lap"] <= thr["lap_hi"]): continue
            if not (thr["hf_lo"] <= m["hf"] <= thr["hf_hi"]): continue
            cands.append((m["lap"], x, y))
        # 取锐度最高的 maxn 个（细节更丰富），但为多样性从前 2*maxn 里均匀取
        cands.sort(reverse=True); cands = cands[:2 * maxn][::2][:maxn] if len(cands) > maxn else cands
        base = os.path.splitext(os.path.basename(path))[0]
        n = 0
        for _, x, y in cands:
            cv2.imwrite(f"{outdir}/{base}_{y}_{x}.png", im[y:y + T, x:x + T], [cv2.IMWRITE_PNG_COMPRESSION, 1]); n += 1
        return (path, n, "ok")
    except Exception as e:
        return (path, 0, str(e)[:60])

def tile(src, cat, d, maxn=12, files=None):
    thr = load_thr(); outdir = f"{OUT}/{src}_{cat}"; os.makedirs(outdir, exist_ok=True)
    files = files if files is not None else sorted(glob.glob(f"{d}/*.jpg") + glob.glob(f"{d}/*.jpeg") + glob.glob(f"{d}/*.png") + glob.glob(f"{d}/*.JPG"))
    with Pool(min(96, max(1, len(files)))) as pool:
        res = pool.map(tile_one, [(f, outdir, maxn, thr) for f in files], chunksize=2)
    tot = sum(r[1] for r in res); reasons = collections.Counter(r[2] if r[2] != "ok" else ("ok" if r[1] else "filtered") for r in res)
    lst = sorted(glob.glob(f"{outdir}/*.png"))
    open(f"{OUT}/list_{src}_{cat}.txt", "w").write("\n".join(lst) + "\n")
    print(f"[{src}/{cat}] imgs={len(files)} tiles={tot} ({len(lst)} on disk) {dict(reasons)}", flush=True)

def uhd():
    root = "/workspace/data/uhdiqa"
    imgs = {os.path.basename(p): p for p in glob.glob(f"{root}/**/*.jpg", recursive=True)}
    meta = {r["image_name"]: r for r in csv.DictReader(open(f"{root}/uhd-iqa-metadata.csv"))}
    orig2name = {r["image_name_orig"]: n for n, r in meta.items()}   # tags 表用的是原始 hash 文件名
    tags = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{root}/uhd-iqa-tags.csv")):
        try: c = float(r["machine_confidence"])
        except: continue
        name = orig2name.get(r["image_name"], r["image_name"])
        tags[name][r["tag"].lower()] = max(c, tags[name].get(r["tag"].lower(), 0))
    M = {"bird": {"bird", "beak", "feather", "parrot", "owl", "eagle", "duck", "seagull", "hummingbird", "heron"},
         "clock": {"clock", "clock tower", "analog clock", "wall clock", "watch", "wristwatch"},
         "flower": {"flower", "blossom", "petal", "rose", "tulip", "daisy", "lotus", "sunflower", "flower arrangement"},
         "plant": {"plant", "tree", "leaf", "vegetation", "forest", "foliage", "grass", "branch", "moss", "fern", "bamboo", "jungle", "woodland"},
         "text_cn": {"text", "sign", "word", "alphabet", "neon", "signboard", "billboard", "chinese", "calligraphy", "lantern", "temple", "pagoda"},
         "light": {"light", "lighting", "night", "city", "urban", "street light", "lamp", "neon"}}
    groups = collections.defaultdict(list)
    for name, tg in tags.items():
        if name not in imgs or name not in meta: continue
        try: mos = float(meta[name]["quality_mos"])
        except: continue
        if mos < 0.6: continue
        best, bc = None, 0
        for cat, words in M.items():
            c = max([tg.get(w, 0) for w in words] + [0])
            if c > bc: best, bc = cat, c
        if best and bc >= 70: groups[best].append(imgs[name])
    print({k: len(v) for k, v in groups.items()}, flush=True)
    for cat, files in groups.items(): tile("uhd", cat, None, 12, files)

if __name__ == "__main__":
    if sys.argv[1] == "stats": stats()
    elif sys.argv[1] == "uhd": uhd()
    else: tile(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 12)
