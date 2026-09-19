"""标定：(A) 合成 GT 软化因子 k，使全分辨率 lap/hf/musiq 贴近官方 GT；(B) 在 5 对真实图上按 SSIM/LPIPS/lap 比拟合退化参数。
输出 /workspace/data/calib.json"""
import glob, os, json, itertools, numpy as np, cv2, torch, warnings, pyiqa
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")
dev = "cuda"
def soften(im, k, q=92):
    h, w = im.shape[:2]
    x = cv2.resize(im, (int(round(w / k)), int(round(h / k))), interpolation=cv2.INTER_AREA)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q]); return cv2.imdecode(buf, 1)
def degrade(g, pre, down, post, a=1.0, b=0.0, noise=0.0, q=90):
    x = cv2.GaussianBlur(g, (0, 0), pre) if pre > 0 else g.copy(); h, w = x.shape[:2]
    if down > 1:
        x = cv2.resize(x, (int(round(w / down)), int(round(h / down))), interpolation=cv2.INTER_AREA); x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    if post > 0: x = cv2.GaussianBlur(x, (0, 0), post)
    x = x.astype(np.float32) * a + b
    if noise > 0: x = x + np.random.normal(0, noise, x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(q)]); return cv2.imdecode(buf, 1)
def lapv(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F).var())
def hf(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32); h, w = g.shape; c = g[h // 2 - 512:h // 2 + 512, w // 2 - 512:w // 2 + 512]; c = c - c.mean()
    f = np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2; yy, xx = np.mgrid[-512:512, -512:512]; r = np.sqrt(yy ** 2 + xx ** 2) / 1024
    return float(f[r > 0.25].sum() / (f.sum() + 1e-9))
def t(im): return torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
def color_fit(d, l):
    dm, lm = d.astype(np.float64).ravel(), l.astype(np.float64).ravel()
    a = ((dm - dm.mean()) * (lm - lm.mean())).sum() / (((dm - dm.mean()) ** 2).sum() + 1e-9); return a, lm.mean() - a * dm.mean()

_G = {}
def mk(args):
    g, l = _G["g"], _G["l"]; pre, down, nz = args
    d0 = degrade(g, pre, down, 0, 1, 0, 0, 100); a, b = color_fit(d0, l); return degrade(g, pre, down, 0, a, b, nz, 90), a, b

def soft_job(args):
    p, k = args; im = cv2.imread(p); s = soften(im, k) if k > 1 else im
    return p, k, lapv(s), hf(s)

if __name__ == "__main__":
    ssim = pyiqa.create_metric("ssim", device=dev); lpips = pyiqa.create_metric("lpips", device=dev); musiq = pyiqa.create_metric("musiq", device=dev)
    import sys
    skipA = "--skipA" in sys.argv
    out = json.load(open("/workspace/data/calib.json")) if skipA and os.path.exists("/workspace/data/calib.json") else {}
    # ---- A: 软化因子 ----
    print("== A: soften factor (target real GT: lap med 44, hf 0.0016, musiq 33-36) ==", flush=True)
    syn = [] if skipA else sorted(glob.glob("/workspace/data/sval/case*_gt.png"))[:8]
    KS = [] if skipA else [1, 1.5, 2, 2.5, 3, 4]
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(soft_job, [(p, k) for k in KS for p in syn]))
    byk = {k: [] for k in KS}
    for p, k, l, h in res: byk[k].append((l, h))
    for k in KS:
        # musiq 在 GPU 上算 4 张
        ms = []
        for p in syn[:4]:
            im = cv2.imread(p); s = soften(im, k) if k > 1 else im
            with torch.no_grad(): ms.append(float(musiq(t(s))))
        a = np.array(byk[k]); print(f"k={k}: lap med={np.median(a[:,0]):.1f} hf med={np.median(a[:,1]):.4f} musiq={np.mean(ms):.1f}", flush=True)
    if KS: out["soften_stats"] = {str(k): [float(np.median(np.array(byk[k])[:, 0])), float(np.median(np.array(byk[k])[:, 1]))] for k in KS}
    # ---- B: 退化拟合（真实 5 对，目标 = 真实 LQ 的 ssim/lpips/lap 比）----
    print("== B: fit degradation on real pairs by ssim/lpips/lap ==", flush=True)
    fits = []
    for i in range(1, 6):
        g = cv2.imread(f"/workspace/data/val/case{i}_gt.jpg"); l = cv2.imread(f"/workspace/data/val/case{i}_lq.jpg"); gt_t = t(g)
        with torch.no_grad(): rs, rl = float(ssim(t(l), gt_t)), float(lpips(t(l), gt_t))
        rlap = lapv(l) / (lapv(g) + 1e-6)
        cands = []
        grid = list(itertools.product([0, 1, 2, 3, 4, 6], [1, 2, 4, 6], [0, 2, 4, 8]))
        # 退化在 CPU 池并行（fork 继承全局 _G），GPU 只算指标
        _G["g"], _G["l"] = g, l
        with ProcessPoolExecutor(24) as ex: outs = list(ex.map(mk, grid))
        for (pre, down, nz), (d, a, b) in zip(grid, outs):
            with torch.no_grad(): s_, l_ = float(ssim(t(d), gt_t)), float(lpips(t(d), gt_t))
            sl = lapv(d) / (lapv(g) + 1e-6)
            cost = abs(s_ - rs) / 0.03 + abs(l_ - rl) / 0.03 + abs(np.log((sl + 1e-3) / (rlap + 1e-3))) / 0.5
            cands.append((cost, pre, down, nz, round(a, 3), round(b, 1), s_, l_, sl, cv2.PSNR(d, l)))
        cands.sort()
        print(f"case{i}: real ssim={rs:.3f} lpips={rl:.3f} lap_ratio={rlap:.3f} psnr(gt,lq)={cv2.PSNR(g, l):.2f}", flush=True)
        for c in cands[:3]: print(f"   cost={c[0]:.2f} pre={c[1]} down={c[2]} noise={c[3]} a={c[4]} b={c[5]} -> ssim={c[6]:.3f} lpips={c[7]:.3f} lap_ratio={c[8]:.3f} psnr(syn,real)={c[9]:.2f}", flush=True)
        fits.append({"case": i, "pre": cands[0][1], "down": cands[0][2], "noise": cands[0][3], "a": cands[0][4], "b": cands[0][5], "real": [rs, rl, rlap]})
    out["fits"] = fits
    json.dump(out, open("/workspace/data/calib.json", "w"), indent=1); print("CALIB_DONE")
