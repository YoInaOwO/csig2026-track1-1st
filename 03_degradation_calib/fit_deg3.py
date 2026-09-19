"""扩大模糊范围，按分块局部线性拟合的平均 PSNR 评估退化模型的贴合度"""
import numpy as np, cv2, itertools
from concurrent.futures import ProcessPoolExecutor
VAL = "/workspace/data/val"
I = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}

def degrade(g, pre, f, di, ui, post):
    h, w = g.shape[:2]; x = g
    if pre > 0: x = cv2.GaussianBlur(x, (0, 0), pre)
    if f > 1:
        x = cv2.resize(x, (round(w / f), round(h / f)), interpolation=I[di]); x = cv2.resize(x, (w, h), interpolation=I[ui])
    if post > 0: x = cv2.GaussianBlur(x, (0, 0), post)
    return x

def local_psnr(bg, lq, B=384):
    h, w = bg.shape[:2]; ps = []
    for by in range(h // B):
        for bx in range(w // B):
            g = bg[by * B:(by + 1) * B, bx * B:(bx + 1) * B].ravel(); l = lq[by * B:(by + 1) * B, bx * B:(bx + 1) * B].ravel()
            if g.std() < 3: continue
            a, b = np.polyfit(g, l, 1); ps.append(10 * np.log10(255 ** 2 / np.mean((a * g + b - l) ** 2)))
    return float(np.mean(ps))

def run(i):
    gt = cv2.cvtColor(cv2.imread(f"{VAL}/case{i}_gt.jpg"), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lq = cv2.cvtColor(cv2.imread(f"{VAL}/case{i}_lq.jpg"), cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = []
    for pre, f, di, ui, post in itertools.product([0, 2, 3, 4, 5, 6, 8], [1, 2, 4, 6, 8, 12, 16], ["area", "linear"], ["linear", "cubic"], [0, 1.5, 3]):
        if f == 1 and (di != "area" or ui != "linear"): continue
        res.append((local_psnr(degrade(gt, pre, f, di, ui, post), lq), pre, f, di, ui, post))
    res.sort(reverse=True)
    out = [f"case{i}:"] + [f"   localPSNR={r[0]:.2f} pre={r[1]} down={r[2]} {r[3]} up={r[4]} post={r[5]}" for r in res[:6]]
    # 等效 sigma 估计：把最优组合施加到脉冲上量 sigma
    b = res[0]; imp = np.zeros((1024, 1024), np.float32); imp[512, 512] = 1e4
    k = degrade(imp, b[1], b[2], b[3], b[4], b[5]); ys, xs = np.mgrid[:1024, :1024]; k = k / k.sum()
    sig = np.sqrt(((k * ((xs - 512) ** 2 + (ys - 512) ** 2)).sum()) / 2)
    out.append(f"   -> best 等效 sigma ≈ {sig:.2f} px")
    return "\n".join(out)

if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex:
        for o in ex.map(run, range(1, 6)): print(o, flush=True)
