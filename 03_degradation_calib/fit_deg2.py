"""宽范围暴力搜索退化流程：pre_blur -> down(interp) -> JPEG(低分辩率) -> up(interp) -> post_blur，再线性拟合亮度。"""
import itertools, numpy as np, cv2, sys, time
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
VAL = "/workspace/data/val"
I = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4, "nearest": cv2.INTER_NEAREST}

def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2); return 10 * np.log10(255 ** 2 / m) if m > 0 else 99

def degrade(g, pre, f, di, jq, ui, post):
    x = g
    if pre > 0: x = cv2.GaussianBlur(x, (0, 0), pre)
    h, w = x.shape[:2]
    if f > 1:
        x = cv2.resize(x, (round(w / f), round(h / f)), interpolation=I[di])
        if jq < 100:
            ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, jq]); x = cv2.imdecode(buf, 1)
        x = cv2.resize(x, (w, h), interpolation=I[ui])
    elif jq < 100:
        ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, jq]); x = cv2.imdecode(buf, 1)
    if post > 0: x = cv2.GaussianBlur(x, (0, 0), post)
    return x

def color_fit(x, l):
    """逐通道线性拟合 l ≈ a*x+b，返回拟合后的 x"""
    out = np.empty_like(x, dtype=np.float32)
    for c in range(3):
        a, b = np.polyfit(x[..., c].ravel()[::7].astype(np.float64), l[..., c].ravel()[::7].astype(np.float64), 1)
        out[..., c] = a * x[..., c] + b
    return np.clip(out, 0, 255).astype(np.uint8)

def run_case(i):
    gt = cv2.imread(f"{VAL}/case{i}_gt.jpg"); lq = cv2.imread(f"{VAL}/case{i}_lq.jpg")
    h, w = gt.shape[:2]
    # 选一个纹理丰富的 1024 区域（梯度能量最大的块）
    best_blk, best_e = None, -1
    for y in range(0, h - 1024, 512):
        for x in range(0, w - 1024, 512):
            e = cv2.Laplacian(cv2.cvtColor(gt[y:y + 1024, x:x + 1024], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if e > best_e: best_e, best_blk = e, (y, x)
    y, x = best_blk; g = gt[y:y + 1024, x:x + 1024]; l = lq[y:y + 1024, x:x + 1024]
    res = []
    for pre, f, di, jq, ui, post in itertools.product([0, 0.7, 1.2, 2.0, 3.0], [1, 1.5, 2, 3, 4, 5, 6, 8], ["area", "linear", "cubic", "nearest"], [100, 85, 60, 40], ["linear", "cubic", "lanczos", "nearest"], [0, 0.8, 1.5]):
        if f == 1 and (di != "area" or ui != "linear"): continue
        d = degrade(g, pre, f, di, jq, ui, post)
        d = color_fit(d, l)
        res.append((psnr(d, l), pre, f, di, jq, ui, post))
    res.sort(reverse=True)
    base = psnr(color_fit(g, l), l)
    lines = [f"case{i} block=({y},{x}) baseline(color only)={base:.2f}"]
    for r in res[:8]:
        lines.append(f"   PSNR={r[0]:.2f}  pre={r[1]} down={r[2]} {r[3]} jpeg={r[4]} up={r[5]} post={r[6]}")
    return "\n".join(lines)

if __name__ == "__main__":
    t = time.time()
    with ProcessPoolExecutor(5) as ex:
        for out in ex.map(run_case, range(1, 6)): print(out, flush=True)
    print(f"done {time.time()-t:.0f}s")
