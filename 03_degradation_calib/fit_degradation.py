"""反推验证集 LQ 是如何从 GT 合成的：对每对图搜索 blur sigma / 下采样倍数 / 插值 / JPEG 质量 / 对比度变化，最大化与 LQ 的 PSNR"""
import io, itertools, numpy as np, cv2
from PIL import Image
VAL = "/workspace/data/val"

def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2); return 10 * np.log10(255 ** 2 / m) if m > 0 else 99

def degrade(gt, sigma, down, interp_down, interp_up, jpeg_q, sharpen):
    x = gt.copy()
    if sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    h, w = x.shape[:2]
    if down > 1:
        x = cv2.resize(x, (round(w / down), round(h / down)), interpolation=interp_down)
        x = cv2.resize(x, (w, h), interpolation=interp_up)
    if sharpen > 0:
        bl = cv2.GaussianBlur(x, (0, 0), 1.0); x = cv2.addWeighted(x, 1 + sharpen, bl, -sharpen, 0)
    if jpeg_q < 100:
        ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q]); x = cv2.imdecode(buf, 1)
    return x

INTERP = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}
for i in range(1, 6):
    gt = cv2.imread(f"{VAL}/case{i}_gt.jpg"); lq = cv2.imread(f"{VAL}/case{i}_lq.jpg")
    h, w = gt.shape[:2]; cy, cx = h // 2, w // 2; S = 768
    g = gt[cy - S:cy + S, cx - S:cx + S]; l = lq[cy - S:cy + S, cx - S:cx + S]
    # 亮度/对比度线性拟合 lq ≈ a*gt + b（对每通道）
    gm, lm = g.reshape(-1, 3).astype(np.float64), l.reshape(-1, 3).astype(np.float64)
    coef = [np.polyfit(gm[:, c], lm[:, c], 1) for c in range(3)]
    print(f"\ncase{i}: linear fit lq≈a*gt+b per BGR channel: " + " ".join(f"({a:.3f},{b:+.1f})" for a, b in coef) + f"  baseline PSNR(gt,lq)={psnr(g, l):.2f}")
    g_adj = np.clip(np.stack([np.polyval(coef[c], g[..., c].astype(np.float64)) for c in range(3)], -1), 0, 255).astype(np.uint8)
    print(f"   after linear color fit only: {psnr(g_adj, l):.2f}")
    best = []
    for sigma, down, idn, iup, jq, sh in itertools.product([0, 1, 1.5, 2, 3, 4, 6], [1, 2, 3, 4, 6], ["area"], ["cubic", "linear", "nearest"], [100, 90, 75], [0, 0.5, 1.0]):
        if down == 1 and iup != "cubic": continue
        d = degrade(g_adj, sigma, down, INTERP[idn], INTERP[iup], jq, sh)
        best.append((psnr(d, l), sigma, down, iup, jq, sh))
    best.sort(reverse=True)
    for b in best[:5]:
        print(f"   PSNR={b[0]:.2f}  sigma={b[1]} down={b[2]} up={b[3]} jpeg={b[4]} sharpen={b[5]}")
