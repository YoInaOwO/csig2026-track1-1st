"""从对齐的 LQ/GT 对估计真实退化核：先做线性亮度拟合 lq≈a*gt+b，再频域最小二乘解 k 使 (k*gt') ≈ lq。
输出每张图的核（npy + 可视化）、核的等效 sigma、残差 PSNR；并检查残差是否还有结构（如锯齿/JPEG）。"""
import numpy as np, cv2, os
from PIL import Image
VAL = "/workspace/data/val"; OUT = "/workspace/output/kernels"; os.makedirs(OUT, exist_ok=True)
KS = 41  # 核尺寸

def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2); return 10 * np.log10(255 ** 2 / m)

def estimate_kernel(gt, lq, lam=1e-2):
    """gt, lq: float32 灰度 HxW，同尺寸。返回 KS x KS 核。"""
    H, W = gt.shape
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    g = (gt - gt.mean()) * win; l = (lq - lq.mean()) * win
    G = np.fft.fft2(g); L = np.fft.fft2(l)
    K = np.conj(G) * L / (np.abs(G) ** 2 + lam * np.mean(np.abs(G) ** 2))
    k = np.real(np.fft.ifft2(K)); k = np.fft.fftshift(k)
    cy, cx = H // 2, W // 2; r = KS // 2
    k = k[cy - r:cy + r + 1, cx - r:cx + r + 1]
    k = np.clip(k, 0, None); k /= k.sum()
    return k

def kernel_sigma(k):
    r = KS // 2; ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    return float(np.sqrt((k * (xs ** 2 + ys ** 2)).sum() / 2))

summary = []
for i in range(1, 6):
    gt = np.asarray(Image.open(f"{VAL}/case{i}_gt.jpg").convert("RGB")).astype(np.float32)
    lq = np.asarray(Image.open(f"{VAL}/case{i}_lq.jpg").convert("RGB")).astype(np.float32)
    # 1) 线性亮度/对比度拟合（逐通道）
    coef = [np.polyfit(gt[..., c].ravel()[::37], lq[..., c].ravel()[::37], 1) for c in range(3)]
    gt_adj = np.stack([np.polyval(coef[c], gt[..., c]) for c in range(3)], -1)
    # 2) 在多个 1024 区域估核取平均（避免单区域偏差）
    H, W = gt.shape[:2]; ks = []
    for (y, x) in [(H // 2 - 512, W // 2 - 512), (H // 4, W // 4), (3 * H // 4 - 1024, 3 * W // 4 - 1024), (H // 4, 3 * W // 4 - 1024), (3 * H // 4 - 1024, W // 4)]:
        y = max(0, min(y, H - 1024)); x = max(0, min(x, W - 1024))
        gg = cv2.cvtColor(gt_adj[y:y + 1024, x:x + 1024].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        ll = cv2.cvtColor(lq[y:y + 1024, x:x + 1024].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        if gg.std() < 5: continue
        ks.append(estimate_kernel(gg, ll))
    k = np.mean(ks, 0); k /= k.sum()
    np.save(f"{OUT}/case{i}_kernel.npy", k)
    # 3) 用估计核重建 LQ，看残差
    rec = np.stack([cv2.filter2D(gt_adj[..., c], -1, k, borderType=cv2.BORDER_REFLECT) for c in range(3)], -1)
    rec = np.clip(rec, 0, 255)
    p_lin = psnr(np.clip(gt_adj, 0, 255), lq); p_rec = psnr(rec, lq)
    # 残差图与噪声估计
    resid = lq - rec
    sig = kernel_sigma(k)
    # 核的各向异性：二阶矩
    r = KS // 2; ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    cov = np.array([[(k * xs * xs).sum(), (k * xs * ys).sum()], [(k * xs * ys).sum(), (k * ys * ys).sum()]])
    ev = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0))
    print(f"case{i}: color a={[round(c[0],3) for c in coef]} b={[round(c[1],1) for c in coef]} | PSNR linear-only={p_lin:.2f}  kernel={p_rec:.2f} | eq.sigma={sig:.2f} axes=({ev[0]:.2f},{ev[1]:.2f}) | resid std={resid.std():.2f}")
    summary.append((i, coef, k, p_rec, sig))
    # 可视化核（放大 8 倍）与中心行剖面
    kv = (k / k.max() * 255).astype(np.uint8); kv = cv2.resize(kv, (KS * 8, KS * 8), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{OUT}/case{i}_kernel.png", kv)
    prof = k[KS // 2]; print("   center row profile:", " ".join(f"{v:.3f}" for v in prof[KS // 2 - 8:KS // 2 + 9]))

# 与高斯核对比：每个估计核的最佳高斯拟合 sigma 与拟合误差
print("\n== Gaussian fit of estimated kernels ==")
for i, coef, k, p_rec, sig in summary:
    best = None
    for s in np.arange(0.5, 8.0, 0.1):
        g = cv2.getGaussianKernel(KS, s); g = g @ g.T
        err = np.abs(g - k).sum()
        if best is None or err < best[1]: best = (s, err)
    print(f"case{i}: best gaussian sigma={best[0]:.1f}  L1 err={best[1]:.3f} (0=完美高斯)")
# 拼所有核
canvas = np.concatenate([cv2.imread(f"{OUT}/case{i}_kernel.png") for i in range(1, 6)], 1)
cv2.imwrite(f"{OUT}/all_kernels.png", canvas)
print("KERNEL_DONE")
