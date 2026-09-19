"""分块局部线性拟合 lq ≈ a*blur(gt)+b，看 a/b 是否随空间变化（雾/低频对比度场）"""
import numpy as np, cv2
VAL = "/workspace/data/val"
def blur_down(g, pre=3.0, f=4):
    h, w = g.shape[:2]
    x = cv2.GaussianBlur(g, (0, 0), pre)
    x = cv2.resize(x, (w // f, h // f), interpolation=cv2.INTER_AREA)
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
for i in range(1, 6):
    gt = cv2.imread(f"{VAL}/case{i}_gt.jpg").astype(np.float32); lq = cv2.imread(f"{VAL}/case{i}_lq.jpg").astype(np.float32)
    bg = blur_down(gt); h, w = gt.shape[:2]; B = 384
    A = np.zeros((h // B, w // B)); Bm = np.zeros_like(A); P = np.zeros_like(A)
    for by in range(h // B):
        for bx in range(w // B):
            g = cv2.cvtColor(bg[by * B:(by + 1) * B, bx * B:(bx + 1) * B], cv2.COLOR_BGR2GRAY).ravel()
            l = cv2.cvtColor(lq[by * B:(by + 1) * B, bx * B:(bx + 1) * B], cv2.COLOR_BGR2GRAY).ravel()
            if g.std() < 3: A[by, bx] = np.nan; Bm[by, bx] = np.nan; P[by, bx] = np.nan; continue
            a, b = np.polyfit(g, l, 1); A[by, bx] = a; Bm[by, bx] = b
            P[by, bx] = 10 * np.log10(255 ** 2 / np.mean((a * g + b - l) ** 2))
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print(f"\n=== case{i} ===  a: mean={np.nanmean(A):.2f} std={np.nanstd(A):.2f} | b: mean={np.nanmean(Bm):.1f} std={np.nanstd(Bm):.1f} | local PSNR mean={np.nanmean(P):.1f}")
    print("a map:\n", A); print("b map:\n", Bm)
    # 检验 b 与 a 是否满足雾模型 b = A0*(1-a)（同一 A0）：拟合 A0
    m = ~np.isnan(A) & (A < 0.98)
    if m.sum() > 3:
        A0 = np.polyfit(1 - A[m], Bm[m], 1)
        print(f"haze check: b ≈ {A0[0]:.1f}*(1-a) + {A0[1]:.1f}   (若截距≈0 且斜率≈亮度 → 雾模型 lq = a*gt + (1-a)*A0)")
