import numpy as np, itertools
names = ["LQ", "final", "fuse", "raw", "cfg2"]
S = np.array([2.1805, 3.0675, 3.4290, 3.6235, 3.7106])
# 验证集 FR（代理）
F = {"psnr": [28.03, 27.81, 27.51, 26.29, 22.23], "ssim": [.803, .800, .794, .767, .686],
     "lpips": [.357, .304, .326, .351, .419], "dists": [.279, .213, .228, .239, .257]}
# 测试抽样 NR
N = {"musiq": [30.97, 49.05, 56.26, 60.89, 67.11], "clipiqa": [.202, .400, .544, .632, .671],
     "maniqa": [.205, .268, .318, .362, .405], "niqe": [8.50, 6.12, 4.92, 4.64, 3.68],
     "topiq": [.207, .338, .448, .507, .590], "dbcnn": [.262, .334, .450, .513, .618], "liqe": [1.09, 1.86, 2.45, 2.85, 3.24]}
# 统一到"越大越好"的 0-1 尺度（常见比赛做法）
norm = {"psnr": lambda v: np.array(v) / 50, "ssim": lambda v: np.array(v), "lpips": lambda v: 1 - np.array(v), "dists": lambda v: 1 - np.array(v),
        "musiq": lambda v: np.array(v) / 100, "clipiqa": lambda v: np.array(v), "maniqa": lambda v: np.array(v), "niqe": lambda v: 1 - np.array(v) / 10,
        "topiq": lambda v: np.array(v), "dbcnn": lambda v: np.array(v), "liqe": lambda v: (np.array(v) - 1) / 4}
X = {k: norm[k](v) for k, v in {**F, **N}.items()}

def fit(feats, intercept=True):
    A = np.stack([X[f] for f in feats], 1)
    if intercept: A = np.c_[A, np.ones(5)]
    w, *_ = np.linalg.lstsq(A, S, rcond=None)
    r = S - A @ w
    # 留一交叉验证
    loo = []
    for i in range(5):
        m = np.arange(5) != i
        wi, *_ = np.linalg.lstsq(A[m], S[m], rcond=None); loo.append(S[i] - A[i] @ wi)
    return w, np.sqrt(np.mean(r ** 2)), np.sqrt(np.mean(np.array(loo) ** 2))

print("== 单指标 + 截距 ==")
for f in X:
    w, rm, lo = fit([f]); print(f"{f:8s} w={w[0]:7.2f} b={w[1]:6.2f}  rmse={rm:.3f} loo={lo:.3f}")
print("\n== 双指标 + 截距（按 loo 排序，前 15）==")
res = []
for a, b in itertools.combinations(X, 2):
    w, rm, lo = fit([a, b]); res.append((lo, rm, a, b, w))
for lo, rm, a, b, w in sorted(res)[:15]:
    print(f"{a:8s}+{b:8s} w=({w[0]:6.2f},{w[1]:6.2f}) b={w[2]:6.2f} rmse={rm:.3f} loo={lo:.3f}")
print("\n== FR 均值 + NR 均值 ==")
FRm = np.mean([X[k] for k in F], 0); NRm = np.mean([X[k] for k in ["musiq", "clipiqa", "maniqa", "niqe"]], 0)
X["FRm"], X["NRm"] = FRm, NRm
w, rm, lo = fit(["FRm", "NRm"]); print(f"FRm,NRm w=({w[0]:.2f},{w[1]:.2f}) b={w[2]:.2f} rmse={rm:.3f} loo={lo:.3f}")
print("FRm:", FRm.round(3), "NRm:", NRm.round(3))
print("\n== 逐步增量（分数差 / 指标差）==")
for k in ["musiq", "clipiqa", "maniqa", "niqe", "topiq", "liqe", "psnr", "lpips", "dists"]:
    v = np.array({**F, **N}[k]); d = np.diff(v); print(f"{k:8s} " + " ".join(f"{s/dd:7.3f}" for s, dd in zip(np.diff(S), d)))

