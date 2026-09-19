import json, numpy as np
rows = {}
for line in open("/workspace/logs/eval_results.jsonl"):
    d = json.loads(line); rows[d["tag"]] = d["mean"]
base = rows["LQ_baseline"]
higher = ["psnr", "ssim", "musiq", "clipiqa", "maniqa"]; lower = ["lpips", "dists", "niqe"]

# 方案A：典型比赛量纲归一化 (PSNR/50, MUSIQ/100, NIQE/10, 其余本身 0-1)，FR/NR 各占一半
def scoreA(m):
    fr = np.mean([m["psnr"] / 50, m["ssim"], 1 - m["lpips"], 1 - m["dists"]])
    nr = np.mean([m["musiq"] / 100, m["clipiqa"], m["maniqa"], 1 - m["niqe"] / 10])
    return 0.5 * fr + 0.5 * nr

# 方案B：相对基线提升，按所有变体中的极差归一化到 [0,1]（PSNR 微小差异会被放大）
allm = list(rows.values())
rng = {k: (min(m[k] for m in allm), max(m[k] for m in allm)) for k in higher + lower}
def scoreB(m):
    s = []
    for k in higher: s.append((m[k] - rng[k][0]) / (rng[k][1] - rng[k][0] + 1e-9))
    for k in lower: s.append((rng[k][1] - m[k]) / (rng[k][1] - rng[k][0] + 1e-9))
    return float(np.mean(s))

tab = [(t, scoreA(m), scoreB(m), m) for t, m in rows.items()]
ra = {t: i for i, (t, *_ ) in enumerate(sorted(tab, key=lambda x: -x[1]))}
rb = {t: i for i, (t, *_ ) in enumerate(sorted(tab, key=lambda x: -x[2]))}
tab.sort(key=lambda x: ra[x[0]] + rb[x[0]])
print(f"{'variant':28s} {'A':>6s} {'rkA':>4s} {'B':>6s} {'rkB':>4s} | psnr  ssim  lpips dists musiq maniqa niqe")
for t, a, b, m in tab:
    print(f"{t:28s} {a:6.4f} {ra[t]:4d} {b:6.4f} {rb[t]:4d} | {m['psnr']:5.2f} {m['ssim']:.3f} {m['lpips']:.3f} {m['dists']:.3f} {m['musiq']:5.1f} {m['maniqa']:.3f} {m['niqe']:.2f}")
