"""验证集多变体 NR 指标电池 + 综合分（各指标在变体间 z-score 后平均，NIQE 取负）。
用法: python nr_eval.py tag1,tag2,...   （tag 对应 /workspace/output/val_<tag>/case{i}.png|jpg；'lq' 表示原图）"""
import sys, os, glob, json, warnings, numpy as np, torch, pyiqa
from PIL import Image
warnings.filterwarnings("ignore")
dev = "cuda"
tags = sys.argv[1].split(",")
NR = ["musiq", "clipiqa", "maniqa", "niqe", "topiq_nr", "dbcnn", "liqe", "hyperiqa"]
FR = ["psnr", "lpips"]
mets = {k: pyiqa.create_metric(k, device=dev) for k in NR + FR}
VAL = "/workspace/data/val"
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
def find(tag, i):
    if tag == "lq": return f"{VAL}/case{i}_lq.jpg"
    c = glob.glob(f"/workspace/output/val_{tag}/case{i}.*") + glob.glob(f"/workspace/output/val_{tag}/case{i}_lq.*")
    return c[0] if c else None
res = {}
for t in tags:
    rows = []
    for i in range(1, 6):
        p = find(t, i)
        if not p: continue
        x = load(p); g = load(f"{VAL}/case{i}_gt.jpg")
        r = {}
        with torch.no_grad():
            for k in NR: r[k] = float(mets[k](x).item())
            for k in FR: r[k] = float(mets[k](x, g).item())
        rows.append(r); torch.cuda.empty_cache()
    if rows:
        res[t] = {k: float(np.mean([r[k] for r in rows])) for k in NR + FR}
        res[t]["n"] = len(rows)
    print(t, res.get(t), flush=True)
# 综合分
keys = NR
M = np.array([[res[t][k] for k in keys] for t in res]); sign = np.array([-1 if k == "niqe" else 1 for k in keys])
Z = (M - M.mean(0)) / (M.std(0) + 1e-9) * sign
comp = Z.mean(1)
print("\n%-18s %6s %7s %7s %6s %7s %7s %6s %7s | %6s %6s | %6s" % ("tag", "musiq", "clipiqa", "maniqa", "niqe", "topiq", "dbcnn", "liqe", "hyper", "psnr", "lpips", "COMP"))
for (t, r), c in sorted(zip(res.items(), comp), key=lambda x: -x[1]):
    print("%-18s %6.2f %7.3f %7.3f %6.2f %7.3f %7.3f %6.2f %7.3f | %6.2f %6.3f | %+6.2f" % (t, r["musiq"], r["clipiqa"], r["maniqa"], r["niqe"], r["topiq_nr"], r["dbcnn"], r["liqe"], r["hyperiqa"], r["psnr"], r["lpips"], c))
json.dump(res, open("/workspace/output/nr_eval.json", "w"), indent=1)
print("NR_EVAL_DONE")
