"""通用评测：--val_dir 下 case{i}_gt.* 为 GT，out_dir 下 case{i}.* 为输出。写 eval_results.jsonl（tag 前缀区分集合）。
用法: python eval2.py <out_dir> --tag T [--val_dir /workspace/data/sval] [--json ...]
      python eval2.py lq --tag sval_LQ --val_dir /workspace/data/sval"""
import os, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
import torch, numpy as np, pyiqa
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--tag", required=True)
ap.add_argument("--val_dir", default="/workspace/data/sval"); ap.add_argument("--json", default="/workspace/logs/eval_results.jsonl")
a = ap.parse_args(); dev = "cuda"
def load(p): return torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
fr = {k: pyiqa.create_metric(k, device=dev) for k in ["psnr", "ssim", "lpips", "dists"]}
nr = {k: pyiqa.create_metric(k, device=dev) for k in ["musiq", "clipiqa", "maniqa", "niqe"]}
gts = sorted(glob.glob(f"{a.val_dir}/case*_gt.*"), key=lambda p: int(os.path.basename(p).split("_")[0][4:]))
rows = []
for gp in gts:
    c = os.path.basename(gp).split("_")[0]
    op = glob.glob(f"{a.val_dir}/{c}_lq.*")[0] if a.out_dir == "lq" else (glob.glob(f"{a.out_dir}/{c}.*") + glob.glob(f"{a.out_dir}/{c}_lq.*"))
    if isinstance(op, list):
        if not op: print(c, "missing"); continue
        op = op[0]
    gt, out = load(gp), load(op)
    if out.shape != gt.shape: print(c, "shape mismatch", out.shape, gt.shape); continue
    r = {"case": c}
    with torch.no_grad():
        for k, m in fr.items(): r[k] = float(m(out, gt))
        for k, m in nr.items(): r[k] = float(m(out))
    rows.append(r)
keys = [k for k in rows[0] if k != "case"]
mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
FR = (mean["psnr"] / 50 + mean["ssim"] + 1 - mean["lpips"] + 1 - mean["dists"]) / 4
NR = (mean["musiq"] / 100 + mean["clipiqa"] + mean["maniqa"] + 1 - mean["niqe"] / 10) / 4
print(f"[{a.tag}] n={len(rows)} MEAN: " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()) + f"  | FRm={FR:.3f} NRm={NR:.3f} PRED={-0.86 + 3.02 * FR + 4.57 * NR:.3f}")
with open(a.json, "a") as f: f.write(json.dumps({"tag": a.tag, "dir": a.out_dir, "mean": mean, "rows": rows}, ensure_ascii=False) + "\n")
