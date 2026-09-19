"""
后处理变体扫描（无需再跑模型）：
  blend:  a*ODTSR + (1-a)*LQ
  blur:   GaussianBlur(ODTSR, sigma)
  ens:    a*ODTSR + b*FiDeSR + c*LQ
  blur+blend 组合
"""
import os, glob, json, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import pyiqa

VAL = "/workspace/data/val"
def load(p):
    return torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0)

def gauss_blur(x, sigma):
    if sigma <= 0: return x
    r = int(3 * sigma + 0.5); k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-k ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    x = F.conv2d(x, g.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return x

VARIANTS_V1 = {
    "blur0.5":            lambda o, f, l: gauss_blur(o, 0.5),
    "blur0.8":            lambda o, f, l: gauss_blur(o, 0.8),
    "blur1.0":            lambda o, f, l: gauss_blur(o, 1.0),
    "blend0.5":           lambda o, f, l: 0.5 * o + 0.5 * l,
    "blur0.5_blend0.6":   lambda o, f, l: 0.6 * gauss_blur(o, 0.5) + 0.4 * l,
    "blur0.8_blend0.7":   lambda o, f, l: 0.7 * gauss_blur(o, 0.8) + 0.3 * l,
    "ens_o.5_f.5":        lambda o, f, l: 0.5 * o + 0.5 * f,
    "ens_o.4_f.3_l.3":    lambda o, f, l: 0.4 * o + 0.3 * f + 0.3 * l,
    "ens_o.35_f.35_l.3":  lambda o, f, l: 0.35 * o + 0.35 * f + 0.3 * l,
}
VARIANTS_V2 = {
    "blur0.65":               lambda o, f, l: gauss_blur(o, 0.65),
    "blur0.8_blend0.85":      lambda o, f, l: 0.85 * gauss_blur(o, 0.8) + 0.15 * l,
    "blur1.0_blend0.8":       lambda o, f, l: 0.8 * gauss_blur(o, 1.0) + 0.2 * l,
    "ensB_o.5_f.5":           lambda o, f, l: 0.5 * gauss_blur(o, 0.8) + 0.5 * gauss_blur(f, 0.8),
    "ensB_o.6_f.4":           lambda o, f, l: 0.6 * gauss_blur(o, 0.8) + 0.4 * gauss_blur(f, 0.8),
    "ensB_o.45_f.3_l.25":     lambda o, f, l: 0.45 * gauss_blur(o, 0.8) + 0.3 * gauss_blur(f, 0.8) + 0.25 * l,
    "ensB_o.5_f.3_l.2":       lambda o, f, l: 0.5 * gauss_blur(o, 0.8) + 0.3 * gauss_blur(f, 0.8) + 0.2 * l,
    "ensB_o.55_f.3_l.15":     lambda o, f, l: 0.55 * gauss_blur(o, 0.8) + 0.3 * gauss_blur(f, 0.8) + 0.15 * l,
}
VARIANTS = {**VARIANTS_V1, **VARIANTS_V2}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odtsr", default="/workspace/output/val_odtsr_f1.0")
    ap.add_argument("--fidesr", default="/workspace/output/val_fidesr_s1")
    ap.add_argument("--only", default=None)
    ap.add_argument("--set", default="all", choices=["all", "v1", "v2"])
    ap.add_argument("--percase", action="store_true")
    ap.add_argument("--json", default="/workspace/logs/eval_results.jsonl")
    args = ap.parse_args()
    dev = "cuda"
    fr = {k: pyiqa.create_metric(k, device=dev) for k in ["psnr", "ssim", "lpips", "dists"]}
    nr = {k: pyiqa.create_metric(k, device=dev) for k in ["musiq", "clipiqa", "maniqa", "niqe"]}
    pool = {"all": VARIANTS, "v1": VARIANTS_V1, "v2": VARIANTS_V2}[args.set]
    variants = {k: v for k, v in pool.items() if not args.only or k in args.only.split(",")}
    res = {k: [] for k in variants}
    for i in range(1, 6):
        gt = load(f"{VAL}/case{i}_gt.jpg").to(dev)
        lq = load(f"{VAL}/case{i}_lq.jpg").to(dev)
        o = load(glob.glob(f"{args.odtsr}/case{i}.*")[0]).to(dev)
        f = load((glob.glob(f"{args.fidesr}/case{i}.*") + glob.glob(f"{args.fidesr}/case{i}_lq.*"))[0]).to(dev)
        for name, fn in variants.items():
            x = fn(o, f, lq).clamp(0, 1)
            r = {"case": f"case{i}"}
            with torch.no_grad():
                for k, m in fr.items(): r[k] = float(m(x, gt))
                for k, m in nr.items(): r[k] = float(m(x))
            res[name].append(r)
            if args.percase:
                print(f"  [{name}] case{i}: " + "  ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "case"), flush=True)
        print(f"case{i} done", flush=True)
    for name, rows in res.items():
        keys = [k for k in rows[0] if k != "case"]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        print(f"[{name}] MEAN: " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))
        with open(args.json, "a") as fh:
            fh.write(json.dumps({"tag": "pp_" + name, "mean": mean, "rows": rows}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
