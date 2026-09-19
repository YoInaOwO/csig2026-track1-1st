"""
验证集评测：对 out_dir 下的 caseN.(jpg|png) 与 /workspace/data/val/caseN_gt.jpg 比较。
有参考: PSNR / SSIM / LPIPS / DISTS   无参考: MUSIQ / CLIPIQA / MANIQA / NIQE
用法: python eval.py <out_dir> [--tag name] [--no-nr]
      python eval.py lq            # 直接评估 LQ 本身作为基线
"""
import sys, os, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
import torch, numpy as np
from PIL import Image
import pyiqa

VAL = "/workspace/data/val"

def load(p):
    im = Image.open(p).convert("RGB")
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255).unsqueeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-nr", action="store_true", help="跳过无参考指标（快）")
    ap.add_argument("--json", default="/workspace/logs/eval_results.jsonl")
    args = ap.parse_args()
    dev = "cuda"

    fr = {k: pyiqa.create_metric(k, device=dev) for k in ["psnr", "ssim", "lpips", "dists"]}
    nr = {} if args.no_nr else {k: pyiqa.create_metric(k, device=dev) for k in ["musiq", "clipiqa", "maniqa", "niqe"]}

    rows = []
    for i in range(1, 6):
        gt_p = f"{VAL}/case{i}_gt.jpg"
        if args.out_dir == "lq":
            out_p = f"{VAL}/case{i}_lq.jpg"
        else:
            cands = glob.glob(f"{args.out_dir}/case{i}.*") + glob.glob(f"{args.out_dir}/case{i}_lq.*")
            if not cands:
                print(f"case{i}: missing output"); continue
            out_p = cands[0]
        gt, out = load(gt_p).to(dev), load(out_p).to(dev)
        if out.shape != gt.shape:
            print(f"case{i}: shape mismatch out={tuple(out.shape)} gt={tuple(gt.shape)}"); continue
        r = {"case": f"case{i}"}
        with torch.no_grad():
            for k, m in fr.items():
                r[k] = float(m(out, gt))
            for k, m in nr.items():
                r[k] = float(m(out))
        rows.append(r)
        print(f"case{i}: " + "  ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "case"), flush=True)

    if not rows:
        return
    keys = [k for k in rows[0] if k != "case"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    tag = args.tag or os.path.basename(args.out_dir.rstrip("/"))
    print(f"\n[{tag}] MEAN: " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))
    with open(args.json, "a") as f:
        f.write(json.dumps({"tag": tag, "dir": args.out_dir, "mean": mean, "rows": rows}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
