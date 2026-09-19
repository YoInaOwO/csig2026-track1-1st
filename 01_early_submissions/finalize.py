"""
最终后处理 + 导出：
  final = alpha * GaussianBlur(model_out, sigma) + (1 - alpha) * LQ  -> JPEG(q95, 4:4:4)
用法:
  python finalize.py --model_dir /workspace/output/test_odtsr_f1.0 --lq_dir /workspace/data/test --out_dir /workspace/submit/output_dir
  python finalize.py --model_dir /workspace/output/val_odtsr_f1.0 --lq_dir /workspace/data/val --out_dir /workspace/output/val_final --val
"""
import os, glob, argparse, re
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

def gauss_blur(x, sigma):
    if sigma <= 0: return x
    r = int(3 * sigma + 0.5); k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-k ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    x = F.conv2d(x, g.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return x

def to_t(im): return torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255).unsqueeze(0)
def to_im(t): return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--lq_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--val", action="store_true", help="LQ 文件名带 _lq 后缀")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    outs = sorted(glob.glob(f"{args.model_dir}/*.png") + glob.glob(f"{args.model_dir}/*.jpg"),
                  key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    bad = []
    for p in outs:
        name = os.path.splitext(os.path.basename(p))[0].replace("_lq", "")
        lq_p = f"{args.lq_dir}/{name}{'_lq' if args.val else ''}.jpg"
        lq = Image.open(lq_p).convert("RGB"); o = Image.open(p).convert("RGB")
        if lq.size != o.size:
            bad.append((name, lq.size, o.size)); continue
        x = args.alpha * gauss_blur(to_t(o).to(dev), args.sigma) + (1 - args.alpha) * to_t(lq).to(dev)
        out_p = f"{args.out_dir}/{name}.jpg"
        to_im(x).save(out_p, quality=args.quality, subsampling=0)
        # 校验
        chk = Image.open(out_p)
        assert chk.size == lq.size and chk.format == "JPEG", (name, chk.size, chk.format)
    print(f"written {len(outs) - len(bad)} files to {args.out_dir}; sigma={args.sigma} alpha={args.alpha} q={args.quality}")
    if bad: print("SIZE MISMATCH:", bad)

if __name__ == "__main__":
    main()
