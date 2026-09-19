"""逐图测试时优化（不使用 GT）：x = clamp(x0 + eps*tanh(theta))，最大化 NR 三项（CLIP-IQA/MUSIQ/MANIQA），
约束 LPIPS(x, x0) 小；JPEG q95 直通估计进循环。局部项在随机 crop 上算，全局项在下采样图上算。
用法: python tto.py <in_dir> <out_dir> [--eps 3] [--steps 80] [--lr 0.05] [--crops 4] [--crop 768] [--only case1,...]"""
import os, sys, glob, time, argparse, warnings, numpy as np, torch, torch.nn.functional as F, pyiqa
from PIL import Image
from torchvision.io import encode_jpeg, decode_jpeg
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("in_dir"); ap.add_argument("out_dir")
ap.add_argument("--eps", type=float, default=3.0); ap.add_argument("--steps", type=int, default=80); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--crops", type=int, default=4); ap.add_argument("--crop", type=int, default=768)
ap.add_argument("--w_clip", type=float, default=1.0); ap.add_argument("--w_musiq", type=float, default=1.0); ap.add_argument("--w_maniqa", type=float, default=1.0)
ap.add_argument("--w_lpips", type=float, default=2.0); ap.add_argument("--jpeg", type=int, default=95); ap.add_argument("--only", default="")
ap.add_argument("--global_side", type=int, default=768)
a = ap.parse_args()
dev = "cuda"; os.makedirs(a.out_dir, exist_ok=True)
M = {"clipiqa": pyiqa.create_metric("clipiqa", as_loss=True, device=dev), "musiq": pyiqa.create_metric("musiq", as_loss=True, device=dev),
     "maniqa": pyiqa.create_metric("maniqa", as_loss=True, device=dev)}
lp = pyiqa.create_metric("lpips", as_loss=True, device=dev)
for m in list(M.values()) + [lp]: m.requires_grad_(False); m.eval()
W = {"clipiqa": a.w_clip, "musiq": a.w_musiq, "maniqa": a.w_maniqa}
def jpeg_st(x):   # 直通 JPEG：前向经 q 压缩，梯度直通
    if a.jpeg <= 0: return x
    with torch.no_grad():
        u8 = (x[0].clamp(0, 1) * 255).round().to(torch.uint8)
        xj = decode_jpeg(encode_jpeg(u8.cpu(), quality=a.jpeg)).to(dev).float().div(255).unsqueeze(0)
    return x + (xj - x).detach()
def nr_terms(x):
    return sum(W[k] * (M[k](x).mean() / (100.0 if k == "musiq" else 1.0)) for k in M)
files = sorted(glob.glob(f"{a.in_dir}/*.png") + glob.glob(f"{a.in_dir}/*.jpg"))
if a.only: files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in a.only.split(",")]
for f in files:
    t0 = time.time()
    x0 = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    _, _, H, Wd = x0.shape
    theta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=a.lr)
    eps = a.eps / 255.0
    g0 = F.interpolate(x0, size=(a.global_side * H // max(H, Wd), a.global_side * Wd // max(H, Wd)), mode="area")
    with torch.no_grad():
        base_nr = {k: float(M[k](g0).item()) for k in M}
    for it in range(a.steps):
        x = (x0 + eps * torch.tanh(theta)).clamp(0, 1)
        xs = jpeg_st(x)
        loss = 0
        # 全局项（下采样视角：CLIP-IQA 与 MUSIQ 的小尺度分支）
        g = F.interpolate(xs, size=g0.shape[-2:], mode="area")
        loss = loss - (W["clipiqa"] * M["clipiqa"](g).mean() + W["musiq"] * M["musiq"](g).mean() / 100.0)
        # 局部项：随机 crop 上的 MUSIQ/MANIQA/CLIP-IQA + LPIPS 约束
        for _ in range(a.crops):
            y, xx = np.random.randint(0, H - a.crop + 1), np.random.randint(0, Wd - a.crop + 1)
            c = xs[:, :, y:y + a.crop, xx:xx + a.crop]; c0 = x0[:, :, y:y + a.crop, xx:xx + a.crop]
            loss = loss - nr_terms(c) / a.crops + a.w_lpips * lp(c, c0).mean() / a.crops
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = (x0 + eps * torch.tanh(theta)).clamp(0, 1)
        g = F.interpolate(x, size=g0.shape[-2:], mode="area"); new_nr = {k: float(M[k](g).item()) for k in M}
        out = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(out).save(f"{a.out_dir}/{os.path.splitext(os.path.basename(f))[0]}.png")
    print(f"{os.path.basename(f)} {time.time()-t0:.0f}s  global NR: " + " ".join(f"{k} {base_nr[k]:.3f}->{new_nr[k]:.3f}" for k in M), flush=True)
print("TTO_DONE")
