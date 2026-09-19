"""逐图测试时优化 v2（不使用 GT）：x = clamp(x0 + eps*tanh(theta))，最大化 NR（CLIP-IQA/MUSIQ/MANIQA），LPIPS(x,x0) 约束。
全局项在两个尺度（长边 768 与 1536）上算 CLIP-IQA/MUSIQ；局部项在随机 crop 上算三项 + LPIPS；JPEG 直通估计。
输出 JPEG q95 (4:4:4)。用法: python tto2.py <in_dir> <out_dir> [--eps 4] [--steps 60] [--lr 0.05] [--crops 4] [--crop 768] [--only a,b]"""
import os, sys, glob, time, argparse, warnings, numpy as np, torch, torch.nn.functional as F, pyiqa
from PIL import Image
from torchvision.io import encode_jpeg, decode_jpeg
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("in_dir"); ap.add_argument("out_dir")
ap.add_argument("--eps", type=float, default=4.0); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--crops", type=int, default=4); ap.add_argument("--crop", type=int, default=768)
ap.add_argument("--w_clip", type=float, default=1.0); ap.add_argument("--w_musiq", type=float, default=1.0); ap.add_argument("--w_maniqa", type=float, default=1.0)
ap.add_argument("--w_lpips", type=float, default=2.0); ap.add_argument("--jpeg", type=int, default=95); ap.add_argument("--only", default="")
ap.add_argument("--globals", default="768,1536"); ap.add_argument("--out_q", type=int, default=95); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--w_dists", type=float, default=0.0); ap.add_argument("--w_niqe", type=float, default=0.0); ap.add_argument("--w_ssim", type=float, default=0.0)
ap.add_argument("--fp32", action="store_true"); ap.add_argument("--adapt", type=float, default=0.0, help=">0: eps 按局部 std 自适应，eps_map=eps*clamp(std/adapt, 0.25, 1)")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"; os.makedirs(a.out_dir, exist_ok=True)
M = {"clipiqa": pyiqa.create_metric("clipiqa", as_loss=True, device=dev), "musiq": pyiqa.create_metric("musiq", as_loss=True, device=dev),
     "maniqa": pyiqa.create_metric("maniqa", as_loss=True, device=dev)}
lp = pyiqa.create_metric("lpips", as_loss=True, device=dev)
ds = pyiqa.create_metric("dists", as_loss=True, device=dev) if a.w_dists > 0 else None
nq = pyiqa.create_metric("niqe", as_loss=True, device=dev) if a.w_niqe > 0 else None
ss = pyiqa.create_metric("ssim", as_loss=True, device=dev) if a.w_ssim > 0 else None
for m in list(M.values()) + [lp] + [z for z in (ds, nq, ss) if z is not None]: m.requires_grad_(False); m.eval()
M["maniqa"].net.test_sample = 6   # 省显存：默认 20 个 224 均匀块（512 crop 上 20 块带梯度约 19GB）
W = {"clipiqa": a.w_clip, "musiq": a.w_musiq, "maniqa": a.w_maniqa}
G = [int(s) for s in a.globals.split(",") if s]
def jpeg_st(x):
    if a.jpeg <= 0: return x
    with torch.no_grad():
        u8 = (x[0].clamp(0, 1) * 255).round().to(torch.uint8)
        xj = decode_jpeg(encode_jpeg(u8.cpu(), quality=a.jpeg)).to(dev).float().div(255).unsqueeze(0)
    return x + (xj - x).detach()
def rs(x, side, H, Wd): return F.interpolate(x, size=(side * H // max(H, Wd), side * Wd // max(H, Wd)), mode="area")
files = sorted(glob.glob(f"{a.in_dir}/*.png") + glob.glob(f"{a.in_dir}/*.jpg"))
if a.only: files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in a.only.split(",")]
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]; outp = f"{a.out_dir}/{name}.jpg"
    if os.path.exists(outp): continue
    t0 = time.time()
    x0 = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    _, _, H, Wd = x0.shape
    theta = torch.zeros_like(x0, requires_grad=True); opt = torch.optim.Adam([theta], lr=a.lr); eps = a.eps / 255.0
    if a.adapt > 0:   # 纹理自适应扰动幅度：局部 std（15x15）越小，允许的 eps 越小
        with torch.no_grad():
            g0 = x0.mean(1, keepdim=True); mu = F.avg_pool2d(g0, 15, 1, 7); sd = (F.avg_pool2d(g0 * g0, 15, 1, 7) - mu * mu).clamp_min(0).sqrt()
            eps = eps * (sd / a.adapt).clamp(0.25, 1.0)
    with torch.no_grad(): base_nr = {k: float(M[k](rs(x0, G[-1], H, Wd)).item()) for k in M}
    def fwd():   # 每一项单独 backward，避免多项计算图同时驻留显存
        x = (x0 + eps * torch.tanh(theta)).clamp(0, 1); return jpeg_st(x)
    AC = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.fp32)
    for it in range(a.steps):
        opt.zero_grad()
        xs = fwd()   # 每步只做一次全图前向 + JPEG 直通；其逐元素计算图很小，各项 backward 时 retain
        for side in G:   # 全局项：CLIP-IQA 多尺度；MUSIQ 只在最小尺度
            g = rs(xs, side, H, Wd)
            with AC:
                l = - W["clipiqa"] * M["clipiqa"](g).float().mean() / len(G)
                if side == G[0]: l = l - W["musiq"] * M["musiq"](g).float().mean() / 100.0
            l.backward(retain_graph=True)
        for _ in range(a.crops):   # 局部项：逐项 backward，避免多项图同时驻留
            y, xx = np.random.randint(0, H - a.crop + 1), np.random.randint(0, Wd - a.crop + 1)
            c = xs[:, :, y:y + a.crop, xx:xx + a.crop]; c0 = x0[:, :, y:y + a.crop, xx:xx + a.crop]
            terms = []
            for k in M: terms.append(lambda k=k: - W[k] * (M[k](c).float().mean() / (100.0 if k == "musiq" else 1.0)))
            terms.append(lambda: a.w_lpips * lp(c, c0).float().mean())
            if ds is not None: terms.append(lambda: a.w_dists * ds(c, c0).float().mean())
            if ss is not None: terms.append(lambda: a.w_ssim * (1 - ss(c, c0).float().mean()))
            for fn in terms:
                with AC: l = fn() / a.crops
                l.backward(retain_graph=True)
            if nq is not None:   # NIQE 含 SVD 保持 fp32；在 512 子块上算（块统计量，代表性足够）
                h2 = min(512, a.crop); yy2, xx2 = np.random.randint(0, a.crop - h2 + 1), np.random.randint(0, a.crop - h2 + 1)
                l = a.w_niqe * nq(c[:, :, yy2:yy2 + h2, xx2:xx2 + h2]).mean() / 10.0 / a.crops
                l.backward(retain_graph=True)
        opt.step()
    with torch.no_grad():
        x = (x0 + eps * torch.tanh(theta)).clamp(0, 1); new_nr = {k: float(M[k](rs(x, G[-1], H, Wd)).item()) for k in M}
        out = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(out).save(outp, quality=a.out_q, subsampling=0)
    print(f"{name} {time.time()-t0:.0f}s peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB reserved={torch.cuda.memory_reserved()/1e9:.1f}GB  NR@{G[-1]}: " + " ".join(f"{k} {base_nr[k]:.3f}->{new_nr[k]:.3f}" for k in M), flush=True)
print("TTO_DONE")
