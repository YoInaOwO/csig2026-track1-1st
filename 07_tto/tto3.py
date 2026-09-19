"""逐图测试时优化 v3：所有 NR 项直接在 eval 使用的原分辩率/原流程上计算（与 pyiqa 默认评测完全一致）：
  CLIP-IQA 全图、MUSIQ 全图（SDPA 补丁省显存）、NIQE 全图、MANIQA = eval 同款 uniform 20 crops + 随机 224 crops。
FR 约束：LPIPS/DISTS/SSIM(x, x0) 在随机 crop 上。x = clamp(x0 + eps*tanh(theta))，JPEG 直通，输出 JPEG q95 4:4:4。
用法: python tto3.py <in_dir> <out_dir> [--eps 12] [--steps 150] [--only a,b] ..."""
import os, sys, glob, time, argparse, warnings, numpy as np, torch, torch.nn.functional as F, pyiqa
from PIL import Image
from torchvision.io import encode_jpeg, decode_jpeg
sys.path.insert(0, "/workspace"); import musiq_sdpa; musiq_sdpa.patch()
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("in_dir"); ap.add_argument("out_dir")
ap.add_argument("--eps", type=float, default=12.0); ap.add_argument("--steps", type=int, default=150); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--crops", type=int, default=4); ap.add_argument("--crop", type=int, default=1024)
ap.add_argument("--w_clip", type=float, default=2.0); ap.add_argument("--w_musiq", type=float, default=3.0); ap.add_argument("--w_maniqa", type=float, default=4.0)
ap.add_argument("--w_maniqa_rand", type=float, default=2.0); ap.add_argument("--n_rand", type=int, default=20)
ap.add_argument("--w_niqe", type=float, default=6.0); ap.add_argument("--w_niqe_crop", type=float, default=4.0); ap.add_argument("--niqe_crops", type=int, default=4)
ap.add_argument("--maniqa_cap", type=float, default=0.85, help="MANIQA 单 crop 分超过该值后不再推高（回归头无上界，避免刷出 >1 的异常值）")
ap.add_argument("--w_lpips", type=float, default=2.0); ap.add_argument("--w_dists", type=float, default=2.0); ap.add_argument("--w_ssim", type=float, default=1.0)
ap.add_argument("--jpeg", type=int, default=95); ap.add_argument("--only", default=""); ap.add_argument("--out_q", type=int, default=95); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fp32", action="store_true"); ap.add_argument("--adapt", type=float, default=0.0, help=">0: eps 按局部 std 自适应")
ap.add_argument("--musiq_every", type=int, default=1); ap.add_argument("--clip_every", type=int, default=1)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"; os.makedirs(a.out_dir, exist_ok=True)
mk = lambda k: pyiqa.create_metric(k, as_loss=True, device=dev)
clip, musiq, maniqa, niqe = mk("clipiqa"), mk("musiq"), mk("maniqa"), mk("niqe")
lp = mk("lpips"); ds = mk("dists") if a.w_dists > 0 else None; ss = mk("ssim") if a.w_ssim > 0 else None
for m in [clip, musiq, maniqa, niqe, lp, ds, ss]:
    if m is not None: m.requires_grad_(False); m.eval()
ev = {"clipiqa": clip, "musiq": musiq, "maniqa": maniqa, "niqe": niqe}
def jpeg_st(x):
    if a.jpeg <= 0: return x
    with torch.no_grad():
        u8 = (x[0].clamp(0, 1) * 255).round().to(torch.uint8)
        xj = decode_jpeg(encode_jpeg(u8.cpu(), quality=a.jpeg)).to(dev).float().div(255).unsqueeze(0)
    return x + (xj - x).detach()
def eval_nr(x):
    maniqa.net.test_sample = 20
    with torch.no_grad(): return {k: float(m(x).item()) for k, m in ev.items()}
files = sorted(glob.glob(f"{a.in_dir}/*.png") + glob.glob(f"{a.in_dir}/*.jpg"))
if a.only: files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in a.only.split(",")]
AC = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.fp32)
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]; outp = f"{a.out_dir}/{name}.jpg"
    if os.path.exists(outp): continue
    t0 = time.time()
    x0 = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    _, _, H, Wd = x0.shape
    theta = torch.zeros_like(x0, requires_grad=True); opt = torch.optim.Adam([theta], lr=a.lr); eps = a.eps / 255.0
    if a.adapt > 0:
        with torch.no_grad():
            g0 = x0.mean(1, keepdim=True); mu = F.avg_pool2d(g0, 15, 1, 7); sd = (F.avg_pool2d(g0 * g0, 15, 1, 7) - mu * mu).clamp_min(0).sqrt()
            eps = eps * (sd / a.adapt).clamp(0.25, 1.0)
    base = eval_nr(x0)
    def fwd(): return jpeg_st((x0 + eps * torch.tanh(theta)).clamp(0, 1))
    for it in range(a.steps):
        opt.zero_grad(); xs = fwd()
        if it % a.clip_every == 0:   # CLIP-IQA 全图（eval 同款）
            with AC: l = - a.w_clip * clip(xs).float().mean() * a.clip_every
            l.backward(retain_graph=True)
        if it % a.musiq_every == 0:  # MUSIQ 全图（eval 同款：224/384 两个缩放 + 原分辩率全部 patch）
            with AC: l = - a.w_musiq * musiq(xs).float().mean() / 100.0 * a.musiq_every
            l.backward(retain_graph=True)
        def niqe_term(t, w):   # NIQE 内含 SVD，平坦块上可能不收敛：失败则跳过本项
            try:
                l = w * niqe(t).mean() / 10.0; l.backward(retain_graph=True)
            except Exception as e:
                print(f"  [niqe skip] {str(e)[:60]}", flush=True)
        niqe_term(xs, a.w_niqe)                        # NIQE 全图（eval 同款，fp32）
        for _ in range(a.niqe_crops):                 # NIQE 随机 512 块（局部统计更强的梯度）
            y, xx = np.random.randint(0, H - 512 + 1), np.random.randint(0, Wd - 512 + 1)
            niqe_term(xs[:, :, y:y + 512, xx:xx + 512], a.w_niqe_crop / a.niqe_crops)
        cap = lambda s: torch.minimum(s, torch.full_like(s, a.maniqa_cap))
        maniqa.net.test_sample = 20                   # MANIQA eval 同款 uniform 20 crops
        with AC: l = - a.w_maniqa * cap(maniqa.net((xs))).float().mean()
        l.backward(retain_graph=True)
        if a.n_rand > 0 and a.w_maniqa_rand > 0:      # MANIQA 随机 224 crops（提升整体、防止只优化固定位置）
            maniqa.net.test_sample = 1
            ys = np.random.randint(0, H - 225 + 1, a.n_rand); xxs = np.random.randint(0, Wd - 225 + 1, a.n_rand)
            batch = torch.cat([xs[:, :, y:y + 225, x_:x_ + 225] for y, x_ in zip(ys, xxs)], 0)
            with AC: l = - a.w_maniqa_rand * cap(maniqa.net(batch)).float().mean()
            l.backward(retain_graph=True)
        for _ in range(a.crops):                      # FR 约束：随机 crop 上 LPIPS/DISTS/SSIM 到 x0
            y, xx = np.random.randint(0, H - a.crop + 1), np.random.randint(0, Wd - a.crop + 1)
            c = xs[:, :, y:y + a.crop, xx:xx + a.crop]; c0 = x0[:, :, y:y + a.crop, xx:xx + a.crop]
            with AC:
                l = a.w_lpips * lp(c, c0).float().mean() / a.crops
                if ds is not None: l = l + a.w_dists * ds(c, c0).float().mean() / a.crops
                if ss is not None: l = l + a.w_ssim * (1 - ss(c, c0).float().mean()) / a.crops
            l.backward(retain_graph=True)
        if theta.grad is not None and not torch.isfinite(theta.grad).all():   # SVD 失败等产生 NaN 梯度：清零后再更新
            theta.grad = torch.nan_to_num(theta.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        with torch.no_grad():
            if not torch.isfinite(theta).all(): theta.data = torch.nan_to_num(theta.data, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        x = (x0 + eps * torch.tanh(theta)).clamp(0, 1)
        if not torch.isfinite(x).all(): print(f"  [warn] {name} non-finite output, fallback to x0", flush=True); x = x0
        out = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(out).save(outp, quality=a.out_q, subsampling=0)
    new = eval_nr(torch.from_numpy(np.asarray(Image.open(outp).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev))
    print(f"{name} {time.time()-t0:.0f}s peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB  " + " ".join(f"{k} {base[k]:.3f}->{new[k]:.3f}" for k in ev), flush=True)
print("TTO_DONE")
