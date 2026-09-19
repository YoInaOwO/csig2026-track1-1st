import torch, pyiqa, torch.nn.functional as F, numpy as np, warnings
warnings.filterwarnings("ignore")
dev = "cuda"
x0 = torch.rand(1, 3, 3072, 4096, device=dev)
theta = torch.zeros_like(x0, requires_grad=True)
def probe(name, fn):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    x = (x0 + 4 / 255 * torch.tanh(theta)).clamp(0, 1)
    loss = fn(x); loss.backward(); theta.grad = None
    print(f"{name:28s} peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
def rs(x, side): H, W = x.shape[-2:]; return F.interpolate(x, size=(side * H // max(H, W), side * W // max(H, W)), mode="area")
clip = pyiqa.create_metric("clipiqa", as_loss=True, device=dev); musiq = pyiqa.create_metric("musiq", as_loss=True, device=dev)
maniqa = pyiqa.create_metric("maniqa", as_loss=True, device=dev); lp = pyiqa.create_metric("lpips", as_loss=True, device=dev)
print("maniqa num_crops:", getattr(maniqa.net, "num_crops", None), "clip backbone:", type(clip.net).__name__)
probe("identity(x.mean)", lambda x: x.mean())
probe("clipiqa@768", lambda x: clip(rs(x, 768)).mean())
probe("clipiqa@1536", lambda x: clip(rs(x, 1536)).mean())
probe("musiq@768", lambda x: musiq(rs(x, 768)).mean())
probe("maniqa crop512", lambda x: maniqa(x[:, :, :512, :512]).mean())
probe("musiq crop512", lambda x: musiq(x[:, :, :512, :512]).mean())
probe("clipiqa crop512", lambda x: clip(x[:, :, :512, :512]).mean())
probe("lpips crop512", lambda x: lp(x[:, :, :512, :512], x0[:, :, :512, :512]).mean())
