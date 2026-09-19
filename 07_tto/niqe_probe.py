import torch, pyiqa, warnings, time
warnings.filterwarnings("ignore")
dev = "cuda"
n = pyiqa.create_metric("niqe", as_loss=True, device=dev); n.requires_grad_(False); n.eval()
x = torch.rand(1, 3, 1024, 1024, device=dev) * 0.5 + 0.25
theta = torch.zeros_like(x, requires_grad=True)
try:
    t0 = time.time(); v = n((x + 4 / 255 * torch.tanh(theta)).clamp(0, 1)); print("niqe value", float(v), "requires_grad", v.requires_grad)
    v.backward(); g = theta.grad
    print("grad ok:", g is not None and torch.isfinite(g).all().item(), "abs mean", float(g.abs().mean()) if g is not None else None, f"{time.time()-t0:.1f}s")
    print("peak", torch.cuda.max_memory_allocated() / 1e9, "GB")
except Exception as e:
    print("NIQE_FAIL", repr(e)[:300])
