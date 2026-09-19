import torch, pyiqa, time, warnings, sys
warnings.filterwarnings("ignore")
dev = "cuda"
ms = {k: pyiqa.create_metric(k, as_loss=True, device=dev) for k in ["clipiqa", "musiq", "niqe"]}
for m in ms.values(): m.requires_grad_(False); m.eval()
H, W = 3072, 4096
x0 = torch.rand(1, 3, H, W, device=dev); theta = torch.zeros_like(x0, requires_grad=True)
def run(name, fn, n=2, bf=True):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    try:
        for _ in range(n):
            x = (x0 + 0.03 * torch.tanh(theta)).clamp(0, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf): l = fn(x).float().mean()
            l.backward(); theta.grad = None
        torch.cuda.synchronize(); print(f"{name:22s} {(time.time()-t)/n*1000:7.0f} ms  peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB bf16={bf}", flush=True)
    except Exception as e:
        print(f"{name:22s} FAIL {str(e)[:80]}", flush=True); torch.cuda.empty_cache()
run("clipiqa full", lambda x: ms["clipiqa"](x))
run("musiq full", lambda x: ms["musiq"](x))
run("niqe full", lambda x: ms["niqe"](x), bf=False)
run("clipiqa full fp32", lambda x: ms["clipiqa"](x), bf=False)
run("musiq full fp32", lambda x: ms["musiq"](x), bf=False)
# eval 值 vs 缩放值
import torch.nn.functional as F
with torch.no_grad():
    from PIL import Image; import numpy as np
    x = torch.from_numpy(np.asarray(Image.open(sys.argv[1]).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    for k in ["clipiqa", "musiq"]:
        print(k, "full", float(ms[k](x)), "@1536", float(ms[k](F.interpolate(x, scale_factor=1536 / max(x.shape[-2:]), mode="area"))), "@768", float(ms[k](F.interpolate(x, scale_factor=768 / max(x.shape[-2:]), mode="area"))))
