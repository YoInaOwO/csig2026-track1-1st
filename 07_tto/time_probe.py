import torch, pyiqa, time, warnings, torch.nn.functional as F
warnings.filterwarnings("ignore")
dev = "cuda"
ms = {k: pyiqa.create_metric(k, as_loss=True, device=dev) for k in ["clipiqa", "musiq", "maniqa", "lpips", "dists", "niqe", "ssim"]}
ms["maniqa"].net.test_sample = 6
for m in ms.values(): m.requires_grad_(False); m.eval()
x0 = torch.rand(1, 3, 3072, 4096, device=dev); theta = torch.zeros_like(x0, requires_grad=True)
def run(name, fn, n=3, bf=False):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        x = (x0 + 0.03 * torch.tanh(theta)).clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf): l = fn(x).float().mean()
        l.backward(); theta.grad = None
    torch.cuda.synchronize(); print(f"{name:22s} {(time.time()-t)/n*1000:7.0f} ms  bf16={bf}", flush=True)
c = lambda x: x[:, :, :1024, :1024]
g768 = lambda x: F.interpolate(x, size=(576, 768), mode="area"); g1536 = lambda x: F.interpolate(x, size=(1152, 1536), mode="area")
run("fwd only (tanh+clamp)", lambda x: x[:, :, :8, :8])
for bf in (False, True):
    run("clip@768", lambda x: ms["clipiqa"](g768(x)), bf=bf); run("clip@1536", lambda x: ms["clipiqa"](g1536(x)), bf=bf)
    run("musiq@768", lambda x: ms["musiq"](g768(x)), bf=bf)
    run("musiq crop1024", lambda x: ms["musiq"](c(x)), bf=bf); run("maniqa crop1024", lambda x: ms["maniqa"](c(x)), bf=bf)
    run("clip crop1024", lambda x: ms["clipiqa"](c(x)), bf=bf); run("lpips crop1024", lambda x: ms["lpips"](c(x), c(x0)), bf=bf)
    run("dists crop1024", lambda x: ms["dists"](c(x), c(x0)), bf=bf); run("ssim crop1024", lambda x: ms["ssim"](c(x), c(x0)), bf=bf)
run("niqe crop1024", lambda x: ms["niqe"](c(x)))
run("niqe crop512", lambda x: ms["niqe"](x[:, :, :512, :512]))
