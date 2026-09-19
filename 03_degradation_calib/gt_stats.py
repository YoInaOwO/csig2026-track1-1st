"""对比官方 GT/LQ 与合成 GT/LQ 的全分辨率统计：lap 全分辨/::4、高频能量比、NR 四项。并行 CPU + GPU NR。"""
import glob, os, sys, numpy as np, cv2, torch, warnings, pyiqa
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")
def stats(p):
    im = cv2.imread(p); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F).var(); lap4 = cv2.Laplacian(g[::4, ::4], cv2.CV_32F).var()
    # 高频能量：用 1024 中心块 FFT
    h, w = g.shape; c = g[h // 2 - 512:h // 2 + 512, w // 2 - 512:w // 2 + 512]; c = c - c.mean()
    f = np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2; yy, xx = np.mgrid[-512:512, -512:512]; r = np.sqrt(yy ** 2 + xx ** 2) / 1024
    hf = float(f[r > 0.25].sum() / (f.sum() + 1e-9)); mf = float(f[(r > 0.1) & (r <= 0.25)].sum() / (f.sum() + 1e-9))
    return os.path.basename(p), lap, lap4, mf, hf
groups = {"real_GT": sorted(glob.glob("/workspace/data/val/case*_gt.jpg")), "real_LQ": sorted(glob.glob("/workspace/data/val/case*_lq.jpg")),
          "syn_GT": sorted(glob.glob("/workspace/data/sval/case*_gt.png")), "syn_LQ": sorted(glob.glob("/workspace/data/sval/case*_lq.png"))}
nr = {k: pyiqa.create_metric(k, device="cuda") for k in ["musiq", "clipiqa", "maniqa", "niqe"]}
def load(p): return torch.from_numpy(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
with ProcessPoolExecutor(16) as ex:
    for name, files in groups.items():
        st = list(ex.map(stats, files))
        rows = []
        for (b, lap, lap4, mf, hf), p in zip(st, files):
            x = load(p)
            with torch.no_grad(): v = {k: float(m(x)) for k, m in nr.items()}
            rows.append((lap, lap4, mf, hf, v["musiq"], v["clipiqa"], v["maniqa"], v["niqe"]))
            print(f"  {name} {b:14s} lap={lap:8.1f} lap4={lap4:8.1f} mf={mf:.4f} hf={hf:.4f} musiq={v['musiq']:.1f} clip={v['clipiqa']:.3f} maniqa={v['maniqa']:.3f} niqe={v['niqe']:.1f}", flush=True)
        a = np.array(rows); med = np.median(a, 0); mean = a.mean(0)
        print(f"{name} MEDIAN lap={med[0]:.1f} lap4={med[1]:.1f} mf={med[2]:.4f} hf={med[3]:.4f} musiq={med[4]:.1f} clip={med[5]:.3f} maniqa={med[6]:.3f} niqe={med[7]:.1f}")
        print(f"{name} MEAN   lap={mean[0]:.1f} lap4={mean[1]:.1f} mf={mean[2]:.4f} hf={mean[3]:.4f} musiq={mean[4]:.1f} clip={mean[5]:.3f} maniqa={mean[6]:.3f} niqe={mean[7]:.1f}", flush=True)
