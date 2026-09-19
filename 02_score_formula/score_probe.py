"""在测试集抽样上算一批无参考指标，对比三份已知分数的提交（3.0675 / 3.4290 / 3.6235）+ LQ，找与榜单一致的指标/尺度。
CPU 运行；为省时取每图 768x768 中心裁剪（原分辩率不缩放）。"""
import os, sys, glob, json, time, warnings, numpy as np, cv2, torch, pyiqa
warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("NT", "16")))
dev = "cpu"
IDS = list(range(1, 101, 8))          # 13 张
CROP = int(os.environ.get("CROP", "768"))
METRICS = os.environ.get("METRICS", "musiq,clipiqa,clipiqa+,maniqa,niqe,liqe,topiq_nr,hyperiqa,dbcnn,nima,arniqa,qualiclip+,paq2piq,tres").split(",")

def center(im):
    h, w = im.shape[:2]; cy, cx = h // 2, w // 2; s = CROP // 2
    return im[cy - s:cy + s, cx - s:cx + s]

def jpeg95(rgb):
    from PIL import Image; import io
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=95, subsampling=0); b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"))

def load_versions(i):
    n = f"case{i}"
    lq = cv2.cvtColor(cv2.imread(f"/workspace/data/test/{n}.jpg"), cv2.COLOR_BGR2RGB)
    raw = cv2.cvtColor(cv2.imread(f"/workspace/output/test_odtsr_f1.0/{n}.png"), cv2.COLOR_BGR2RGB)
    fuse = cv2.cvtColor(cv2.imread(f"/workspace/output/submission_fuse/{n}.jpg"), cv2.COLOR_BGR2RGB)
    # 首次提交：raw 高斯模糊 0.8 后与 LQ 按 0.85:0.15 混合，再 JPEG95
    fin = cv2.GaussianBlur(raw, (0, 0), 0.8).astype(np.float32) * 0.85 + lq.astype(np.float32) * 0.15
    fin = jpeg95(np.clip(fin, 0, 255).astype(np.uint8))
    return {"LQ": lq, "final(3.07)": fin, "fuse(3.43)": fuse, "raw(3.62)": jpeg95(raw)}

def to_t(rgb):
    return torch.from_numpy(np.ascontiguousarray(center(rgb))).permute(2, 0, 1).float().div(255).unsqueeze(0)

imgs = {i: {k: to_t(v) for k, v in load_versions(i).items()} for i in IDS}
names = ["LQ", "final(3.07)", "fuse(3.43)", "raw(3.62)"]
out = {}
for m in METRICS:
    t0 = time.time()
    try:
        met = pyiqa.create_metric(m, device=dev)
    except Exception as e:
        print(f"{m}: create failed {e}"); continue
    res = {k: [] for k in names}
    with torch.no_grad():
        for i in IDS:
            for k in names:
                try: res[k].append(float(met(imgs[i][k]).item()))
                except Exception as e: print(f"{m} {k} case{i} err {e}"); res[k].append(float("nan"))
    means = {k: float(np.nanmean(v)) for k, v in res.items()}
    out[m] = means
    print(f"{m:12s} lower_better={met.lower_better!s:5s} " + "  ".join(f"{k}={means[k]:.4f}" for k in names) + f"   ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open("/workspace/output/score_probe.json", "w"), indent=1)
print("PROBE_DONE")
