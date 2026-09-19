"""CPU 上用 CLIP 给切块打主体存在分，过滤只含背景的块。写 /workspace/data/tiles/clipscore_<cat>.json 与 list_clip_<cat>.txt
用法: python clip_filter.py bird [thr=0.35]"""
import sys, glob, json, os, torch, numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
torch.set_num_threads(int(os.environ.get("NT", "64")))
cat = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
P = {"bird": (["a photo of a bird", "a close-up of a bird's feathers and eye"], ["tree branches without animals", "the sky", "blurry background", "rocks and ground", "water surface", "leaves"]),
     "clock": (["a clock face with hands and numerals", "a clock tower with a clock"], ["a building wall", "the sky", "a street", "blurry background", "a person"]),
     "text_cn": (["chinese characters on a sign", "a neon sign with text", "a shop sign with writing"], ["a street without signs", "a building wall", "the sky", "people", "cars", "blurry background"]),
     "flower": (["a close-up of flower petals", "flowers"], ["green leaves", "blurry background", "the sky", "soil"])}
pos, neg = P[cat]
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(); proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
with torch.no_grad():
    te = model.get_text_features(**proc(text=pos + neg, return_tensors="pt", padding=True)); te = te / te.norm(dim=-1, keepdim=True)
files = []
for f in glob.glob(f"/workspace/data/tiles/list_*_{cat}.txt"): files += [l.strip() for l in open(f) if l.strip()]
print(cat, len(files), "tiles", flush=True)
scores = {}
B = 128
for i in range(0, len(files), B):
    ims = []
    for p in files[i:i + B]:
        try: ims.append(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR))
        except Exception: ims.append(Image.new("RGB", (224, 224)))
    with torch.no_grad():
        ie = model.get_image_features(**proc(images=ims, return_tensors="pt")); ie = ie / ie.norm(dim=-1, keepdim=True)
        pr = (100 * ie @ te.T).softmax(-1)
        ps = pr[:, :len(pos)].sum(-1)
    for p, s in zip(files[i:i + B], ps.tolist()): scores[p] = s
    if (i // B) % 20 == 0: print(f"{i}/{len(files)}", flush=True)
json.dump(scores, open(f"/workspace/data/tiles/clipscore_{cat}.json", "w"))
keep = [p for p, s in scores.items() if s >= thr]
open(f"/workspace/data/tiles/list_clip_{cat}.txt", "w").write("\n".join(sorted(keep)) + "\n")
v = np.array(list(scores.values()))
print(f"{cat}: keep {len(keep)}/{len(files)} at thr {thr}; score pct 25/50/75 = {np.percentile(v,25):.2f}/{np.percentile(v,50):.2f}/{np.percentile(v,75):.2f}")
