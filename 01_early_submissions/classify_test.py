"""CLIP 零样本分类测试集/验证集，为按类别路由模型做准备。整图 + 2x2 分块，取各类最大分数。"""
import os, glob, json, torch, numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

dev = "cuda:1"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float16).to(dev).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

CLASSES = {
    "clock":    ["a photo of a clock", "a photo of a clock tower with a clock face", "a photo of a watch face with hands"],
    "text_cn":  ["a photo of Chinese characters on a sign", "a photo of a Chinese shop signboard", "a photo of glowing Chinese text on a building", "a photo of Chinese calligraphy"],
    "text":     ["a photo of text and letters on a sign", "a photo of a neon sign with words", "a photo of a poster with printed text", "a photo of a book page with text"],
    "bird":     ["a photo of a bird", "a photo of a bird perched on a branch", "a photo of pigeons or seagulls"],
    "flower":   ["a photo of flowers", "a close-up photo of flower petals", "a photo of blossoms on a tree"],
    "plant":    ["a photo of dense tree leaves and foliage", "a photo of a forest with trees", "a photo of grass and plants", "a photo of tree branches"],
    "building": ["a photo of a building facade", "a photo of city architecture and skyscrapers", "a photo of a temple or pagoda", "a photo of a bridge or tower"],
    "people":   ["a photo of a person", "a photo of a human face", "a photo of a crowd of people"],
    "animal":   ["a photo of a cat or dog", "a photo of a wild animal", "a photo of an insect"],
    "other":    ["a photo of food", "a photo of a vehicle or car", "a photo of a landscape with sky and water", "a photo of an indoor room", "a photo of an object on a table"],
}
names, texts = [], []
for k, v in CLASSES.items():
    for t in v: names.append(k); texts.append(t)
with torch.no_grad():
    ti = proc(text=texts, return_tensors="pt", padding=True).to(dev)
    tf = model.get_text_features(**ti); tf = tf / tf.norm(dim=-1, keepdim=True)

def views(im):
    w, h = im.size
    out = [im]
    for i in range(2):
        for j in range(2):
            out.append(im.crop((j * w // 2, i * h // 2, (j + 1) * w // 2, (i + 1) * h // 2)))
    return out

def classify(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((1400, 1400))
    vs = views(im)
    with torch.no_grad():
        pi = proc(images=vs, return_tensors="pt").to(dev)
        f = model.get_image_features(pixel_values=pi["pixel_values"].half()); f = f / f.norm(dim=-1, keepdim=True)
        logits = (100 * f @ tf.T).float()                     # [views, prompts]
        probs = logits.softmax(-1)
    # 每个类取其 prompts 的概率和，再对 views 取最大（小物体在某个分块里）
    cls_scores = {}
    for k in CLASSES:
        idx = [i for i, n in enumerate(names) if n == k]
        cls_scores[k] = float(probs[:, idx].sum(-1).max())
    whole = {k: float(probs[0, [i for i, n in enumerate(names) if n == k]].sum()) for k in CLASSES}
    return cls_scores, whole

def run(files, out_json):
    res = {}
    for p in files:
        s, whole = classify(p)
        top = sorted(s.items(), key=lambda x: -x[1])[:3]
        res[os.path.splitext(os.path.basename(p))[0]] = {"scores": s, "whole": whole, "top": top}
        print(os.path.basename(p), " ".join(f"{k}:{v:.2f}" for k, v in top), flush=True)
    json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
    return res

val = sorted(glob.glob("/workspace/data/val/*_lq.jpg"))
print("== val =="); run(val, "/workspace/output/cls_val.json")
test = sorted(glob.glob("/workspace/data/test/*.jpg"), key=lambda p: int(os.path.basename(p)[4:-4]))
print("== test =="); res = run(test, "/workspace/output/cls_test.json")

# 路由规则：优先级 text_cn > clock > bird > text > flower > plant > building > 其它
PRI = ["text_cn", "clock", "bird", "text", "flower", "plant", "building", "people", "animal", "other"]
route = {}
for n, r in res.items():
    s = r["scores"]; best = max(s, key=s.get)
    chosen = best
    for k in PRI[:7]:
        if s[k] >= 0.5 or (s[k] >= 0.35 and k == best):
            chosen = k; break
    route[n] = chosen
json.dump(route, open("/workspace/output/route_test.json", "w"), indent=1)
from collections import Counter; print("route counts:", Counter(route.values()))

# 每类缩略图拼图
os.makedirs("/workspace/output/cls_sheets", exist_ok=True)
for k in set(route.values()):
    ns = [n for n, c in route.items() if c == k]
    T = 256; cols = 8; rows = (len(ns) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * T, rows * T), "white")
    from PIL import ImageDraw
    for i, n in enumerate(ns):
        im = Image.open(f"/workspace/data/test/{n}.jpg").convert("RGB"); im.thumbnail((T, T))
        canvas.paste(im, ((i % cols) * T, (i // cols) * T))
        ImageDraw.Draw(canvas).text(((i % cols) * T + 4, (i // cols) * T + 4), n, fill="red")
    canvas.save(f"/workspace/output/cls_sheets/{k}.jpg", quality=80)
print("CLS_DONE")
