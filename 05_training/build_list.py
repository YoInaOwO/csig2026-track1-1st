"""按测试集配比组训练清单 + 每类抽 12 块拼接对比图。
用法: python build_list.py [total=40000] [with_pairs=0|1]"""
import glob, random, os, sys, collections
from PIL import Image, ImageDraw
random.seed(0)
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
T = "/workspace/data/tiles"
# 配比（测试集: text_cn 31 / bird 21 / plant 17 / clock 15 / flower 5 / other 11）
W = {"text_cn": 0.30, "bird": 0.20, "plant": 0.18, "clock": 0.15, "flower": 0.07, "light": 0.06, "building": 0.04}
by_cat = collections.defaultdict(list)
for f in glob.glob(f"{T}/list_*.txt"):
    if os.path.basename(f).startswith("list_clip_"): continue
    cat = os.path.basename(f)[5:-4].split("_", 1)[1]
    by_cat[cat] += [l.strip() for l in open(f) if l.strip()]
USE_CLIP = int(sys.argv[3]) if len(sys.argv) > 3 else 0
if USE_CLIP:
    for f in glob.glob(f"{T}/list_clip_*.txt"):
        cat = os.path.basename(f)[10:-4]; keep = [l.strip() for l in open(f) if l.strip()]
        print(f"clip filter {cat}: {len(by_cat[cat])} -> {len(keep)}"); by_cat[cat] = keep
out = []
for cat, w in W.items():
    pool = by_cat.get(cat, []); n = int(TOTAL * w)
    if not pool: print("EMPTY", cat); continue
    pick = random.sample(pool, n) if len(pool) >= n else pool + random.choices(pool, k=n - len(pool))
    out += pick; print(f"{cat:9s} pool={len(pool):6d} -> {len(pick)}")
if PAIRS:
    # 真实配对 case1-4 各重复 150 次（每次随机裁一块）；case5 留作验证
    for i in range(1, 5):
        out += [f"/workspace/data/val/case{i}_gt.jpg|/workspace/data/val/case{i}_lq.jpg"] * 150
    print("pairs added:", 600)
random.shuffle(out)
name = f"/workspace/data/train/list_tiles_{TOTAL}{'_pairs' if PAIRS else ''}{'_clip' if USE_CLIP else ''}.txt"
open(name, "w").write("\n".join(out) + "\n"); print("wrote", name, len(out))
# 对比图：每类随机 12 块，缩到 256
rows = []
for cat in W:
    pool = by_cat.get(cat, [])
    if not pool: continue
    row = Image.new("RGB", (256 * 12, 256))
    for j, p in enumerate(random.sample(pool, min(12, len(pool)))):
        row.paste(Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR), (j * 256, 0))
    ImageDraw.Draw(row).text((5, 5), cat, fill="red"); rows.append(row)
sheet = Image.new("RGB", (256 * 12, 256 * len(rows)))
for i, r in enumerate(rows): sheet.paste(r, (0, i * 256))
sheet.save("/workspace/output/tiles_sheet.jpg", quality=85); print("sheet saved")
