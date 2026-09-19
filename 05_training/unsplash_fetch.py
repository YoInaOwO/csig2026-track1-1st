"""从 Unsplash Lite 按关键词筛类别，下载 4096 宽 jpg"""
import csv, os, sys, time, requests, collections
from concurrent.futures import ThreadPoolExecutor
csv.field_size_limit(1 << 30)
D = "/workspace/data/train/_dl/unsplash"
OUT = "/workspace/data/train/unsplash"
CATS = {
    "bird":     ({"bird", "birds", "parrot", "owl", "eagle", "duck", "hummingbird", "sparrow", "seagull", "heron", "flamingo", "penguin", "peacock"}, 700),
    "clock":    ({"clock", "clock tower", "wall clock", "analog clock", "watch", "wristwatch", "big ben"}, 500),
    "flower":   ({"flower", "flowers", "blossom", "rose", "petal", "tulip", "sunflower", "daisy", "cherry blossom", "lotus"}, 600),
    "plant":    ({"tree", "trees", "forest", "leaf", "leaves", "foliage", "branch", "woodland", "grass", "bamboo"}, 600),
    "text":     ({"sign", "signage", "neon", "neon sign", "text", "signboard", "billboard", "chinese", "chinatown", "hong kong", "shanghai", "taipei", "tokyo", "lantern", "storefront", "street sign", "poster", "typography", "letters", "word"}, 900),
    "building": ({"building", "architecture", "skyscraper", "temple", "pagoda", "cathedral", "facade", "city", "tower", "bridge", "palace", "castle"}, 700),
    "people":   ({"person", "people", "face", "portrait", "crowd", "human", "woman", "man", "child"}, 400),
}
UA = {"User-Agent": "Mozilla/5.0 (research; tianchi-sr)"}

def main():
    photos = {}
    with open(f"{D}/photos.tsv000", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try: w, h = int(r["photo_width"]), int(r["photo_height"])
            except: continue
            if max(w, h) >= 3000: photos[r["photo_id"]] = (r["photo_image_url"], w, h, (r.get("ai_description") or "").lower())
    print("photos >=3000px:", len(photos))
    kws = collections.defaultdict(dict)
    with open(f"{D}/keywords.tsv000", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pid = r["photo_id"]
            if pid not in photos: continue
            c = 0.0
            for k in ("ai_service_1_confidence", "ai_service_2_confidence"):
                try: c = max(c, float(r[k] or 0))
                except: pass
            if r.get("suggested_by_user") == "t": c = max(c, 60.0)
            kws[pid][r["keyword"].lower()] = max(kws[pid].get(r["keyword"].lower(), 0), c)
    assigned = {}
    for cat, (words, target) in CATS.items():
        scored = []
        for pid, (url, w, h, desc) in photos.items():
            if pid in assigned: continue
            s = max([kws[pid].get(k, 0) for k in words] + [0])
            if any(k in desc.split() or (" " in k and k in desc) for k in words): s = max(s, 55)
            if s >= 45: scored.append((s, pid))
        scored.sort(reverse=True)
        pick = [pid for _, pid in scored[:target]]
        for pid in pick: assigned[pid] = cat
        print(f"{cat}: {len(scored)} candidates -> {len(pick)}", flush=True)
    jobs = []
    for pid, cat in assigned.items():
        d = f"{OUT}/{cat}"; os.makedirs(d, exist_ok=True)
        p = f"{d}/{pid}.jpg"
        if not os.path.exists(p): jobs.append((photos[pid][0], p))
    print("to download:", len(jobs), flush=True)
    def dl(job):
        url, p = job
        for k in range(3):
            try:
                r = requests.get(f"{url}?w=4096&q=92&fm=jpg&fit=max", headers=UA, timeout=120)
                if r.status_code == 200 and len(r.content) > 100_000:
                    open(p, "wb").write(r.content); return 1
                time.sleep(2 + 3 * k)
            except Exception:
                time.sleep(2 + 3 * k)
        return 0
    t0 = time.time(); ok = 0
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(dl, jobs)):
            ok += r
            if i % 200 == 0: print(f"{i}/{len(jobs)} ok={ok} {time.time()-t0:.0f}s", flush=True)
    print("UNSPLASH_DONE", ok)
    for cat in CATS: print(cat, len(os.listdir(f"{OUT}/{cat}")) if os.path.isdir(f"{OUT}/{cat}") else 0)

if __name__ == "__main__":
    main()
