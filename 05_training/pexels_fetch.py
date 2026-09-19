"""Pexels API 按类别关键词搜索，下载原图（min side >= 3000, max side >= 4000）。
用法: python pexels_fetch.py [cat1,cat2]   key 读 /workspace/.pexels_key"""
import os, sys, time, json, requests, collections
from concurrent.futures import ThreadPoolExecutor
KEY = open("/workspace/.pexels_key").read().strip()
OUT = "/workspace/data/train/pexels"
H = {"Authorization": KEY}
# 类别: (查询词列表, 目标张数)   配比参照测试集: text_cn 31 / bird 21 / plant 17 / clock 15 / flower 5
CATS = {
    "bird":   (["bird closeup", "bird on branch", "wild bird", "parrot", "owl", "eagle", "sparrow", "heron", "hummingbird", "kingfisher", "seagull", "duck", "songbird", "pigeon", "crane bird", "peacock", "robin bird", "woodpecker"], 1400),
    "clock":  (["clock", "clock tower", "wall clock", "analog clock", "vintage clock", "station clock", "clock face", "big ben", "old clock", "roman numerals clock", "church clock", "tower clock night", "alarm clock", "pocket watch", "wristwatch"], 900),
    "plant":  (["green leaves", "foliage", "tree leaves", "forest canopy", "bamboo", "fern", "leaves closeup", "tree branches", "dense trees", "garden plants", "tropical leaves", "autumn leaves", "willow tree", "pine tree", "ivy wall", "maple leaves"], 1100),
    "flower": (["flowers", "cherry blossom", "rose", "tulip", "lotus", "sunflower", "flower closeup", "peony", "wildflowers", "orchid", "chrysanthemum", "plum blossom"], 450),
    "text_cn": (["chinese sign", "chinatown", "hong kong street", "hong kong neon", "chinese lantern", "shanghai street", "taipei night market", "chinese calligraphy", "chinese temple", "chinese shop sign", "chinese characters", "japanese street sign", "kanji sign", "macau street", "chinese restaurant sign", "beijing hutong", "chinese new year decorations", "chinese street night", "hanzi"], 1000),
    "light":  (["city lights night", "street lamp night", "neon lights", "lanterns night", "shop window night", "night street signs", "illuminated sign", "light bulbs", "christmas lights"], 400),
    "building": (["temple roof", "pagoda", "traditional chinese architecture", "old building facade", "brick wall building", "cathedral facade"], 300),
}
PER_PAGE = 80
_req = [0.0]

def search(q, page):
    for k in range(4):
        wait = 0.6 - (time.time() - _req[0])
        if wait > 0: time.sleep(wait)
        try:
            r = requests.get("https://api.pexels.com/v1/search", headers=H, params={"query": q, "per_page": PER_PAGE, "page": page, "orientation": "landscape"}, timeout=60)
            _req[0] = time.time()
            if r.status_code == 200: return r.json()
            if r.status_code == 429:
                rst = int(r.headers.get("X-Ratelimit-Reset", "60") or 60); print(f"429 on {q} p{page}, sleep {min(rst, 900)}", flush=True); time.sleep(min(rst, 900) + 2); continue
            print("http", r.status_code, q, page, flush=True); time.sleep(3)
        except Exception as e:
            print("err", e, flush=True); time.sleep(3)
    return {}

def collect(cat, queries, target, taken):
    picks = {}
    for q in queries:
        if len(picks) >= target: break
        for page in range(1, 40):
            d = search(q, page)
            ph = d.get("photos", [])
            if not ph: break
            for p in ph:
                w, h = p["width"], p["height"]
                if min(w, h) >= 3000 and max(w, h) >= 4000 and p["id"] not in taken and p["id"] not in picks:
                    picks[p["id"]] = (p["src"]["original"], w, h, q)
            if len(picks) >= target or not d.get("next_page"): break
        print(f"  [{cat}] after '{q}': {len(picks)}", flush=True)
    return picks

def dl(job):
    url, p = job
    for k in range(3):
        try:
            r = requests.get(url, timeout=180, stream=True)
            if r.status_code == 200:
                with open(p + ".part", "wb") as f:
                    for c in r.iter_content(1 << 20): f.write(c)
                if os.path.getsize(p + ".part") > 300_000: os.rename(p + ".part", p); return 1
                os.remove(p + ".part")
            time.sleep(2 + 3 * k)
        except Exception:
            time.sleep(2 + 3 * k)
    return 0

def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CATS)
    taken = set(); meta = {}
    mp = f"{OUT}/meta.json"
    if os.path.exists(mp): meta = json.load(open(mp)); taken = {int(k) for k in meta}
    for cat in only:
        qs, target = CATS[cat]; d = f"{OUT}/{cat}"; os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".jpg")])
        if have >= target: print(f"[{cat}] already {have}"); continue
        picks = collect(cat, qs, target - have, taken)
        for pid, (url, w, h, q) in picks.items(): taken.add(pid); meta[str(pid)] = {"cat": cat, "w": w, "h": h, "q": q}
        json.dump(meta, open(mp, "w"))
        jobs = [(url, f"{d}/{pid}.jpg") for pid, (url, w, h, q) in picks.items() if not os.path.exists(f"{d}/{pid}.jpg")]
        t0 = time.time(); ok = 0
        with ThreadPoolExecutor(12) as ex:
            for i, r in enumerate(ex.map(dl, jobs)):
                ok += r
                if i % 100 == 0: print(f"  [{cat}] dl {i}/{len(jobs)} ok={ok} {time.time()-t0:.0f}s", flush=True)
        print(f"[{cat}] done: {len([f for f in os.listdir(d) if f.endswith('.jpg')])} files", flush=True)
    print("PEXELS_DONE")

if __name__ == "__main__":
    main()
