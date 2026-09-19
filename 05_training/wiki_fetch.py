"""从 Wikimedia Commons 按分类抓高分辨率原图（宽或高 >= MIN_SIDE），用于各类别 GT 数据"""
import os, sys, time, json, requests, hashlib
from concurrent.futures import ThreadPoolExecutor

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "tianchi-sr-research/1.0 (contact: student research; python-requests)"}
MIN_SIDE = 3500
OUT_ROOT = "/workspace/data/train/wiki"

CATEGORIES = {
    "text_cn":  (["Signs in Chinese", "Neon signs in Hong Kong", "Shop signs in China", "Signs in Taiwan", "Chinese calligraphy on signs", "Billboards in China", "Chinese shop signs", "Shop signs in Hong Kong", "Signs in Hong Kong", "Neon signs in China", "Shop signs in Taiwan", "Signs in Macau", "Chinese-language signs in Japan"], 800),
    "clock":    (["Quality images of clocks", "Clock faces", "Tower clocks", "Station clocks", "Public clocks", "Clock towers in Europe", "Clock towers in China", "Street clocks", "Clocks in churches"], 500),
    "bird":     (["Quality images of birds", "Featured pictures of birds", "Valued images of birds"], 500),
    "plant":    (["Quality images of leaves", "Quality images of trees", "Quality images of forests", "Featured pictures of trees"], 300),
    "flower":   (["Quality images of flowers", "Featured pictures of flowers"], 200),
}

_last_api = [0.0]
def api(params, retries=5):
    params = dict(params, format="json", action="query")
    for k in range(retries):
        wait = 1.2 - (time.time() - _last_api[0])
        if wait > 0: time.sleep(wait)
        try:
            r = requests.get(API, params=params, headers=UA, timeout=60)
            _last_api[0] = time.time()
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                print("429, backoff", 40 * (k + 1), flush=True); time.sleep(40 * (k + 1)); continue
            time.sleep(2 + 3 * k)
        except Exception:
            time.sleep(2 + 3 * k)
    return {}

def list_files(category, limit, depth=1, seen=None):
    """遍历分类（含一层子分类）返回文件标题列表"""
    seen = seen if seen is not None else set()
    files, cont = [], {}
    while len(files) < limit:
        d = api({"list": "categorymembers", "cmtitle": f"Category:{category}", "cmlimit": 500, "cmtype": "file|subcat", **cont})
        members = d.get("query", {}).get("categorymembers", [])
        for m in members:
            t = m["title"]
            if t.startswith("File:") and t not in seen:
                seen.add(t); files.append(t)
            elif t.startswith("Category:") and depth > 0 and len(files) < limit:
                files += list_files(t[len("Category:"):], limit - len(files), depth - 1, seen)
        if "continue" in d and len(files) < limit:
            cont = d["continue"]
        else:
            break
    return files[:limit]

def file_infos(titles):
    infos = []
    for i in range(0, len(titles), 50):
        d = api({"prop": "imageinfo", "titles": "|".join(titles[i:i + 50]), "iiprop": "url|size|mime"})
        for p in d.get("query", {}).get("pages", {}).values():
            ii = p.get("imageinfo", [{}])[0]
            if ii.get("mime") in ("image/jpeg", "image/png") and max(ii.get("width", 0), ii.get("height", 0)) >= MIN_SIDE and ii.get("size", 0) < 30_000_000:
                infos.append(ii)
    return infos

def download(ii, out_dir):
    url = ii["url"]; name = hashlib.md5(url.encode()).hexdigest()[:16] + os.path.splitext(url)[1].lower()
    p = os.path.join(out_dir, name)
    if os.path.exists(p): return True
    for k in range(3):
        try:
            time.sleep(0.4)
            r = requests.get(url, headers=UA, timeout=120, stream=True)
            if r.status_code == 200:
                with open(p + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20): f.write(chunk)
                os.rename(p + ".part", p); return True
            if r.status_code == 429:
                time.sleep(40 * (k + 1)); continue
            time.sleep(3 + 5 * k)
        except Exception:
            time.sleep(3 + 5 * k)
    return False

def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CATEGORIES)
    for cat in only:
        cats, target = CATEGORIES[cat]
        out_dir = os.path.join(OUT_ROOT, cat); os.makedirs(out_dir, exist_ok=True)
        have = len(os.listdir(out_dir))
        if have >= target: print(f"[{cat}] already {have}"); continue
        titles, seen = [], set()
        for c in cats:
            titles += list_files(c, target * 2, depth=1, seen=seen)
            if len(titles) >= target * 2: break
        infos = file_infos(titles)
        print(f"[{cat}] {len(titles)} titles -> {len(infos)} hi-res candidates", flush=True)
        infos = infos[:target]
        with ThreadPoolExecutor(2) as ex:
            ok = sum(ex.map(lambda ii: download(ii, out_dir), infos))
        print(f"[{cat}] downloaded {ok}, dir now {len(os.listdir(out_dir))} files", flush=True)
    print("WIKI_DONE")

if __name__ == "__main__":
    main()
