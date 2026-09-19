"""比较 eval_results.jsonl 中若干 tag 在共同 case 子集上的均值。用法: python cmp.py tagA tagB ..."""
import json, sys, numpy as np
R = {}
for l in open("/workspace/logs/eval_results.jsonl"):
    d = json.loads(l); R[d["tag"]] = {r["case"]: r for r in d["rows"]}
tags = sys.argv[1:]
common = set.intersection(*[set(R[t]) for t in tags])
print("common n =", len(common))
for t in tags:
    m = {k: float(np.mean([R[t][c][k] for c in common])) for k in ["psnr", "ssim", "lpips", "dists", "musiq", "clipiqa", "maniqa", "niqe"]}
    mc = dict(m); mc["maniqa"] = min(mc["maniqa"], 1.0)
    FR = (m["psnr"] / 50 + m["ssim"] + 1 - m["lpips"] + 1 - m["dists"]) / 4
    NR = (m["musiq"] / 100 + m["clipiqa"] + mc["maniqa"] + 1 - m["niqe"] / 10) / 4
    print(f"{t:32s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()) + f" | FR={FR:.3f} NR={NR:.3f} PRED={-0.86+3.02*FR+4.57*NR:.3f}")
