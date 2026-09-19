"""按拟合公式 Score = -0.86 + 3.02*FRm + 4.57*NRm 给 eval_results.jsonl 里的各 tag 算预测分。
FRm = mean(psnr/50, ssim, 1-lpips, 1-dists)   NRm = mean(musiq/100, clipiqa, maniqa, 1-niqe/10)
用法: python score_pred.py [tag_substring]"""
import json, sys
rows = {}
for l in open("/workspace/logs/eval_results.jsonl"):
    d = json.loads(l); rows[d["tag"]] = d["mean"]
flt = sys.argv[1] if len(sys.argv) > 1 else ""
OFF = 0.20   # 验证集全图 NR -> 测试集中心裁剪 NR 的口径差（base 0.19 / cfg2 0.26 / LQ 0.00），对模型输出取 0.2
print("%-26s %6s %6s %6s %6s | %6s %7s %6s %6s | %5s %5s %6s %6s" % ("tag", "psnr", "ssim", "lpips", "dists", "musiq", "clipiqa", "maniqa", "niqe", "FRm", "NRm", "PRED", "榜单估"))
res = []
for t, m in rows.items():
    if flt and flt not in t: continue
    FR = (m["psnr"] / 50 + m["ssim"] + (1 - m["lpips"]) + (1 - m["dists"])) / 4
    NR = (m["musiq"] / 100 + m["clipiqa"] + m["maniqa"] + (1 - m["niqe"] / 10)) / 4
    res.append((-0.86 + 3.02 * FR + 4.57 * NR, t, m, FR, NR))
for p, t, m, FR, NR in sorted(res, key=lambda x: -x[0]):
    lb = p + (0.0 if t in ("LQ", "GT") else OFF)
    print("%-26s %6.2f %6.3f %6.3f %6.3f | %6.2f %7.3f %6.3f %6.2f | %5.3f %5.3f %6.3f %6.2f" % (t[-26:], m["psnr"], m["ssim"], m["lpips"], m["dists"], m["musiq"], m["clipiqa"], m["maniqa"], m["niqe"], FR, NR, p, lb))
