import json, sys
rows = {}
for l in open('/workspace/logs/eval_results.jsonl'):
    d = json.loads(l); rows[d['tag']] = d
tags = sys.argv[1].split(',') if len(sys.argv) > 1 else ['LQ_baseline', 'odtsr_f1.0', 'odtsr_f0.6', 'odtsr_f0.4', 'fidesr_s1_hf0.2_lf0.2']
for tag in tags:
    print('==', tag)
    for r in rows[tag]['rows']:
        print('  %s psnr=%.2f ssim=%.3f lpips=%.3f dists=%.3f musiq=%.1f clipiqa=%.3f maniqa=%.3f niqe=%.2f' % (
            r['case'], r['psnr'], r['ssim'], r['lpips'], r['dists'], r['musiq'], r['clipiqa'], r['maniqa'], r['niqe']))
