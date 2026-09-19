"""
ODTSR 批量推理（改自 examples/qwen_image/test_sr.py）
- 默认 scale=1.0：输入已是 4K，只做增强不放大，输出分辨率与输入一致
- 可调 fidelity / prompt，支持每张图单独 prompt (json: {"case1": "...", ...})
- 输出 caseN.jpg（quality 95），跳过已存在结果，便于断点续跑
运行前需 export qwen_path=/workspace/models/Qwen-Image，并在 /workspace/ODTSR/examples/qwen_image 目录下执行
"""
import os, sys, argparse, time, glob, json
import torch
from PIL import Image
from generator import Generator
from wavelet_color_fix import adain_color_fix, wavelet_color_fix

def adaptive_pad(img, tilesize, stride, mode="reflect"):
    """补边到 tile 对齐尺寸。默认 reflect：黑边会让边缘 tile 被模型当成高对比内容而过度锐化。"""
    w, h = img.size
    pad_h = (tilesize - h) if h <= tilesize else ((h - tilesize + stride - 1) // stride) * stride + tilesize - h
    pad_w = (tilesize - w) if w <= tilesize else ((w - tilesize + stride - 1) // stride) * stride + tilesize - w
    if pad_h == 0 and pad_w == 0:
        return img
    if mode == "reflect":
        import numpy as np, cv2
        a = np.asarray(img)
        # reflect 单次最多只能补 (尺寸-1)，大图不会触发；保险起见分两段
        out = cv2.copyMakeBorder(a, 0, min(pad_h, h - 1), 0, min(pad_w, w - 1), cv2.BORDER_REFLECT_101)
        if out.shape[0] < h + pad_h or out.shape[1] < w + pad_w:
            out = cv2.copyMakeBorder(out, 0, h + pad_h - out.shape[0], 0, w + pad_w - out.shape[1], cv2.BORDER_REFLECT_101)
        return Image.fromarray(out)
    new_img = Image.new(img.mode, (w + pad_w, h + pad_h), color=0)
    new_img.paste(img, (0, 0))
    return new_img

def _ramp(n, ov):
    """长度 n 的一维权重：两端 ov 像素线性渐变（0->1），中间 1"""
    import numpy as np
    w = np.ones(n, dtype=np.float32)
    if ov > 0:
        r = (np.arange(ov) + 1) / (ov + 1)
        w[:ov] = r; w[-ov:] = r[::-1]
    return w


@torch.no_grad()
def infer_patched(model, img, patch, overlap, tilesize, tile_stride, **kw):
    """把大图切成 patch x patch 的独立子图逐块推理（位置编码从 0 开始），重叠区线性融合。"""
    import numpy as np
    W, H = img.size
    step = patch - overlap
    xs = list(range(0, max(W - patch, 0) + 1, step)); ys = list(range(0, max(H - patch, 0) + 1, step))
    if xs[-1] + patch < W: xs.append(W - patch)
    if ys[-1] + patch < H: ys.append(H - patch)
    acc = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W, 1), np.float32)
    for y in ys:
        for x in xs:
            pw, ph = min(patch, W - x), min(patch, H - y)
            sub = img.crop((x, y, x + pw, y + ph))
            padded = adaptive_pad(sub, tilesize=tilesize * 8, stride=tile_stride * 8)
            out = model.infer(condition_image=padded, tiled=True, tile_size=tilesize, tile_stride=tile_stride, **kw).crop((0, 0, pw, ph))
            o = np.asarray(out, dtype=np.float32)
            # 边界 patch 在图像边缘一侧不做渐变
            wx = _ramp(pw, overlap).copy(); wy = _ramp(ph, overlap).copy()
            if x == 0: wx[:overlap] = 1
            if x + pw >= W: wx[-overlap:] = 1
            if y == 0: wy[:overlap] = 1
            if y + ph >= H: wy[-overlap:] = 1
            wgt = (wy[:, None] * wx[None, :])[..., None]
            acc[y:y + ph, x:x + pw] += o * wgt; wsum[y:y + ph, x:x + pw] += wgt
    return Image.fromarray(np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--trained_ckpt", default="/workspace/models/ODTSR/weight.pth")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pre_down", type=float, default=1.0,
                    help="先把输入缩小 k 倍再做 k 倍 SR（重度模糊图用，输出分辨率不变）；>1 时忽略 --scale")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--fidelity", type=float, default=1.0, help="1.0 最保真, 越小生成越多")
    ap.add_argument("--prompt", default="High Contrast, hyper detailed photo, 2k UHD")
    ap.add_argument("--prompt_file", default=None, help="json: {name: prompt}")
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--align_method", default="wavelet", choices=["wavelet", "adain", "no"])
    ap.add_argument("--tile", type=int, default=64, help="latent tile size (64 -> 512px)")
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--start_end", default="0,")
    ap.add_argument("--only", default=None, help="逗号分隔的文件名(不含后缀)，只跑这些")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--also_save_noalign", action="store_true", help="同时把未做颜色对齐的原始输出存到 <output_path>_noalign")
    ap.add_argument("--extra_fidelity", default="", help="逗号分隔的额外 fidelity 档位，如 0.7,0.5，输出到 <output_path>_f<val>")
    ap.add_argument("--patch", type=int, default=0, help=">0 时按 patch x patch 独立子图推理（避免大图位置编码失效），如 1024")
    ap.add_argument("--patch_overlap", type=int, default=128)
    args = ap.parse_args()
    args.extra_fidelity = [float(x) for x in args.extra_fidelity.split(",") if x.strip()]

    tilesize = args.tile
    tile_stride = tilesize - tilesize // 4
    q = os.environ["qwen_path"]
    weights = json.dumps([
        [f"{q}/transformer/diffusion_pytorch_model-0000{i}-of-00009.safetensors" for i in range(1, 10)],
        [f"{q}/text_encoder/model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
        f"{q}/vae/diffusion_pytorch_model.safetensors",
    ])

    t0 = time.time()
    model = Generator(torch_dtype=torch.bfloat16, pretrained_weights=weights, tokenizer_path=f"{q}/tokenizer",
                      learning_rate=0, use_gradient_checkpointing=False, pretrained_ckpt_path_gen=args.trained_ckpt)
    model.pipe.requires_grad_(False)
    model = model.to(device="cuda")
    model.device = next(model.parameters()).device
    model.pipe.device = model.device
    print(f"model loaded in {time.time()-t0:.0f}s, GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths += glob.glob(os.path.join(args.input_path, ext))
    def key(p):
        b = os.path.splitext(os.path.basename(p))[0]
        digits = "".join(c for c in b if c.isdigit())
        return (int(digits) if digits else 0, b)
    paths = sorted(paths, key=key)
    if args.only:
        keep = set(args.only.split(","))
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in keep]
    s, e = args.start_end.split(",")
    paths = paths[int(s or 0): int(e) if e else None]
    print("total:", len(paths), flush=True)

    prompts = json.load(open(args.prompt_file)) if args.prompt_file else {}
    os.makedirs(args.output_path, exist_ok=True)

    for idx, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        out_name = name.replace("_lq", "")
        res_path = os.path.join(args.output_path, f"{out_name}.{args.ext}")
        if os.path.exists(res_path) and not args.overwrite:
            print(f"skip {res_path}"); continue
        t1 = time.time()
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if args.pre_down > 1:
            k = args.pre_down
            small = img.resize((round(w / k), round(h / k)), Image.LANCZOS)
            wd, hd = w, h
            up = small.resize((wd, hd), Image.BICUBIC)
        else:
            wd, hd = round(w * args.scale), round(h * args.scale)
            up = img if (wd, hd) == (w, h) else img.resize((wd, hd), Image.BICUBIC)
        padded = adaptive_pad(up, tilesize=tilesize * 8, stride=tile_stride * 8)
        prompt = prompts.get(name, prompts.get(out_name, args.prompt))
        def run_fid(fid):
            if args.patch > 0:
                return infer_patched(model, up, args.patch, args.patch_overlap, tilesize, tile_stride,
                                     prompt=prompt, negative_prompt=args.negative_prompt, cfg_scale=args.cfg, fidelity=fid)
            return model.infer(prompt=prompt, negative_prompt=args.negative_prompt, condition_image=padded,
                               cfg_scale=args.cfg, fidelity=fid, tiled=True,
                               tile_size=tilesize, tile_stride=tile_stride).crop((0, 0, wd, hd))
        res = run_fid(args.fidelity)
        # 额外 fidelity 档位：同一次加载下再跑几档，输出到 <output_path>_f<val>
        for fx in args.extra_fidelity:
            r2 = run_fid(fx)
            if args.align_method == "wavelet": r2 = wavelet_color_fix(target=r2, source=up)
            elif args.align_method == "adain": r2 = adain_color_fix(target=r2, source=up)
            d2 = args.output_path.rstrip("/") + f"_f{fx}"; os.makedirs(d2, exist_ok=True)
            p2 = os.path.join(d2, f"{out_name}.{args.ext}")
            r2.save(p2, quality=args.quality, subsampling=0) if args.ext.lower() in ("jpg", "jpeg") else r2.save(p2)
        if args.also_save_noalign:
            na_dir = args.output_path.rstrip("/") + "_noalign"; os.makedirs(na_dir, exist_ok=True)
            na_path = os.path.join(na_dir, f"{out_name}.{args.ext}")
            res.save(na_path, quality=args.quality, subsampling=0) if args.ext.lower() in ("jpg", "jpeg") else res.save(na_path)
        if args.align_method == "adain":
            res = adain_color_fix(target=res, source=up)
        elif args.align_method == "wavelet":
            res = wavelet_color_fix(target=res, source=up)
        if args.ext.lower() in ("jpg", "jpeg"):
            res.save(res_path, quality=args.quality, subsampling=0)
        else:
            res.save(res_path)
        print(f"[{idx+1}/{len(paths)}] {out_name} {w}x{h} f={args.fidelity} pre_down={args.pre_down} patch={args.patch} {time.time()-t1:.1f}s  peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

    print(f"done, total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
