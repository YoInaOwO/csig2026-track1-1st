"""
copy from https://github.com/csslc/PiSA-SR/blob/main/src/datasets/dataset.py
change record: https://www.diffchecker.com/wEIJtoyR/
"""
import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from .realesrgan import RealESRGAN_degradation

import pillow_heif
pillow_heif.options.DISABLE_SECURITY_LIMITS = True

def get_prompt_given_img_path(img_path):
    return 'High Contrast, hyper detailed photo, 2k UHD'  # TIANCHI_PATCH fixed prompt

def _old_get_prompt_given_img_path(img_path):
    txt_path = img_path.replace("/img/", "/prompt/").replace(".png", ".txt")
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()  # 读取内容并去除首尾空白字符
    except FileNotFoundError:
        print(f"Warning: Prompt file not found for {img_path}")
        return ""
    except Exception as e:
        print(f"Error reading prompt file {txt_path}: {str(e)}")
        return ""

def tensor_01_to_pil(tensor):
    tensor = tensor.detach().cpu()
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    tensor = tensor.permute(1, 2, 0)
    image = Image.fromarray(tensor.numpy())
    return image


# ===================== TIANCHI_PATCH helpers =====================
import torch as _torch, random as _random, numpy as _np
import torch.nn.functional as _F
from PIL import Image as _Image, ImageFilter as _ImageFilter

def tianchi_pre_resize(img):
    """训练图多为 2K，比赛图为 4K；以 0.5 概率放大 1.2~2.0 倍使物体尺度接近。并保证短边 >= 520。"""
    w, h = img.size
    if _random.random() < 0.5:
        k = _random.uniform(1.2, 2.0)
        img = img.resize((round(w * k), round(h * k)), _Image.BICUBIC)
        w, h = img.size
    if min(w, h) < 520:
        k = 520 / min(w, h)
        img = img.resize((int(w * k) + 1, int(h * k) + 1), _Image.BICUBIC)
    return img

def tianchi_extra_degrade(lq):
    """lq: tensor [1,3,H,W] in [0,1]，与 GT 同尺寸。
    (a) 对比度压缩 + 亮度抬升: lq = a*lq + b      (验证集: a 0.86~0.98 常见, 0.45 极端; b>0)
    (b) 锯齿 + 过锐化 (case5 型): 下采样后最近邻放大 + unsharp
    """
    r = _random.random()
    if r < 0.15:
        # 锯齿过锐化
        _, _, H, W = lq.shape
        k = _random.choice([2, 3, 4])
        small = _F.interpolate(lq, size=(H // k, W // k), mode="area")
        lq = _F.interpolate(small, size=(H, W), mode="nearest")
        blur = _F.avg_pool2d(_F.pad(lq, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        amt = _random.uniform(0.4, 1.2)
        lq = (lq + amt * (lq - blur)).clamp(0, 1)
    # 对比度/亮度（大多数样本都有轻微偏移）
    r2 = _random.random()
    if r2 < 0.2:
        a = _random.uniform(0.40, 0.70); b = _random.uniform(0.08, 0.16)
    elif r2 < 0.85:
        a = _random.uniform(0.85, 1.0); b = _random.uniform(0.0, 0.06)
    else:
        a, b = 1.0, 0.0
    lq = (lq * a + b).clamp(0, 1)
    # TIANCHI_PATCH_V2: 末端微弱白噪声 + JPEG，模拟真实 LQ 的高频底噪（Immerkaer 0.15~0.6）
    if _random.random() < 0.9:
        sig = _random.uniform(0.1, 1.5) if _random.random() < 0.85 else _random.uniform(1.5, 3.0)
        lq = (lq + _torch.randn_like(lq) * (sig / 255.0)).clamp(0, 1)
    if _random.random() < 0.85:
        import cv2 as _cv2
        qf = int(_random.uniform(50, 98))
        arr = (lq[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(_np.uint8)
        ok, buf = _cv2.imencode(".jpg", _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR), [_cv2.IMWRITE_JPEG_QUALITY, qf])
        arr = _cv2.cvtColor(_cv2.imdecode(buf, 1), _cv2.COLOR_BGR2RGB)
        lq = _torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(lq.dtype)
    return lq
# ===================== end TIANCHI_PATCH =====================

class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    """
        args should have:
        * deg_file_path
        * dataset_txt_paths
        * use_qwen
        * highquality_dataset_txt_paths
        * null_text_ratio
    """
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                # transforms.Resize(1440),
                transforms.RandomCrop((512, 512)),
                # transforms.RandomHorizontalFlip(),
            ])
            
            self.gt_list = []
            for txt in args.dataset_txt_paths.split():   # 按空格切分成多个文件路径
                with open(txt, 'r') as f:
                    self.gt_list.extend([line.strip() for line in f])
            if args.highquality_dataset_txt_paths is not None:
                with open(args.highquality_dataset_txt_paths, 'r') as f:
                    self.hq_gt_list = [line.strip() for line in f.readlines()]

        elif split == 'test':
            self.input_folder = os.path.join(args.dataset_test_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(args.dataset_test_folder, "test_HR")
            self.lr_list = []
            self.gt_list = []
            lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder,gt_names[i]))
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        def safe_open(idx, paths):
            # 尝试使用指定索引
            if 0 <= idx < len(paths) and "|" in paths[idx]:   # TIANCHI_PATCH_V3 配对行
                return None, paths[idx]
            if 0 <= idx < len(paths) and os.path.exists(paths[idx]):
                try:
                    img = Image.open(paths[idx]).convert('RGB')
                    return img, paths[idx]
                except:
                    pass

            # 随机选择可用路径
            while True:
                p = random.choice(paths)
                if "|" in p: return None, p
                if os.path.exists(p):
                    try:
                        img = Image.open(p).convert('RGB')
                        return img, p
                    except:
                        continue

        if self.split == 'train':
            if self.args.highquality_dataset_txt_paths is not None:
                if np.random.uniform() < self.args.prob:
                    gt_img = Image.open(self.gt_list[idx]).convert('RGB')
                else:
                    idx = random.sample(range(0, len(self.hq_gt_list)), 1)
                    gt_img = Image.open(self.hq_gt_list[idx[0]]).convert('RGB')
            else:
                gt_img, gt_img_path = safe_open(idx, self.gt_list)
            if "|" in gt_img_path:   # TIANCHI_PATCH_V3: 真实配对 gt|lq，同位置裁剪，不做合成退化
                gp, lp = gt_img_path.split("|")
                gt_img = Image.open(gp).convert('RGB'); lq_img = Image.open(lp).convert('RGB')
                i, j, h, w = transforms.RandomCrop.get_params(gt_img, (512, 512))
                gt_img = F.crop(gt_img, i, j, h, w); lq_img = F.crop(lq_img, i, j, h, w)
                output_t, img_t = F.to_tensor(gt_img), F.to_tensor(lq_img)
            else:
                if "/tiles/" not in gt_img_path: gt_img = tianchi_pre_resize(gt_img)  # TIANCHI_PATCH_V3: 原生 4K 块不缩放
                gt_img = self.crop_preproc(gt_img)
                if random.random() < 0.5 and "text" not in gt_img_path and "light" not in gt_img_path:
                    gt_img = F.hflip(gt_img)  # 文字/招牌类不做水平翻转
                output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
                img_t = tianchi_extra_degrade(img_t)  # TIANCHI_PATCH
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            # img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            # output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            # [0,1] c,h,w  to uint8 pil IMage 


            example = {}
            # example["prompt"] = caption
            # example["neg_prompt"] = self.args.neg_prompt_csd
            # example["null_prompt"] = ""
            example["gt"] = tensor_01_to_pil(output_t)
            example["lq"] = tensor_01_to_pil(img_t)
            
            if not self.args.use_qwen:
                random_value = random.random()
                if random_value < self.args.null_text_ratio:
                    text = ""
                else:
                    text = get_prompt_given_img_path(gt_img_path)
                    
                example['text'] = text
            
            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_t = self.crop_preproc(input_img)
            output_t = self.crop_preproc(output_img)
            # input images scaled to -1, 1
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example