# 天池 · 可控生成式 4K 图像增强

赛题：对 100 张 4K 低质量图（LQ）做增强，要求使用 Diffusion 架构。评分同时包含有参考指标（与官方高质量图 GT 比较）和无参考指标，官方未公布具体公式。

本仓库记录 7 次提交从 2.18 到 4.54 的完整过程，分三部分：

1. 前 5 次提交分别做了什么、拿了多少分
2. 怎样只靠这 5 个分数把评分公式反推出来
3. 最终提交（榜单 4.5353）是怎么生成的

| # | 提交内容 | 榜单分 | 公式预测 |
|---|---|---|---|
| 1 | LQ 原图直接打包 | 2.1805 | — |
| 2 | ODTSR 预训练权重推理 + 高斯模糊 + 与 LQ 混合 | 3.0675 | — |
| 3 | 微调模型与预训练模型按图像类别加权融合 | 3.4290 | — |
| 4 | ODTSR 预训练权重推理，原始输出（fidelity=1.0） | 3.6235 | — |
| 5 | ODTSR 预训练权重推理，cfg=2 + 负提示 + fidelity=0.4 | 3.7106 | — |
| 6 | 配对数据微调 750 步 + 测试时优化 v2（在缩小图上算指标） | 4.118 | 4.130 |
| 7 | 配对数据微调 750 步 + 测试时优化 v3（在评测同分辨率上算指标） | **4.5353** | 4.631 |

基座模型：[ODTSR](https://huggingface.co/double8fun/ODTSR)（基于 Qwen-Image 20B 的单步扩散超分模型）。硬件：单卡 RTX PRO 6000 96 GB。脚本中的路径均为训练机器上的 `/workspace/...` 布局。

术语：

- **FR**：有参考指标 PSNR / SSIM / LPIPS / DISTS，需要 GT
- **NR**：无参考指标 MUSIQ / CLIP-IQA / MANIQA / NIQE，不需要 GT
- **TTO**：测试时优化（Test-Time Optimization），对每张输出图单独做梯度优化
- **预训练模型**：ODTSR 官方发布的 `weight.pth`，未经我们微调

---

## 一、前 5 个提交

### #1 LQ 原图（2.1805）

100 张 LQ 不做任何处理，改名 `case{i}.jpg` 打包。目的是拿一个零点：知道"什么都不做"值多少分，后面所有改动都有参照。

### #2 预训练模型推理 + 后处理（3.0675）

`01_early_submissions/infer_odtsr.py` 用预训练权重、`fidelity=1.0`、latent tile 64 推理，再用 `finalize.py` 做后处理：

```
final = 0.85 * GaussianBlur(模型输出, σ=0.8) + 0.15 * LQ    → JPEG q95 4:4:4
```

后处理参数是在官方 5 对验证图上用 `postproc_eval.py` / `rank_variants.py` 按 PSNR/LPIPS 挑出来的。当时的判断是"有参考指标占大头，输出越像 GT 越好"。事后证明这个方向错了：模糊和混合把无参考指标全压低了，比不做后处理的原始输出（#4）还低 0.56。

### #3 按图像类别融合两个模型（3.4290）

- `classify_test.py`：用 CLIP ViT-L/14-336 零样本把 100 张测试图分成中文文字 / 鸟 / 植物 / 钟表 / 花 / 灯光 / 建筑
- 用 Real-ESRGAN 风格的在线退化（`configs/train_tianchi.yaml`、`train_v4.yaml`）微调 ODTSR 1500 步，得到一个保真更好但更模糊的微调模型
- `fuse_submit.py`：`输出 = α·预训练模型输出 + (1-α)·微调模型输出`，钟表 α=0.5、文字 α=0.6、其余 0.7

比 #2 高 0.36，主要因为混进了更多预训练模型的锐度；说明无参考指标权重不小。

### #4 预训练模型原始输出（3.6235）

`remote_raw_export.sh`：把 #2 的推理结果不模糊、不混合直接导出。比 #2 高 0.56，确认"后处理往 LQ 靠是负分"。

### #5 cfg=2 + 负提示 + fidelity=0.4（3.7106）

`run_test_cfg2.sh`：同一预训练权重，加 `--cfg 2.0 --fidelity 0.4` 和负提示词 `"blurry, out of focus, low resolution, noisy, jpeg artifacts, oversmoothed, low quality"`。

验证集上无参考指标全部大涨（MUSIQ 42.9→53.2，NIQE 4.92→3.83），但 PSNR 从 26.3 掉到 22.2。榜单只涨了 0.09，说明有参考指标的惩罚是实打实的。正是这个"NR 涨、FR 跌、总分微涨"的样本，让 5 个点足以把两组权重同时解出来。

---

## 二、反推评分公式

代码：`02_score_formula/fit_score.py`（拟合脚本），`score_probe2.py`（对各提交算 NR 指标），`score_pred.py`（用公式给任意输出预测分）。

### 数据

对 5 次提交各算 8 项指标：

- FR：PSNR / SSIM / LPIPS / DISTS，在官方 5 对验证图全分辨率上算（测试集没有 GT）
- NR：MUSIQ / CLIP-IQA / MANIQA / NIQE，在测试集抽 13 张（case1, 9, 17, …, 97）、768×768 中心裁剪上算（`score_probe2.py`，CPU 即可）

全部用 [pyiqa](https://github.com/chaofengc/IQA-PyTorch) 默认设置。指标名来自赛题说明。

### 归一化

把 8 项统一成"越大越好、大致落在 0–1"，各取平均：

```
FRm = mean( PSNR/50,  SSIM,  1-LPIPS,  1-DISTS )
NRm = mean( MUSIQ/100, CLIPIQA, MANIQA, 1-NIQE/10 )
```

### 拟合

```
Score = c0 + c1·FRm + c2·NRm
```

5 个点、3 个未知数，最小二乘：

```
Score = -0.86 + 3.02·FRm + 4.57·NRm      拟合 rmse = 0.014    留一交叉验证 rmse = 0.050
```

对照：只用 FRm 拟合 rmse 0.50，只用 NRm 拟合 rmse 0.12，两者一起 0.014。两组指标缺一不可，系数比约 3 : 4.5。

```
python 02_score_formula/fit_score.py
```

### 事后验证

第 6、7 次提交前先用公式在验证集上预测，再对照榜单：

| 提交 | 公式预测 | 榜单 | 误差 |
|---|---|---|---|
| #6 | 4.130 | 4.118 | -0.012 |
| #7 | 4.631 | 4.535 | -0.096 |

在拟合区间（2.2–3.7）之外外推 0.8 分仍只差 0.1，说明公式形态是对的。高分端略有高估，推测官方对个别指标有截断或用了略不同的归一化。

### 公式给出的结论

- 每单位 NRm 值 4.57 分，每单位 FRm 值 3.02 分。MUSIQ 涨 10 分 ≈ +0.114；PSNR 掉 1 dB ≈ -0.015。**无参考指标是主战场，有参考指标是约束。**
- 4 个 NR 项等权，当时 NIQE（约 4.6）和 MUSIQ（约 50）离上限最远，是优先攻击对象。
- 有了公式，就不必每次都提交榜单：在自己的验证集上算预测分即可。后文 `eval2.py` 每次评测都直接打印这个预测分（表格里的 PRED 列）。

---

## 三、最终提交的生成流程

```
官方 5 对 LQ/GT ──► 标定退化参数 calib.json
                          │
          ┌───────────────┼───────────────┐
          ▼                               ▼
   合成验证集 23 对                配对训练数据 (GT, LQ) 1024 块
          │                               │
          │                     ODTSR 预训练权重
          │                       ──► 阶段一：在线退化 + NR 损失，250 步
          │                       ──► 阶段二：配对数据，3 × 250 步（累计 750 步）
          │                               │
          ▼                               ▼
    评测 eval2.py  ◄────────  推理 100 张测试图 (test_infer.sh)
          ▲                               │
          │                               ▼
          └───────────────  TTO v3：tto3.py（评测同分辨率，逐图梯度优化）
                                          │
                                          ▼
                                  pack_submit.sh ──► 提交 zip
```

### 3.0 环境（`00_env/`）

`env_odtsr.sh` 建 venv（torch 2.x + transformers 4.56.0 + pyiqa + basicsr），`dl.sh` 下载 Qwen-Image、ODTSR 生成器 `weight.pth` 与判别器 `net_dis_iter_5001.pth`。`fix_env*.sh` 是环境问题的修复记录（basicsr 的 `functional_tensor` 补丁、transformers 版本回退）。

### 3.1 退化标定（`03_degradation_calib/`）

前几次训练效果差的根因：合成退化和真实 LQ 不像，模型把真实图当分布外输入。

- `est_kernel.py` / `fit_deg*.py` / `local_fit.py`：在官方 5 对图上反解退化。真实 LQ ≈ 强模糊（等效 σ 6–8 px，远超 Real-ESRGAN 默认上限）+ 大倍率下采样再上采样 + 空间非均匀的对比度/亮度衰减 + 轻 JPEG。
- `gt_stats.py`：官方 GT 本身并不完全清晰，Laplacian 方差明显低于 Unsplash 原图。
- `calib.py`：(A) 求"GT 软化因子" k∈[1.6, 2.4]，让合成 GT 的清晰度统计贴近官方 GT；(B) 对每一对真实图网格搜索 `(前置模糊, 下采样倍率, 对比度 a, 亮度 b, 噪声, JPEG 质量)`，目标是 SSIM/LPIPS/Laplacian 比值同时贴近，而不是 PSNR。输出 `calib.json`，含 5 组拟合参数。

```
python 03_degradation_calib/calib.py
```

### 3.2 合成验证集（`04_synthetic_val/`）

官方只有 5 对验证图，直接在上面调参必然过拟合。`synth_val5.py` 从未进训练集的 Unsplash/Pexels 原图裁 23 张 4K：GT 按 k 软化，LQ 从 `calib.json` 的 5 组参数轮转抽样 + 抖动 + 2×2 空间变化的对比度/亮度场。

可靠性校验（`sval_check.sh`）用两个锚点：合成 LQ 直接对 GT 算公式分 2.32（官方 5 张 2.18）；用提交 #5 的配置跑合成集，公式分 3.3–3.4（官方 5 张 3.45）。两个锚点都对上、8 项指标逐项贴近，才把它当正式验证集。`synth_val.py`–`synth_val4.py` 是没对上的早期版本（Real-ESRGAN 退化过重、按 PSNR 拟合导致纹理过锐等），保留作对照。

`eval2.py` 是通用评测：8 项指标 + FRm / NRm / 公式预测分一行打印，结果追加到 `eval_results.jsonl`；`cmp.py` 在共同子集上对比多组结果。

```
python 04_synthetic_val/synth_val5.py 23
bash   04_synthetic_val/sval_check.sh
python 04_synthetic_val/eval2.py <输出目录> --tag <结果名> --val_dir <验证集目录>
```

### 3.3 训练数据（`05_training/`）

- 抓图：`unsplash_fetch*.py`（Unsplash Lite 元数据筛关键词）、`pexels_fetch.py`、`wiki_fetch.py`，按测试集类别配比（中文招牌 30% / 鸟 20% / 植物 18% / 钟表 15% / 花 7% / 灯光 6% / 建筑 4%）。`clip_filter.py` 用 CLIP 剔除不对题的图。
- `make_tiles.py` + `build_list.py`：切 1024 块、按配比组 40000 块的清单，供阶段一的在线退化训练。
- `make_pairs.py`：按 `calib.json` 离线造 (GT, LQ) 配对块，清单每行 `gt路径|lq路径`。GT 先软化，LQ 用 5 组参数轮转 + 抖动，JPEG q∈[75, 90]。供阶段二。

```
python 05_training/make_pairs.py 2 600      # 每张源图 2 块，每类最多 600 张源图
```

### 3.4 训练

对 ODTSR 训练代码的改动在 `05_training/odtsr_patched/`（以 `TIANCHI_` 注释标记），需覆盖到 ODTSR 原仓库对应位置：

- `train_gan.py`：加 NR 损失（pyiqa `as_loss=True` 的 CLIP-IQA / MUSIQ / MANIQA，多尺度），加 DISTS、SSIM 损失，权重从 yaml 读
- `diffsynth/extensions/realesrgan/dataset.py`：清单行含 `|` 时读真实配对并同位置裁剪，跳过在线退化

`launch_train.sh` 生成 yaml 并启动，参数顺序 `<实验名> <清单> <nr_w> <dists_w> <gan_w> <lpips_w> <rgb_w> [生成器ckpt] [判别器ckpt]`。lr 2e-5，梯度累积 4，bf16，gradient checkpointing，每 250 步存一次。命令里的实验名对应 `configs/` 下同名 yaml。

```
# 阶段一（实验名 nr2）：在线退化 + 强 NR 损失，从预训练权重出发训 250 步，让模型先学会"往评测指标方向生成"
NR_METRICS=clipiqa,musiq,maniqa NR_SCALES=1.0,0.5 SSIM_W=0.75 \
  bash launch_train.sh nr2 list_tiles_40000_clip.txt 3.43 0.75 0.1 0.75 0.05

# 阶段二（实验名 pair1 → pair1b → pair1c）：配对数据，NR 权重降到 1.5，FR 损失加重，每段 250 步接着上一段续训
SSIM_W=1.0 NR_METRICS=clipiqa,musiq,maniqa NR_SCALES=1.0,0.5 \
  bash launch_train.sh pair1  list_pairs.txt 1.5 1.0 0.05 1.0 0.1 <阶段一生成器> <阶段一判别器>
  bash launch_train.sh pair1b list_pairs.txt 1.5 1.0 0.05 1.0 0.1 <pair1 生成器>  <pair1 判别器>
  bash launch_train.sh pair1c list_pairs.txt 1.5 1.0 0.05 1.0 0.1 <pair1b 生成器> <pair1b 判别器>
```

停在累计 750 步的依据：合成验证集上不做 TTO 的公式分，500 步 3.376，750 步 3.412，LPIPS 0.259 → 0.265 持平，FRm 0.729 → 0.728 进入平台。最终使用的权重是阶段二第三段的 `net_gen_iter_251.pth`。

### 3.5 推理（`06_inference/`）

```
bash 06_inference/test_infer.sh <结果名> <生成器ckpt> 1.0
```

`--fidelity 1.0 --tile 64 --scale 1.0`，输出 PNG 到 `output/test_<结果名>/`。100 张 4K 约 100 分钟。

### 3.6 测试时优化 TTO（`07_tto/`）—— 分数从 3.4 到 4.5 的来源

思路：评分里 4 个 NR 指标都是可微网络。对每张输出图 `x0` 做参数化 `x = clamp(x0 + ε·tanh(θ))`，用 Adam 直接最大化 NR 指标，同时用 LPIPS/DISTS/SSIM(x, x0) 约束别偏离原图。不需要 GT，不涉及验证集拟合。

三个版本：

- `tto.py`（v1）：随机 crop 上算 NR，ε=4，60 步。官方 5 张 3.52 → 3.79。
- `tto2.py`（v2，提交 #6，榜单 4.118）：加长边 768 / 1536 的缩小全图视角、NIQE 项、按纹理自适应 ε、bf16、逐项 backward 省显存。参数 `--eps 12 --steps 200 --crop 1024 --crops 6 --w_musiq 3 --w_maniqa 4 --w_clip 2 --w_lpips 2 --w_dists 2 --w_niqe 6 --w_ssim 1 --adapt 0.03`。
- `tto3.py`（v3，提交 #7，榜单 4.5353）：**所有 NR 项改到评测使用的原分辨率与流程上算。**

v3 的关键发现：pyiqa 评测 MUSIQ 时把 4K 原图的全部 32×32 patch（约 1.2 万 token）送进 transformer，CLIP-IQA 也是整张 4K 图直接算。同一张图 MUSIQ 在 768 尺度是 80 分、原图只有 60 分——v1/v2 一直在优化一个和评测不一致的代理目标。让 4K 全图可反传需要：

- `musiq_sdpa.py`：把 MUSIQ 的显式注意力换成 `F.scaled_dot_product_attention`（数值差 3e-5），反向显存 99 GB → 10 GB；CLIP RN50 的 stem / layer1 / layer2 做 gradient checkpoint，14 GB → 5 GB
- MANIQA 用评测同款 `uniform_crop` 20 块 + 20 个随机 224 块；NIQE 全图 + 4 个 512 块
- MANIQA 回归头无上界，直接优化会刷到 2.6–3.8，`--maniqa_cap 0.85` 封顶
- NIQE 内部 SVD 在纯平坦块上会失败：失败则跳过该项；梯度出现 NaN 时清零

```
# 用法：run_tto3_par.sh <输入目录> <输出目录> <结果名> <验证集目录> <并行进程数> [tto3.py 参数...]
bash 07_tto/run_tto3_par.sh output/test_final output/test_final_tto <结果名> <验证集目录> 4 \
     --steps 150 --adapt 0.03 --w_clip 4
python 07_tto/sanity.py output/test_final output/test_final_tto     # 检查有没有整图跑飞（均值/方差对比）
```

其余为默认参数：ε=12/255，lr 0.05，`w_musiq 3 / w_maniqa 4 / w_maniqa_rand 2 / w_niqe 6 / w_niqe_crop 4 / w_lpips 2 / w_dists 2 / w_ssim 1`。单张 4K 约 200 s，4 进程并行 100 张约 5.5 小时。

合成验证集 23 张对比（同一微调模型的输出）：

| | PSNR | SSIM | LPIPS | DISTS | MUSIQ | CLIPIQA | MANIQA | NIQE | FRm | NRm | 公式预测 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 不做 TTO | 29.46 | 0.770 | 0.265 | 0.182 | 37.8 | 0.661 | 0.368 | 5.93 | 0.728 | 0.454 | 3.412 |
| TTO v2 | 28.87 | 0.734 | 0.291 | 0.196 | 50.8 | 0.902 | 0.478 | 3.51 | 0.706 | 0.634 | 4.171 |
| **TTO v3** | 28.85 | 0.724 | 0.279 | 0.192 | **74.2** | 0.899 | **0.840** | 4.21 | 0.708 | **0.765** | **4.772** |

官方 5 张：不做 TTO 3.504 → v2 4.130 → v3 4.631。FR 几乎不动，涨分全部来自 NR。

### 3.7 打包（`08_submit/`）

```
bash 08_submit/pack_submit.sh <TTO 输出目录> <zip 名>
```

zip 根目录为 `case{1..100}.jpg`，JPEG q95 4:4:4，逐张校验尺寸与 LQ 一致，ZIP_STORED。

---

## 目录

```
00_env/                 环境与模型下载
01_early_submissions/   提交 #2–#5 的推理、后处理、融合脚本与训练配置
02_score_formula/       评分公式反推：指标采样、最小二乘拟合、预测
03_degradation_calib/   真实退化反解与参数标定
04_synthetic_val/       合成验证集生成、可靠性校验、通用评测
05_training/            抓图、切块、配对数据、训练启动、ODTSR 补丁、各阶段 yaml
06_inference/           测试集推理
07_tto/                 测试时优化 v1/v2/v3、显存补丁、并行运行、健全性检查
08_submit/              打包
```

## 说明

- 所有脚本按训练机器的 `/workspace` 布局写死路径，换机器需要改。
- 微调后的生成器权重（数 GB）不在仓库内，按 3.4 节可以重新训出来。
- `05_training/odtsr_patched/` 只放改动过的文件，需覆盖到 [ODTSR](https://github.com/double8fun/ODTSR) 原仓库对应位置。
- 脚本中的中文注释是当时的工作记录，保留原样，其中出现的 `nr2` / `pair1` / `sval` / `e8d` / `v7` 之类是实验代号，README 中已对应到阶段名。
