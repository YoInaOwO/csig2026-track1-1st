#!/bin/bash
# 新机一次性搭建：工具 -> 下载 venv -> 四路并行下载 -> ODTSR clone+补丁 -> 训练 venv
set -x
cd /workspace
mkdir -p models data/train data/val data/test data/uhdiqa logs output
export DEBIAN_FRONTEND=noninteractive
which tmux git unzip curl aria2c >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux git unzip curl aria2 >/dev/null)
which uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH=$HOME/.local/bin:$PATH
grep -q '.local/bin' ~/.bashrc || echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

# ---- 下载用轻量 venv ----
uv venv --python 3.11 /workspace/venv_dl -q
source /workspace/venv_dl/bin/activate
uv pip install -q huggingface_hub hf_xet requests pillow numpy opencv-python-headless

# ---- 1) HF 权重 ----
cat > /workspace/dl_hf.sh <<'EOF'
source /workspace/venv_dl/bin/activate
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/workspace/.hf_home
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import time; t=time.time()
print("== ODTSR weight", flush=True); hf_hub_download("double8fun/ODTSR", "weight.pth", local_dir="/workspace/models/ODTSR")
print("== Wan2.1-T2V-1.3B (判别器骨干)", flush=True); snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="/workspace/models/Wan2.1-T2V-1.3B")
print("== Qwen-Image", flush=True); snapshot_download("Qwen/Qwen-Image", local_dir="/workspace/models/Qwen-Image")
print(f"HF_DONE {time.time()-t:.0f}s", flush=True)
PY
du -sh /workspace/models/*
EOF
tmux new-session -d -s dl_hf "bash /workspace/dl_hf.sh 2>&1 | tee /workspace/logs/dl_hf.log"

# ---- 2) UHD-IQA（aria2 多线程）----
tmux new-session -d -s dl_uhd "cd /workspace/data/uhdiqa && aria2c -x 8 -s 8 -c -o UHD-IQA-database.zip https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/UHD-IQA-database.zip 2>&1 | tee /workspace/logs/dl_uhd.log; unzip -q -o UHD-IQA-database.zip && echo UHD_DONE | tee -a /workspace/logs/dl_uhd.log"

# ---- 3) Unsplash Lite 元数据（含 photo_image_url，之后按关键词下原图）----
tmux new-session -d -s dl_uns "mkdir -p /workspace/data/_dl/unsplash && cd /workspace/data/_dl/unsplash && curl -L -o unsplash_lite.zip https://unsplash.com/data/lite/latest 2>&1 | tail -1; unzip -q -o unsplash_lite.zip && ls && echo UNS_META_DONE | tee /workspace/logs/dl_uns.log"

# ---- 4) ODTSR 代码 + 补丁 ----
[ -d /workspace/ODTSR ] || git clone -q https://github.com/RedMediaTech/ODTSR.git /workspace/ODTSR
cd /workspace/ODTSR && git checkout -q 0baa083e2fdca4eda65e30ff29f8dd44658125f6 && cd /workspace
tar xzf /workspace/backup_vast1.tgz -C /tmp
cp /tmp/bk/odtsr_patched/diffsynth_realesrgan/dataset.py ODTSR/diffsynth/extensions/realesrgan/dataset.py
cp /tmp/bk/odtsr_patched/qwen_image/train_gan.py ODTSR/examples/qwen_image/train_gan.py
cp /tmp/bk/odtsr_patched/qwen_image/*.yaml ODTSR/examples/qwen_image/
cp /tmp/bk/odtsr_patched/qwen_image/configs/*.yaml ODTSR/examples/qwen_image/configs/
cp /tmp/bk/odtsr_patched/qwen_image/infer_odtsr.py ODTSR/examples/qwen_image/ 2>/dev/null || cp /workspace/infer_odtsr.py ODTSR/examples/qwen_image/
cp /tmp/bk/scripts/*.py /tmp/bk/scripts/*.sh /workspace/ 2>/dev/null
cp /tmp/bk/uhdiqa_meta/*.csv /workspace/data/uhdiqa/
mkdir -p /workspace/output && cp /tmp/bk/outputs_meta/*.json /workspace/output/
cd /workspace/ODTSR && git status --short | head; cd /workspace

# ---- 5) 训练 venv ----
cat > /workspace/env_odtsr.sh <<'EOF'
set -x
export PATH=$HOME/.local/bin:$PATH
cd /workspace
uv venv --python 3.11 /workspace/venv_odtsr
source /workspace/venv_odtsr/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
cd /workspace/ODTSR
grep -v -E '^(gradio|modelscope|lightning|cupy|controlnet-aux|imageio\[ffmpeg\])' requirements.txt > /tmp/req_odtsr.txt
uv pip install -r /tmp/req_odtsr.txt
uv pip install lightning pyiqa icecream huggingface_hub hf_xet
python -c "import torch, transformers, peft, pyiqa; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), 'transformers', transformers.__version__, 'peft', peft.__version__, 'pyiqa', pyiqa.__version__)"
echo ODTSR_ENV_DONE
EOF
tmux new-session -d -s env "bash /workspace/env_odtsr.sh 2>&1 | tee /workspace/logs/env_odtsr.log"

# ---- Pexels key 验证 ----
KEY=$(cat /workspace/.pexels_key)
curl -s -H "Authorization: $KEY" 'https://api.pexels.com/v1/search?query=bird&per_page=1' | python3 -c "import sys,json;d=json.load(sys.stdin);p=d['photos'][0];print('pexels ok total',d['total_results'],'first',p['width'],'x',p['height'])"
sleep 20; tmux ls; echo SETUP_LAUNCHED
