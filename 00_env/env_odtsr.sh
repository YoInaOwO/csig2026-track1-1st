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
