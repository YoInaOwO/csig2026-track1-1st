source /venv/main/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=/workspace/.hf_home
python3 - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import time
t=time.time()
print("== ODTSR weight", flush=True)
hf_hub_download("double8fun/ODTSR", "weight.pth", local_dir="/workspace/models/ODTSR")
print("== FiDeSR ckpt", flush=True)
snapshot_download("jmjin2/FiDeSR", local_dir="/workspace/models/FiDeSR")
print("== SD2.1-base", flush=True)
snapshot_download("Manojb/stable-diffusion-2-1-base", local_dir="/workspace/models/stable-diffusion-2-1-base",
                  ignore_patterns=["*.ckpt", "*.bin", "*.msgpack", "*non_ema*"])
print("== RAM", flush=True)
hf_hub_download("xinyu1205/recognize-anything", "ram_swin_large_14m.pth", repo_type="space", local_dir="/workspace/models/RAM")
print("== Qwen-Image (58GB)", flush=True)
snapshot_download("Qwen/Qwen-Image", local_dir="/workspace/models/Qwen-Image")
print(f"ALL DOWNLOADS DONE in {time.time()-t:.0f}s", flush=True)
PY
du -sh /workspace/models/*
