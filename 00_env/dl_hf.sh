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
