cd /workspace; source venv_odtsr/bin/activate; export PATH=$HOME/.local/bin:$PATH
uv pip install -q "transformers==4.56.0" basicsr 2>&1 | tail -1
python - <<'PY'
import basicsr, os
p = os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py')
s = open(p).read()
if 'functional_tensor' in s and '_functional_tensor' not in s:
    open(p, 'w').write(s.replace('torchvision.transforms.functional_tensor', 'torchvision.transforms._functional_tensor')); print('basicsr patched')
PY
export qwen_path=/workspace/models/Qwen-Image PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image
cd ODTSR/examples/qwen_image
for i in 1 2 3 4 5 6; do
  ERR=$(python -c "from generator import Generator; import pyiqa; print('IMPORT_OK')" 2>&1 | grep -E "IMPORT_OK|Error" | tail -1)
  echo "try $i: $ERR"
  case "$ERR" in
    IMPORT_OK) break;;
    *"No module named"*) M=$(echo "$ERR" | sed -E "s/.*No module named '([^'.]+).*/\1/"); echo "installing $M"; uv pip install -q "$M" 2>&1 | tail -1;;
    *) echo "$ERR"; break;;
  esac
done
cd /workspace
uv pip list 2>/dev/null | grep -i -E "^(transformers|basicsr|torchvision|diffusers) "
tmux kill-session -t valbase 2>/dev/null
tmux new-session -d -s valbase 'bash /workspace/run_val_base.sh 2>&1 | tee /workspace/logs/val_base.log'
sleep 150
grep -E 'Error|Traceback' -A2 logs/val_base.log | head; tail -2 logs/val_base.log | cut -c1-200
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
