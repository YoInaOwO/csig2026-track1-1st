cd /workspace; source venv_odtsr/bin/activate; export PATH=$HOME/.local/bin:$PATH
P=/workspace/venv_odtsr/lib/python3.11/site-packages/basicsr/data/degradations.py
sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms._functional_tensor/' $P && grep -n functional_tensor $P
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
tmux kill-session -t valbase 2>/dev/null
tmux new-session -d -s valbase 'bash /workspace/run_val_base.sh 2>&1 | tee /workspace/logs/val_base.log'
sleep 180
grep -E 'Error|Traceback' -A2 logs/val_base.log | head; tail -2 logs/val_base.log | cut -c1-200
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
tail -2 logs/tiles.log; tail -1 logs/pexels.log; tail -1 logs/unsplash.log; tail -1 logs/wiki.log
