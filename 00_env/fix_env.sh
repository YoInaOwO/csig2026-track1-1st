cd /workspace; source venv_odtsr/bin/activate; export PATH=$HOME/.local/bin:$PATH
uv pip install -q modelscope 2>&1 | tail -2
python -c "import modelscope; print('modelscope', modelscope.__version__)"
tmux kill-session -t valbase 2>/dev/null
tmux new-session -d -s valbase 'bash /workspace/run_val_base.sh 2>&1 | tee /workspace/logs/val_base.log'
sleep 90
grep -E 'Error|Traceback' -A2 logs/val_base.log | head; tail -2 logs/val_base.log | cut -c1-200
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
tail -4 logs/tiles_uhd.log
