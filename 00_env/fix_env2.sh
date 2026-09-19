cd /workspace; source venv_odtsr/bin/activate; export PATH=$HOME/.local/bin:$PATH
uv pip list 2>/dev/null | grep -i -E "^(transformers|peft|diffusers|accelerate|tokenizers|modelscope|pyiqa|lightning|torch) "
grep -i -E "transformers|peft|accelerate" ODTSR/requirements.txt
grep -rn "PretrainedConfig" ODTSR/examples/qwen_image/generator.py ODTSR/diffsynth --include=*.py | head -3
uv pip install -q "transformers<5" 2>&1 | tail -1
uv pip list 2>/dev/null | grep -i -E "^transformers "
tmux kill-session -t valbase 2>/dev/null
tmux new-session -d -s valbase 'bash /workspace/run_val_base.sh 2>&1 | tee /workspace/logs/val_base.log'
sleep 120
grep -E 'Error|Traceback' -A2 logs/val_base.log | head; tail -2 logs/val_base.log | cut -c1-200
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
