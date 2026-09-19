# 用法: bash run_train.sh <list_txt> <deg_yaml> <train_yaml>
LIST=${1:-/workspace/data/train/list_smoke.txt}
DEG=${2:-/workspace/ODTSR/examples/qwen_image/deg_tianchi.yaml}
CFG=${3:-/workspace/ODTSR/examples/qwen_image/configs/train_tianchi.yaml}
source /workspace/venv_odtsr/bin/activate
export qwen_path=/workspace/models/Qwen-Image
export wan_path=/workspace/models/Wan2.1-T2V-1.3B
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/ODTSR
accelerate launch --num_processes=2 --gpu_ids="0,1" --main_process_port 29311 --mixed_precision bf16 \
  examples/qwen_image/train_gan.py \
  --mmaigc_dataset_yml "$CFG" \
  --deg_file_path "$DEG" \
  --dataset_txt_paths "$LIST" \
  --null_text_ratio 0.0 \
  --task train
echo TRAIN_EXIT
