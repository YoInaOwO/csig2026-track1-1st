GPU=$1; LIST=$2; DEG=$3; CFG=$4; PORT=$5
source /workspace/venv_odtsr/bin/activate
export qwen_path=/workspace/models/Qwen-Image
export wan_path=/workspace/models/Wan2.1-T2V-1.3B
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/ODTSR
CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port $PORT --mixed_precision bf16 \
  examples/qwen_image/train_gan.py --mmaigc_dataset_yml "$CFG" --deg_file_path "$DEG" --dataset_txt_paths "$LIST" --null_text_ratio 0.0 --task train
echo TRAIN_EXIT
