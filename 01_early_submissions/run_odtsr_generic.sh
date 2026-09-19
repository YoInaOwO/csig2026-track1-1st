# 用法: bash run_odtsr_generic.sh <GPU> <out_dir> [extra args...]
GPU=$1; OUT=$2; shift 2
source /workspace/venv_odtsr/bin/activate
export qwen_path=/workspace/models/Qwen-Image
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image
cd /workspace/ODTSR/examples/qwen_image
CUDA_VISIBLE_DEVICES=$GPU python infer_odtsr.py --input_path /workspace/data/val_lq --output_path $OUT --ext png --align_method wavelet "$@"
echo RUN_DONE
