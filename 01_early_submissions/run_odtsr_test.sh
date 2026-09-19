source /workspace/venv_odtsr/bin/activate
export qwen_path=/workspace/models/Qwen-Image
export PYTHONPATH=/workspace/ODTSR:/workspace/ODTSR/examples/qwen_image
cd /workspace/ODTSR/examples/qwen_image
CUDA_VISIBLE_DEVICES=0 python infer_odtsr.py --input_path /workspace/data/test --output_path /workspace/output/test_odtsr_f1.0 \
  --scale 1.0 --fidelity 1.0 --align_method wavelet --ext png
echo TEST_ODTSR_DONE
